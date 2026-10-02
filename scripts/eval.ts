/**
 * Measures what compaction would lose on real Claude Code transcripts.
 *
 *   npx tsx scripts/eval.ts [--model gliner-decide-1b] [--cuts 0.4,0.6,0.8] [--out eval-report.md] file.jsonl...
 *
 * Each transcript is cut at several points and the part before the cut is
 * compacted. A tool output counts as "used later" when one of its distinctive
 * lines (>= 25 chars) reappears after the cut in the assistant's text or tool
 * inputs: the assistant quoted it, or edited code it had read. For every
 * used-later output we check whether that line survived compaction (kept
 * whole, or inside a trimmed head/tail). That is the loss that matters;
 * everything else removed is free space.
 *
 * The label only sees verbatim reuse, so it misses information that guided a
 * decision without being quoted. Treat "lost" as a lower bound.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';

import { GlinerClient } from '../src/client.js';
import { compact } from '../src/compact.js';
import { collectToolCalls, resultText } from '../src/state.js';
import type { CallDecision, Message } from '../src/types.js';

type Block = {
  type: string;
  text?: string;
  name?: string;
  id?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  content?: unknown;
  is_error?: boolean;
};

function blockText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((b: Block) => (b.type === 'text' ? b.text ?? '' : '')).join('\n');
  }
  return '';
}

/** Main-thread user/assistant lines of a Claude Code session JSONL as `Message`s. */
export function loadTranscript(file: string): Message[] {
  const messages: Message[] = [];
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let entry: { type?: string; isSidechain?: boolean; message?: { role?: string; content?: unknown } };
    try {
      entry = JSON.parse(line);
    } catch {
      continue;
    }
    if (entry.isSidechain || (entry.type !== 'user' && entry.type !== 'assistant')) continue;
    const role = entry.message?.role;
    if (role !== 'user' && role !== 'assistant') continue;
    const content = entry.message?.content;
    const message: Message = { role, text: '', toolUses: [] };
    if (typeof content === 'string') message.text = content;
    else if (Array.isArray(content)) {
      const texts: string[] = [];
      for (const block of content as Block[]) {
        if (block.type === 'text' && block.text) texts.push(block.text);
        else if (block.type === 'tool_use' && block.id && block.name) {
          message.toolUses.push({ tool_use_id: block.id, tool: block.name, input: block.input ?? {} });
        } else if (block.type === 'tool_result' && block.tool_use_id) {
          (message.toolResults ??= []).push({
            tool_use_id: block.tool_use_id,
            text: blockText(block.content),
            isError: block.is_error ?? false,
          });
        }
      }
      message.text = texts.join('\n');
    }
    if (message.text || message.toolUses.length || message.toolResults?.length) messages.push(message);
  }
  return messages;
}

/** Distinctive lines of a tool result, with Read's `   12→` gutters removed. */
function distinctiveLines(text: string): string[] {
  const lines = new Set<string>();
  for (const raw of text.split('\n')) {
    const line = raw.replace(/^\s*\d+[→\t]/, '').trim();
    if (line.length >= 25 && /[A-Za-z]{3}/.test(line)) lines.add(line);
  }
  return [...lines];
}

/** Assistant text and tool inputs after the cut, raw and JSON-escaped forms both searchable. */
function futureText(suffix: readonly Message[]): string {
  const parts: string[] = [];
  for (const message of suffix) {
    if (message.role !== 'assistant') continue;
    parts.push(message.text);
    for (const tool of message.toolUses) parts.push(JSON.stringify(tool.input));
  }
  return parts.join('\n');
}

const used = (line: string, future: string): boolean =>
  future.includes(line) || future.includes(JSON.stringify(line).slice(1, -1));

function arg(name: string, fallback: string): string {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] ?? fallback : fallback;
}

interface Row {
  file: string;
  cut: number;
  decision: CallDecision;
  describe: string;
  chars: number;
  usedLater: boolean;
  preserved: boolean;
}

async function main(): Promise<void> {
  const model = arg('model', 'gliner-decide-1b');
  const cuts = arg('cuts', '0.4,0.6,0.8').split(',').map(Number);
  const out = arg('out', 'eval-report.md');
  const flags = new Set(['--model', '--cuts', '--out']);
  const files = process.argv.slice(2).filter((a, i, all) => !a.startsWith('--') && !flags.has(all[i - 1] ?? ''));
  if (files.length === 0) throw new Error('pass one or more transcript .jsonl files');

  const rows: Row[] = [];
  const freed: string[] = [];
  for (const file of files) {
    const all = loadTranscript(file);
    for (const cut of cuts) {
      const at = Math.floor(all.length * cut);
      const prefix = all.slice(0, at);
      const future = futureText(all.slice(at));
      const result = await compact(prefix, new GlinerClient({ model }));
      const after = result.messages
        .flatMap((m) => [m.text, ...(m.toolResults ?? []).map((r) => r.text), ...m.toolUses.map((t) => t.text ?? '')])
        .join('\n');
      const calls = collectToolCalls(prefix, 6);
      const decisions = new Map(result.decisions.map((d) => [d.id, d]));
      for (const call of calls) {
        const decision = decisions.get(call.id)!;
        if (decision.reason === 'pinned') continue;
        const text = resultText(prefix, call);
        const reused = distinctiveLines(text).filter((line) => used(line, future));
        const command = typeof call.input.command === 'string' ? call.input.command : JSON.stringify(call.input);
        rows.push({
          file: basename(file),
          cut,
          decision,
          describe: `${call.tool} ${command.replace(/\s+/g, ' ').slice(0, 100)}`,
          chars: text.length,
          usedLater: reused.length > 0,
          preserved: reused.length > 0 && (decision.action === 'keep' || reused.some((line) => after.includes(line))),
        });
      }
      const s = result.stats;
      freed.push(
        `| ${basename(file).slice(0, 8)} | ${cut} | ${s.messagesBefore} | ${s.calls} | ${s.removed} | ${s.trimmed} | ${s.kept - s.pinned} | ${s.byModel} | ${Math.round((100 * (s.charsBefore - s.charsAfter)) / Math.max(1, s.charsBefore))}% |`,
      );
      console.error(`${basename(file)} cut=${cut}: ${s.calls} calls, ${s.byModel} asked, model ${s.modelMs}ms${s.modelError ? ` (${s.modelError})` : ''}`);
    }
  }

  const usedRows = rows.filter((r) => r.usedLater);
  const lost = usedRows.filter((r) => !r.preserved);
  const lines: string[] = [`# Compaction loss eval (model: ${model})`, ''];
  lines.push(
    `${rows.length} tool outputs outside the pinned messages, over ${files.length} transcripts × ${cuts.length} cut points.`,
    `${usedRows.length} (${((100 * usedRows.length) / rows.length).toFixed(1)}%) were used later. **${lost.length} of them lost their used information** (${((100 * lost.length) / Math.max(1, usedRows.length)).toFixed(0)}%); ${usedRows.length - lost.length} kept it.`,
    '',
  );

  lines.push('## By decision', '');
  lines.push('| Action | Reason | Outputs | Used later | Used info lost | Result chars |');
  lines.push('| --- | --- | ---: | ---: | ---: | ---: |');
  const groups = new Map<string, Row[]>();
  for (const row of rows) {
    const key = `${row.decision.action}|${row.decision.reason}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  for (const [key, group] of [...groups].sort((a, b) => b[1].length - a[1].length)) {
    const [action, reason] = key.split('|');
    lines.push(
      `| ${action} | ${reason} | ${group.length} | ${group.filter((r) => r.usedLater).length} | ${group.filter((r) => r.usedLater && !r.preserved).length} | ${group.reduce((t, r) => t + r.chars, 0)} |`,
    );
  }

  lines.push('', '## Space freed per cut', '');
  lines.push('| Session | Cut | Messages | Calls | Removed | Trimmed | Kept | Asked model | Chars freed |');
  lines.push('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
  lines.push(...freed);

  const summaryEnd = lines.length;
  lines.push('', '## Used-later outputs whose used information was lost', '');
  lines.push('| Session | Cut | Decision | Call |', '| --- | ---: | --- | --- |');
  for (const row of lost) {
    lines.push(`| ${row.file.slice(0, 8)} | ${row.cut} | ${row.decision.action}/${row.decision.reason} | \`${row.describe.replace(/[|`]/g, ' ')}\` |`);
  }
  lines.push('', '## Calls the model decided', '');
  lines.push('| P(useful) | Action | Used later | Call |', '| ---: | --- | --- | --- |');
  for (const row of rows.filter((r) => r.decision.pUseful !== undefined).sort((a, b) => b.decision.pUseful! - a.decision.pUseful!)) {
    lines.push(`| ${row.decision.pUseful!.toFixed(2)} | ${row.decision.action} | ${row.usedLater ? 'yes' : ''} | \`${row.describe.replace(/[|`]/g, ' ')}\` |`);
  }
  writeFileSync(out, `${lines.join('\n')}\n`);
  console.log(lines.slice(0, summaryEnd).join('\n'));
  console.log(`\nfull report (lost outputs, model decisions): ${out}`);
}

if (process.argv[1]?.endsWith('eval.ts')) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
