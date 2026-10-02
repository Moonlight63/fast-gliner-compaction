/**
 * Evaluates the "is this expensive to re-run?" check on real Claude Code
 * transcripts.
 *
 *   npx tsx scripts/eval.ts [--models a,b] [--threshold 0.7] [--out eval-report.md] file.jsonl...
 *
 * 1. Every tool call is classified by the rules (cheap / expensive / unknown).
 * 2. Rule-labelled calls are a check set: each model sees the call without the
 *    label, and we measure whether its P(expensive) agrees with the rules.
 *    Shell commands are reported separately, because there the model has to
 *    read the command; for named tools the tool name gives a lot away.
 * 3. Unknown calls are what the model decides in practice. They are listed in
 *    the report with each model's score, for a human to eyeball.
 * 4. Each transcript is compacted as a whole, to show how much it frees.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';

import { GlinerClient } from '../src/client.js';
import { askCosts, compact, resolveOptions } from '../src/compact.js';
import { costRules, ruleCost } from '../src/cost.js';
import { collectToolCalls } from '../src/state.js';
import type { Message, RerunCost, ToolCall } from '../src/types.js';

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

/** Area under the ROC curve: P(score of a random positive > a random negative). */
function auc(scores: number[], labels: boolean[]): number {
  const pairs = scores.map((s, i) => [s, labels[i]] as const).sort((a, b) => a[0] - b[0]);
  let rankSum = 0;
  let positives = 0;
  for (let i = 0; i < pairs.length; ) {
    let j = i;
    while (j < pairs.length && pairs[j]![0] === pairs[i]![0]) j++;
    for (let k = i; k < j; k++) {
      if (pairs[k]![1]) {
        rankSum += (i + j + 1) / 2;
        positives++;
      }
    }
    i = j;
  }
  const negatives = pairs.length - positives;
  if (positives === 0 || negatives === 0) return Number.NaN;
  return (rankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

function arg(name: string, fallback: string): string {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] ?? fallback : fallback;
}

function describe(call: ToolCall): string {
  const command = call.input.command;
  const text = typeof command === 'string' ? command : JSON.stringify(call.input);
  return `${call.tool} ${text.replace(/\s+/g, ' ').slice(0, 110)}`;
}

interface Row {
  file: string;
  call: ToolCall;
  shell: boolean;
  rule: RerunCost;
  p: Record<string, number>;
}

async function main(): Promise<void> {
  const models = arg('models', 'gliner25-multi,gliner25-base,gliner-decide-1b').split(',');
  const threshold = Number(arg('threshold', '0.7'));
  const out = arg('out', 'eval-report.md');
  const flags = new Set(['--models', '--threshold', '--out']);
  const files = process.argv.slice(2).filter((a, i, all) => !a.startsWith('--') && !flags.has(all[i - 1] ?? ''));
  if (files.length === 0) throw new Error('pass one or more transcript .jsonl files');

  const rules = costRules();
  const options = resolveOptions();
  const rows: Row[] = [];
  const reduction: string[] = [];
  for (const file of files) {
    const messages = loadTranscript(file);
    const calls = collectToolCalls(messages, 0);
    const fileRows = calls.map((call): Row => ({
      file: basename(file),
      call,
      shell: typeof call.input.command === 'string' || Array.isArray(call.input.commands),
      rule: ruleCost(call, rules),
      p: {},
    }));
    for (const model of models) {
      const { pExpensive } = await askCosts(new GlinerClient({ model }), messages, calls, options);
      for (const row of fileRows) row.p[model] = pExpensive.get(row.call.id)!;
      console.error(`${basename(file)} ${model}: ${calls.length} calls scored`);
    }
    rows.push(...fileRows);
    for (const model of models) {
      const result = await compact(messages, new GlinerClient({ model }), { spareThreshold: threshold });
      const s = result.stats;
      reduction.push(
        `| ${basename(file).slice(0, 8)} | ${model} | ${s.calls} | ${s.byRule} | ${s.byModel} | ${s.resultsDropped} | ${s.kept - s.pinned} | ${Math.round((100 * (s.charsBefore - s.charsAfter)) / Math.max(1, s.charsBefore))}% |`,
      );
    }
  }

  const lines: string[] = [];
  const counts = { cheap: 0, expensive: 0, unknown: 0 };
  for (const row of rows) counts[row.rule]++;
  lines.push(`# Re-run cost eval`, '', `${rows.length} tool calls from ${files.length} transcripts.`);
  lines.push(`Rules: ${counts.cheap} cheap, ${counts.expensive} expensive, ${counts.unknown} unknown (${((100 * counts.unknown) / rows.length).toFixed(1)}% left to the model).`, '');

  lines.push('## Agreement with the rules on rule-labelled calls', '');
  lines.push(`Threshold ${threshold}: "spared" means P(expensive) >= ${threshold}.`, '');
  lines.push('| Model | Subset | n (expensive) | AUC | Cheap correctly dropped | Expensive correctly spared |');
  lines.push('| --- | --- | --- | ---: | ---: | ---: |');
  for (const model of models) {
    for (const [subset, filter] of [
      ['all', (r: Row) => r.rule !== 'unknown'],
      ['shell commands', (r: Row) => r.rule !== 'unknown' && r.shell],
      ['named tools', (r: Row) => r.rule !== 'unknown' && !r.shell],
    ] as const) {
      const set = rows.filter(filter);
      const expensive = set.filter((r) => r.rule === 'expensive');
      const cheap = set.filter((r) => r.rule === 'cheap');
      const a = auc(set.map((r) => r.p[model]!), set.map((r) => r.rule === 'expensive'));
      const dropOk = cheap.filter((r) => r.p[model]! < threshold).length / Math.max(1, cheap.length);
      const spareOk = expensive.filter((r) => r.p[model]! >= threshold).length / Math.max(1, expensive.length);
      lines.push(
        `| ${model} | ${subset} | ${set.length} (${expensive.length}) | ${Number.isNaN(a) ? '-' : a.toFixed(3)} | ${(100 * dropOk).toFixed(0)}% | ${(100 * spareOk).toFixed(0)}% |`,
      );
    }
  }

  lines.push('', '## Threshold sweep on rule-labelled calls', '');
  lines.push('Lowest threshold at which the model drops at least the given share of cheap calls, and how many expensive calls it still spares there.', '');
  lines.push('| Model | Cheap dropped ≥ | Threshold | Expensive spared |');
  lines.push('| --- | ---: | ---: | ---: |');
  for (const model of models) {
    const labelled = rows.filter((r) => r.rule !== 'unknown');
    const cheapScores = labelled.filter((r) => r.rule === 'cheap').map((r) => r.p[model]!).sort((a, b) => a - b);
    const expensive = labelled.filter((r) => r.rule === 'expensive');
    for (const share of [0.8, 0.9, 0.95]) {
      const t = cheapScores[Math.min(cheapScores.length - 1, Math.ceil(share * cheapScores.length) - 1)]! + 1e-9;
      const spared = expensive.filter((r) => r.p[model]! >= t).length / Math.max(1, expensive.length);
      lines.push(`| ${model} | ${share * 100}% | ${t.toFixed(3)} | ${(100 * spared).toFixed(0)}% |`);
    }
  }

  lines.push('', '## Where kept characters go (rules only, largest session)', '');
  {
    const largest = files
      .map((file) => ({ file, messages: loadTranscript(file) }))
      .sort((a, b) => b.messages.length - a.messages.length)[0]!;
    const result = await compact(largest.messages, null, {});
    const calls = collectToolCalls(largest.messages, options.preserveRecentMessages);
    const byReason = new Map<string, { n: number; chars: number }>();
    for (const decision of result.decisions.filter((d) => d.action === 'keep')) {
      const call = calls.find((c) => c.id === decision.id)!;
      const key = `${decision.reason} ${decision.tool.replace(/^mcp__plugin_[^_]+_[^_]+__/, 'mcp:')}`;
      const entry = byReason.get(key) ?? { n: 0, chars: 0 };
      entry.n += 1;
      entry.chars += call.resultChars;
      byReason.set(key, entry);
    }
    lines.push(`${basename(largest.file)}: ${result.stats.charsBefore} chars before, ${result.stats.charsAfter} after.`, '');
    lines.push('| Kept because | Results | Chars |', '| --- | ---: | ---: |');
    for (const [key, { n, chars }] of [...byReason].sort((a, b) => b[1].chars - a[1].chars).slice(0, 12)) {
      lines.push(`| ${key} | ${n} | ${chars} |`);
    }
  }

  lines.push('', '## Whole-transcript compaction', '');
  lines.push('| Session | Model | Calls | By rule | By model | Truncated | Spared | Chars freed |');
  lines.push('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |');
  lines.push(...reduction);

  lines.push('', '## Calls the rules leave to the model', '');
  lines.push(`| ${models.map((m) => `P ${m}`).join(' | ')} | Call |`);
  lines.push(`| ${models.map(() => '---:').join(' | ')} | --- |`);
  for (const row of rows.filter((r) => r.rule === 'unknown').sort((a, b) => b.p[models[0]!]! - a.p[models[0]!]!)) {
    lines.push(`| ${models.map((m) => row.p[m]!.toFixed(2)).join(' | ')} | \`${describe(row.call).replace(/[|`]/g, ' ')}\` |`);
  }
  writeFileSync(out, `${lines.join('\n')}\n`);
  const summaryEnd = lines.indexOf('## Calls the rules leave to the model');
  console.log(lines.slice(0, summaryEnd).join('\n'));
  console.log(`\nfull report with every unclassified call: ${out}`);
}

if (process.argv[1]?.endsWith('eval.ts')) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
