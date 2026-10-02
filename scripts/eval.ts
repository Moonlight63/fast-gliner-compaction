/**
 * Compares decision models on real Claude Code transcripts.
 *
 *   npx tsx scripts/eval.ts [--models a,b] [--cuts 0.4,0.6,0.8] [--out eval.json] file.jsonl...
 *
 * Each transcript is cut at several points; the prefix is compacted and every
 * candidate tool result is labelled "needed" when one of its distinctive lines
 * (>= 25 chars) reappears in the suffix's assistant text or tool inputs, i.e.
 * the assistant actually used that content after the would-be compaction.
 * This is a proxy label, identical for every model; AUC is threshold-free, so
 * it compares models with different calibration fairly.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';

import { GlinerClient } from '../src/client.js';
import { compact } from '../src/compact.js';
import { collectToolCalls, resultText } from '../src/state.js';
import type { Message, ToolCall } from '../src/types.js';

const MIN_LINE = 25;
const PRESERVE = 6;

type Block = { type: string; text?: string; name?: string; id?: string; input?: Record<string, unknown>; tool_use_id?: string; content?: unknown; is_error?: boolean };

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
    if (line.length >= MIN_LINE && /[A-Za-z]{3}/.test(line)) lines.add(line);
  }
  return [...lines];
}

function futureText(suffix: readonly Message[]): string {
  const parts: string[] = [];
  for (const message of suffix) {
    if (message.role !== 'assistant') continue;
    parts.push(message.text);
    for (const tool of message.toolUses) parts.push(JSON.stringify(tool.input));
  }
  // Tool inputs are JSON-escaped; compare against both raw and escaped forms.
  return parts.join('\n');
}

function isNeeded(result: string, future: string): boolean {
  return distinctiveLines(result).some(
    (line) => future.includes(line) || future.includes(JSON.stringify(line).slice(1, -1)),
  );
}

/** Area under the ROC curve: P(score of a random positive > a random negative). */
function auc(scores: number[], labels: boolean[]): number {
  const pairs = scores.map((s, i) => [s, labels[i]] as const).sort((a, b) => a[0] - b[0]);
  let rankSum = 0;
  let positives = 0;
  for (let i = 0; i < pairs.length; ) {
    let j = i;
    while (j < pairs.length && pairs[j]![0] === pairs[i]![0]) j++;
    const rank = (i + j + 1) / 2;
    for (let k = i; k < j; k++) {
      if (pairs[k]![1]) {
        rankSum += rank;
        positives++;
      }
    }
    i = j;
  }
  const negatives = pairs.length - positives;
  if (positives === 0 || negatives === 0) return Number.NaN;
  return (rankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

/**
 * AUC within result-size quintiles, weighted by positives. The proxy label is
 * confounded with size (a long result has more lines that can reappear), so
 * this is the number to compare; raw AUC rewards "keep the big ones".
 */
function stratifiedAuc(samples: readonly Sample[], score: (s: Sample) => number): number {
  const sorted = [...samples].sort((a, b) => a.chars - b.chars);
  let weighted = 0;
  let weight = 0;
  for (let q = 0; q < 5; q++) {
    const bucket = sorted.slice(Math.floor((q * sorted.length) / 5), Math.floor(((q + 1) * sorted.length) / 5));
    const value = auc(bucket.map(score), bucket.map((s) => s.needed));
    const positives = bucket.filter((s) => s.needed).length;
    if (!Number.isNaN(value)) {
      weighted += value * positives;
      weight += positives;
    }
  }
  return weight === 0 ? Number.NaN : weighted / weight;
}

function arg(name: string, fallback: string): string {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] ?? fallback : fallback;
}

interface Sample {
  file: string;
  cut: number;
  call: string;
  tool: string;
  chars: number;
  position: number;
  needed: boolean;
  scores: Record<string, { keepCall: number; keepResult: number }>;
}

async function main(): Promise<void> {
  const models = arg('models', 'gliner-decide,gliner-decide-1b,laya,laya-multilingual').split(',');
  const cuts = arg('cuts', '0.4,0.6,0.8').split(',').map(Number);
  const out = arg('out', 'eval-results.json');
  const flags = new Set(['--models', '--cuts', '--out']);
  const files = process.argv.slice(2).filter((a, i, all) => !a.startsWith('--') && !flags.has(all[i - 1] ?? ''));
  if (files.length === 0) throw new Error('pass one or more transcript .jsonl files');

  const samples: Sample[] = [];
  const timing: Record<string, { ms: number; modelMs: number; items: number; reduction: number[] }> = {};
  for (const model of models) timing[model] = { ms: 0, modelMs: 0, items: 0, reduction: [] };

  for (const file of files) {
    const messages = loadTranscript(file);
    for (const cut of cuts) {
      const at = Math.floor(messages.length * cut);
      const prefix = messages.slice(0, at);
      const future = futureText(messages.slice(at));
      const candidates = collectToolCalls(prefix, PRESERVE).filter((c: ToolCall) => !c.pinned);
      if (candidates.length === 0) continue;
      const bySample = new Map<string, Sample>();
      candidates.forEach((call, index) => {
        const text = resultText(prefix, call);
        const sample: Sample = {
          file: basename(file),
          cut,
          call: call.id,
          tool: call.tool,
          chars: text.length,
          position: index / Math.max(1, candidates.length - 1),
          needed: isNeeded(text, future),
          scores: {},
        };
        bySample.set(call.id, sample);
        samples.push(sample);
      });
      for (const model of models) {
        const result = await compact(prefix, new GlinerClient({ model }), {
          preserveRecentMessages: PRESERVE,
        });
        timing[model]!.ms += result.stats.ms;
        timing[model]!.modelMs += result.stats.modelMs;
        timing[model]!.items += candidates.length;
        timing[model]!.reduction.push(
          (result.stats.charsBefore - result.stats.charsAfter) / Math.max(1, result.stats.charsBefore),
        );
        for (const decision of result.decisions) {
          const sample = bySample.get(decision.id);
          if (sample) sample.scores[model] = { keepCall: decision.keepCall, keepResult: decision.keepResult };
        }
        console.error(
          `${basename(file)} cut=${cut} ${model}: ${candidates.length} items in ${result.stats.ms}ms`,
        );
      }
    }
  }

  const labels = samples.map((s) => s.needed);
  const positives = labels.filter(Boolean).length;
  const rows: string[] = [];
  rows.push(`samples=${samples.length} needed=${positives} (${((100 * positives) / samples.length).toFixed(1)}% base rate)`);
  rows.push('model                 AUC(size-ctl)  AUC(raw)  AUC(call)  kept-needed@0.5  dropped@0.5  reduction  ms/item');
  const fmt = (value: number, width: number) => (Number.isNaN(value) ? '-' : value.toFixed(3)).padStart(width);
  const pct = (value: number, width: number) => (Number.isNaN(value) ? '-' : `${(value * 100).toFixed(1)}%`).padStart(width);
  const line = (name: string, ctl: number, a: number, b: number, recall: number, dropped: number, red: number, ms: number) =>
    `${name.padEnd(22)}${fmt(ctl, 13)}${fmt(a, 10)}${fmt(b, 11)}${pct(recall, 17)}${pct(dropped, 13)}${pct(red, 11)}${ms.toFixed(1).padStart(9)}`;
  const none = Number.NaN;
  rows.push(line('baseline:recency', stratifiedAuc(samples, (s) => s.position), auc(samples.map((s) => s.position), labels), none, none, none, none, 0));
  rows.push(line('baseline:size(large)', none, auc(samples.map((s) => s.chars), labels), none, none, none, none, 0));
  for (const model of models) {
    const keepResult = samples.map((s) => s.scores[model]?.keepResult ?? 1);
    const keepCall = samples.map((s) => s.scores[model]?.keepCall ?? 1);
    const kept = samples.filter((s, i) => s.needed && keepResult[i]! >= 0.5).length;
    const dropped = keepResult.filter((p) => p < 0.5).length;
    const t = timing[model]!;
    const reduction = t.reduction.reduce((a, b) => a + b, 0) / Math.max(1, t.reduction.length);
    rows.push(
      line(
        model,
        stratifiedAuc(samples, (s) => s.scores[model]?.keepResult ?? 1),
        auc(keepResult, labels),
        auc(keepCall, labels),
        kept / Math.max(1, positives),
        dropped / samples.length,
        reduction,
        t.modelMs / Math.max(1, t.items),
      ),
    );
  }
  console.log(rows.join('\n'));
  writeFileSync(out, JSON.stringify({ models, cuts, files: files.map((f) => basename(f)), timing, samples }, null, 1));
  console.log(`\nper-sample scores written to ${out}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
