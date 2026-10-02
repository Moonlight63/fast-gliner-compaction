import { callTarget, categorize, categoryRules, toolMatcher } from './rules.js';
import { collectToolCalls, focusText, resultText, truncate } from './state.js';
import type {
  CallDecision,
  CompactOptions,
  CompactResult,
  Decider,
  DecisionQuestions,
  Message,
  ResolvedCompactOptions,
  ToolCall,
  ToolResult,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  preserveRecentMessages: 6,
  keepThreshold: 0.85,
  removeTools: [],
  keepTools: [],
  breadcrumbs: true,
  trimHeadChars: 300,
  trimTailChars: 300,
  trimInputChars: 300,
  focusResultChars: 1_500,
  maxRequestItems: 128,
};

/**
 * The one question asked about calls no rule places. It is the same for every
 * item, so the server batches all items through one forward pass.
 */
export const QUESTIONS = {
  useful:
    'Does this output contain specific information that the rest of the task still needs, such as findings, values, error details or answers, which could not be recovered cheaply by re-reading files or re-running a quick command?',
} as const satisfies DecisionQuestions;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function count(value: number | undefined, fallback: number, min: number): number {
  return Math.max(min, Math.floor(finite(value, fallback)));
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  const d = DEFAULT_OPTIONS;
  return {
    preserveRecentMessages: count(options.preserveRecentMessages, d.preserveRecentMessages, 0),
    keepThreshold: finite(options.keepThreshold, d.keepThreshold),
    removeTools: options.removeTools ?? d.removeTools,
    keepTools: options.keepTools ?? d.keepTools,
    breadcrumbs: options.breadcrumbs ?? d.breadcrumbs,
    trimHeadChars: count(options.trimHeadChars, d.trimHeadChars, 0),
    trimTailChars: count(options.trimTailChars, d.trimTailChars, 0),
    trimInputChars: count(options.trimInputChars, d.trimInputChars, 0),
    focusResultChars: count(options.focusResultChars, d.focusResultChars, 0),
    maxRequestItems: count(options.maxRequestItems, d.maxRequestItems, 1),
  };
}

/** Splits calls into requests of at most `maxRequestItems` items. */
export function batchCalls(
  calls: readonly ToolCall[],
  options: Pick<ResolvedCompactOptions, 'maxRequestItems'>,
): ToolCall[][] {
  const batches: ToolCall[][] = [];
  for (let i = 0; i < calls.length; i += options.maxRequestItems) {
    batches.push(calls.slice(i, i + options.maxRequestItems));
  }
  return batches;
}

/** Asks the model P(still useful) for each call, in order. */
export async function askUseful(
  decider: Decider,
  messages: readonly Message[],
  calls: readonly ToolCall[],
  options: Pick<ResolvedCompactOptions, 'focusResultChars' | 'maxRequestItems'>,
): Promise<{ pUseful: Map<string, number>; requests: number; ms: number }> {
  const pUseful = new Map<string, number>();
  let requests = 0;
  let ms = 0;
  // Sequential: one server serialises a model anyway, and this keeps request sizes bounded.
  for (const batch of batchCalls(calls, options)) {
    const items = batch.map((call) => focusText(call, resultText(messages, call), options.focusResultChars));
    const response = await decider.decide(items, QUESTIONS);
    if (response.answers.length !== batch.length) {
      throw new Error(`decider answered ${response.answers.length} of ${batch.length} items`);
    }
    batch.forEach((call, index) => pUseful.set(call.id, response.answers[index]!.useful!));
    requests += 1;
    ms += response.ms ?? 0;
  }
  return { pUseful, requests, ms };
}

/** Ids of calls a later call supersedes: same file, command, URL or query. */
export function supersededCalls(calls: readonly ToolCall[]): Set<string> {
  const superseded = new Set<string>();
  const latest = new Map<string, string>();
  for (const call of calls) {
    const target = callTarget(call);
    if (!target) continue;
    const previous = latest.get(target);
    if (previous) superseded.add(previous);
    latest.set(target, call.id);
  }
  return superseded;
}

/** A trimmed result: its first and last characters around a note. */
export function trimmedText(text: string, isError: boolean, head: number, tail: number): string {
  if (text.length <= head + tail + 120) return text;
  const note = `[fast-gliner-compaction trimmed ${text.length - head - tail} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
  return [head > 0 ? text.slice(0, head) : '', note, tail > 0 ? text.slice(-tail) : '']
    .filter(Boolean)
    .join('\n');
}

/**
 * Long string fields of a trimmed call's input (a long script, a heredoc)
 * shortened to a head and a note; short fields such as paths and commands
 * stay whole. Returns the same object when nothing is long.
 */
export function truncatedInput(input: Record<string, unknown>, inputChars: number): Record<string, unknown> {
  let changed = false;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    if (typeof value === 'string' && value.length > inputChars + 120) {
      out[key] = `${value.slice(0, inputChars)}\n[fast-gliner-compaction truncated ${value.length - inputChars} chars]`;
      changed = true;
    } else out[key] = value;
  }
  return changed ? out : input;
}

/** A short description of a removed call for its trace line. */
function traceEntry(tool: ToolUse): string {
  const target = callTarget(tool);
  const pattern = typeof tool.input.pattern === 'string' ? tool.input.pattern : undefined;
  const what = target ? target.replace(/^(file|cmd|url|query):/, '') : pattern;
  return what ? `${tool.tool} ${truncate(what, 70)}` : tool.tool;
}

const TRACE_MAX_CHARS = 600;

/** One line naming a run of removed calls, repeats folded into `×n`. */
export function traceText(removed: readonly ToolUse[]): string {
  const counts = new Map<string, number>();
  for (const tool of removed) {
    const entry = traceEntry(tool);
    counts.set(entry, (counts.get(entry) ?? 0) + 1);
  }
  const head = `[fast-gliner-compaction removed ${removed.length} tool call${removed.length === 1 ? '' : 's'}: `;
  let body = '';
  let shown = 0;
  for (const [entry, n] of counts) {
    const item = n > 1 ? `${entry} ×${n}` : entry;
    if (head.length + body.length + item.length + 2 > TRACE_MAX_CHARS) break;
    body += (body ? ', ' : '') + item;
    shown += 1;
  }
  const more = counts.size - shown;
  return `${head}${body}${more > 0 ? `, … +${more} more` : ''}]`;
}

/**
 * Rebuilds the conversation from the decisions. Removed calls disappear with
 * their results; messages left empty are dropped, and each run of removed
 * calls is replaced by one trace line (when `breadcrumbs` is on). Trimmed
 * results keep their head and tail. Untouched messages are returned as the
 * same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  options: Pick<ResolvedCompactOptions, 'trimHeadChars' | 'trimTailChars' | 'trimInputChars' | 'breadcrumbs'>,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallDecision['action']>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
  }
  const trim = (text: string, isError: boolean): string =>
    trimmedText(text, isError, options.trimHeadChars, options.trimTailChars);

  const out: Message[] = [];
  let pending: ToolUse[] = [];
  const flush = (): void => {
    if (pending.length > 0 && options.breadcrumbs) {
      out.push({ role: 'assistant', text: traceText(pending), toolUses: [] });
    }
    pending = [];
  };
  for (const message of messages) {
    let changed = false;
    const removedHere: ToolUse[] = [];
    const toolUses: ToolUse[] = [];
    for (const tool of message.toolUses) {
      const action = actions.get(tool.tool_use_id);
      if (action === 'remove') {
        removedHere.push(tool);
        changed = true;
      } else if (action === 'trim') {
        const text = tool.text === undefined ? undefined : trim(tool.text, tool.isError ?? false);
        const input = truncatedInput(tool.input, options.trimInputChars);
        if (text === tool.text && input === tool.input) {
          toolUses.push(tool);
          continue;
        }
        changed = true;
        const copy: ToolUse = { tool_use_id: tool.tool_use_id, tool: tool.tool, input };
        if (text !== undefined) copy.text = text;
        if (tool.isError) copy.isError = true;
        toolUses.push(copy);
      } else toolUses.push(tool);
    }
    const toolResults: ToolResult[] = [];
    for (const result of message.toolResults ?? []) {
      const action = actions.get(result.tool_use_id);
      if (action === 'remove') {
        changed = true;
        continue;
      }
      if (action === 'trim') {
        const text = trim(result.text, result.isError ?? false);
        if (text !== result.text) {
          changed = true;
          toolResults.push({ tool_use_id: result.tool_use_id, text, isError: result.isError });
          continue;
        }
      }
      toolResults.push(result);
    }

    let rebuilt: Message | undefined = message;
    if (changed) {
      if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
        rebuilt = undefined;
      } else {
        rebuilt = { role: message.role, text: message.text, toolUses };
        if (toolResults.length > 0 || message.toolResults) rebuilt.toolResults = toolResults;
      }
    }
    if (rebuilt) {
      // Never put a trace between a tool_use and its tool_result.
      if (!(rebuilt.toolResults && rebuilt.toolResults.length > 0)) flush();
      out.push(rebuilt);
    }
    pending.push(...removedHere);
  }
  flush();
  return out;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

/**
 * Removal-first compaction, judged by how useful each call's information
 * still is. Outside the pinned first and newest messages:
 * - anything a later call supersedes (same file, command, URL or query) is removed;
 * - workspace calls (reads, edits, searches, local shell) are removed, since
 *   the files hold their information;
 * - the latest result of running something is trimmed to its head and tail;
 * - research (web, subagents, the user) is kept;
 * - the model is asked about the rest: kept when P(useful) reaches
 *   `keepThreshold`, otherwise trimmed.
 *
 * `decider` may be null for rules only. If the model cannot be asked, the
 * unplaced results are trimmed and `stats.modelError` says why.
 */
export async function compact(
  messages: readonly Message[],
  decider: Decider | null,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const rules = categoryRules();
  const userKeep = toolMatcher(resolved.keepTools);
  const userRemove = toolMatcher(resolved.removeTools);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const superseded = supersededCalls(calls);
  const categories = new Map(calls.map((call) => [call.id, categorize(call, rules)]));
  const ruled = (call: ToolCall): boolean =>
    call.pinned || userKeep(call.tool) || userRemove(call.tool) || superseded.has(call.id);
  const unknown = calls.filter((call) => !ruled(call) && categories.get(call.id) === 'unknown');

  let asked: Awaited<ReturnType<typeof askUseful>> | undefined;
  let modelError: string | undefined;
  if (unknown.length > 0) {
    if (!decider) modelError = 'no decision model configured';
    else {
      try {
        asked = await askUseful(decider, messages, unknown, resolved);
      } catch (error) {
        modelError = error instanceof Error ? error.message : String(error);
      }
    }
  }

  const decisions = calls.map((call): CallDecision => {
    const category = categories.get(call.id)!;
    const base = { id: call.id, tool: call.tool, category };
    if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
    if (userKeep(call.tool)) return { ...base, action: 'keep', reason: 'user_keep' };
    if (userRemove(call.tool)) return { ...base, action: 'remove', reason: 'user_remove' };
    if (superseded.has(call.id)) return { ...base, action: 'remove', reason: 'superseded' };
    if (category === 'workspace') return { ...base, action: 'remove', reason: 'workspace' };
    if (category === 'outcome') return { ...base, action: 'trim', reason: 'outcome' };
    if (category === 'research') return { ...base, action: 'keep', reason: 'research' };
    const pUseful = asked?.pUseful.get(call.id);
    if (pUseful === undefined) return { ...base, action: 'trim', reason: 'model_unavailable' };
    return pUseful >= resolved.keepThreshold
      ? { ...base, action: 'keep', reason: 'model_keep', pUseful }
      : { ...base, action: 'trim', reason: 'model_trim', pUseful };
  });

  const kept = applyDecisions(messages, decisions, calls, resolved);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);
  const candidates = calls.filter((call) => !call.pinned).length;
  const stats: CompactResult['stats'] = {
    messagesBefore: messages.length,
    messagesAfter: kept.length,
    charsBefore,
    charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
    calls: calls.length,
    kept: decisions.filter((d) => d.action === 'keep').length,
    trimmed: decisions.filter((d) => d.action === 'trim').length,
    removed: decisions.filter((d) => d.action === 'remove').length,
    pinned: decisions.filter((d) => d.reason === 'pinned').length,
    byRule: candidates - unknown.length,
    byModel: asked ? unknown.length : 0,
    requests: asked?.requests ?? 0,
    modelMs: asked?.ms ?? 0,
    ms: Date.now() - started,
  };
  if (modelError) stats.modelError = modelError;
  return { messages: kept, decisions, stats };
}
