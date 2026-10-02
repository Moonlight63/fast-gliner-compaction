import { costRules, ruleCost } from './cost.js';
import { collectToolCalls, focusText, resultText } from './state.js';
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
  // GLiNER2.5-Decide-1B drops >= 90% of rule-cheap calls from 0.84 up (scripts/eval.ts).
  spareThreshold: 0.85,
  cheapTools: [],
  expensiveTools: [],
  focusResultChars: 1_500,
  maxRequestItems: 128,
  truncateHeadChars: 300,
  truncateInputChars: 300,
};

/**
 * The one question asked about calls no rule classifies. It is the same for
 * every item, so the server batches all items through one forward pass.
 */
export const QUESTIONS = {
  expensive:
    'Would it be expensive to get this output again by re-running the same tool call, because the call is slow, reaches the network or an external service, involves another agent or a person, or captured a one-off result?',
} as const satisfies DecisionQuestions;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function count(value: number | undefined, fallback: number, min: number): number {
  return Math.max(min, Math.floor(finite(value, fallback)));
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  return {
    preserveRecentMessages: count(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages, 0),
    spareThreshold: finite(options.spareThreshold, DEFAULT_OPTIONS.spareThreshold),
    cheapTools: options.cheapTools ?? DEFAULT_OPTIONS.cheapTools,
    expensiveTools: options.expensiveTools ?? DEFAULT_OPTIONS.expensiveTools,
    focusResultChars: count(options.focusResultChars, DEFAULT_OPTIONS.focusResultChars, 0),
    maxRequestItems: count(options.maxRequestItems, DEFAULT_OPTIONS.maxRequestItems, 1),
    truncateHeadChars: count(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars, 0),
    truncateInputChars: count(options.truncateInputChars, DEFAULT_OPTIONS.truncateInputChars, 0),
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

/** Asks the model P(expensive) for each call, in order. */
export async function askCosts(
  decider: Decider,
  messages: readonly Message[],
  calls: readonly ToolCall[],
  options: Pick<ResolvedCompactOptions, 'focusResultChars' | 'maxRequestItems'>,
): Promise<{ pExpensive: Map<string, number>; requests: number; ms: number }> {
  const pExpensive = new Map<string, number>();
  let requests = 0;
  let ms = 0;
  // Sequential: one server serialises a model anyway, and this keeps request sizes bounded.
  for (const batch of batchCalls(calls, options)) {
    const items = batch.map((call) => focusText(call, resultText(messages, call), options.focusResultChars));
    const response = await decider.decide(items, QUESTIONS);
    if (response.answers.length !== batch.length) {
      throw new Error(`decider answered ${response.answers.length} of ${batch.length} items`);
    }
    batch.forEach((call, index) => pExpensive.set(call.id, response.answers[index]!.expensive!));
    requests += 1;
    ms += response.ms ?? 0;
  }
  return { pExpensive, requests, ms };
}

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-gliner-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}

/**
 * Long string fields of a dropped call's input (a Write's file content, an
 * Edit's strings) shortened to a head and a note; short fields such as paths
 * and commands stay whole. Returns the same object when nothing is long.
 */
export function truncatedInput(
  input: Record<string, unknown>,
  inputChars: number,
): Record<string, unknown> {
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

/**
 * Rebuilds the conversation from the decisions. A dropped result keeps a
 * bounded head and a note, and long input fields of its call are shortened
 * the same way; the call itself always stays. Untouched messages are
 * returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
  inputChars: number = Number.POSITIVE_INFINITY,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const dropped = new Set<string>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action === 'drop_result') dropped.add(call.tool_use_id);
  }
  return messages.map((message) => {
    let changed = false;
    const toolUses = message.toolUses.map((tool) => {
      if (!dropped.has(tool.tool_use_id)) return tool;
      const text = truncatedResultText(tool.text ?? '', tool.isError ?? false, headChars);
      const input = truncatedInput(tool.input, inputChars);
      if ((tool.text ?? '') === text && input === tool.input) return tool;
      changed = true;
      const copy: ToolUse = { tool_use_id: tool.tool_use_id, tool: tool.tool, input };
      if (tool.text !== undefined) copy.text = text;
      if (tool.isError) copy.isError = true;
      return copy;
    });
    const toolResults = message.toolResults?.map((result): ToolResult => {
      if (!dropped.has(result.tool_use_id)) return result;
      const text = truncatedResultText(result.text, result.isError ?? false, headChars);
      if (text === result.text) return result;
      changed = true;
      return { tool_use_id: result.tool_use_id, text, isError: result.isError };
    });
    if (!changed) return message;
    const rebuilt: Message = { role: message.role, text: message.text, toolUses };
    if (toolResults) rebuilt.toolResults = toolResults;
    return rebuilt;
  });
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
 * Removal-first compaction. Every tool result outside the pinned first and
 * newest messages is truncated to its head, unless getting it again would be
 * expensive. Rules decide the clear cases (local lookups and edit
 * confirmations are cheap; web, subagents, user answers, tests and builds
 * are expensive); the model is asked only about the rest, and spares a
 * result when P(expensive) reaches `spareThreshold`.
 *
 * `decider` may be null for rules only. If the model cannot be asked, the
 * unclassified results are kept and `stats.modelError` says why.
 */
export async function compact(
  messages: readonly Message[],
  decider: Decider | null,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const rules = costRules(resolved.cheapTools, resolved.expensiveTools);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const costs = new Map(calls.map((call) => [call.id, ruleCost(call, rules)]));
  const unknown = calls.filter((call) => !call.pinned && costs.get(call.id) === 'unknown');

  let asked: Awaited<ReturnType<typeof askCosts>> | undefined;
  let modelError: string | undefined;
  if (unknown.length > 0) {
    if (!decider) modelError = 'no decision model configured';
    else {
      try {
        asked = await askCosts(decider, messages, unknown, resolved);
      } catch (error) {
        modelError = error instanceof Error ? error.message : String(error);
      }
    }
  }

  const decisions = calls.map((call): CallDecision => {
    const cost = costs.get(call.id)!;
    const base = { id: call.id, tool: call.tool, cost };
    if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
    if (cost === 'expensive') return { ...base, action: 'keep', reason: 'expensive' };
    if (cost === 'cheap') return { ...base, action: 'drop_result', reason: 'cheap' };
    const pExpensive = asked?.pExpensive.get(call.id);
    if (pExpensive === undefined) return { ...base, action: 'keep', reason: 'model_unavailable' };
    return pExpensive >= resolved.spareThreshold
      ? { ...base, action: 'keep', reason: 'model_spared', pExpensive }
      : { ...base, action: 'drop_result', reason: 'model_dropped', pExpensive };
  });

  const kept = applyDecisions(
    messages,
    decisions,
    calls,
    resolved.truncateHeadChars,
    resolved.truncateInputChars,
  );
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);
  const candidates = calls.filter((call) => !call.pinned).length;
  const stats: CompactResult['stats'] = {
    messagesBefore: messages.length,
    messagesAfter: kept.length,
    charsBefore,
    charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
    calls: calls.length,
    kept: decisions.filter((d) => d.action === 'keep').length,
    resultsDropped: decisions.filter((d) => d.action === 'drop_result').length,
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
