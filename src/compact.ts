import { collectToolCalls, fitState, focusText, renderState, resultText } from './state.js';
import type {
  CallAnswer,
  CallDecision,
  CompactOptions,
  CompactResult,
  Decider,
  DecisionQuestions,
  Message,
  ResolvedCompactOptions,
  ToolCall,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  keepThreshold: 0.5,
  preserveRecentMessages: 6,
  maxStateTokens: 3_000,
  focusResultChars: 1_500,
  maxRequestItems: 128,
  truncateHeadChars: 300,
};

/**
 * The two yes/no questions asked of every item. They name "the focused tool
 * call" rather than an id so that every item shares one question set, which
 * lets the server batch all items through one forward pass.
 */
export const QUESTIONS = {
  keep_call:
    'Should the focused tool call stay in the history, because knowing that this call was made, with its input, still matters for what the assistant does next?',
  keep_result:
    'Does the assistant still need the full output of the focused tool call verbatim, so that re-running the tool would not do?',
} as const satisfies DecisionQuestions;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepThreshold: finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold),
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    focusResultChars: Math.max(
      0,
      Math.floor(finite(options.focusResultChars, DEFAULT_OPTIONS.focusResultChars)),
    ),
    maxRequestItems: Math.max(
      1,
      Math.floor(finite(options.maxRequestItems, DEFAULT_OPTIONS.maxRequestItems)),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
  };
}

/**
 * One decision item per candidate call: its focus block first (so a model
 * that truncates loses outline, never the call), then the shared outline.
 */
export function decisionItems(
  messages: readonly Message[],
  candidates: readonly ToolCall[],
  outline: string,
  options: Pick<ResolvedCompactOptions, 'focusResultChars'>,
): string[] {
  return candidates.map(
    (call) =>
      `${focusText(call, resultText(messages, call), options.focusResultChars)}\n\n${outline}`,
  );
}

/** Splits the candidate calls into requests of at most `maxRequestItems` items. */
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

export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned'>,
  answer: CallAnswer,
  options: Pick<ResolvedCompactOptions, 'keepThreshold'>,
): CallDecision {
  const base = { id: call.id, tool: call.tool, ...answer };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (answer.keepResult >= options.keepThreshold) {
    return { ...base, action: 'keep', reason: 'kept' };
  }
  if (answer.keepCall >= options.keepThreshold) {
    return { ...base, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, action: 'drop_call', reason: 'call_dropped' };
}

async function askBatch(
  decider: Decider,
  messages: readonly Message[],
  batch: readonly ToolCall[],
  outline: string,
  options: Pick<ResolvedCompactOptions, 'focusResultChars'>,
): Promise<{ answers: Map<string, CallAnswer>; ms: number }> {
  const items = decisionItems(messages, batch, outline, options);
  const response = await decider.decide(items, QUESTIONS);
  if (response.answers.length !== batch.length) {
    throw new Error(`decider answered ${response.answers.length} of ${batch.length} items`);
  }
  return {
    answers: new Map(
      batch.map((call, index) => {
        const row = response.answers[index]!;
        return [call.id, { keepCall: row.keep_call!, keepResult: row.keep_result! }];
      }),
    ),
    ms: response.ms ?? 0,
  };
}

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-gliner-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}

/**
 * Rebuilds the conversation from the decisions. A dropped call disappears
 * together with its result; a dropped result keeps a bounded head and note.
 * Messages that lose all their content are removed; untouched messages are
 * returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallDecision['action']>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
  }
  const kept: Message[] = [];
  for (const message of messages) {
    const touched =
      message.toolUses.some((tool) => actions.has(tool.tool_use_id)) ||
      (message.toolResults ?? []).some((result) => actions.has(result.tool_use_id));
    if (!touched) {
      kept.push(message);
      continue;
    }
    const toolUses = message.toolUses
      .filter((tool) => actions.get(tool.tool_use_id) !== 'drop_call')
      .map((tool) => {
        if (actions.get(tool.tool_use_id) !== 'drop_result') return tool;
        const text = truncatedResultText(
          tool.text ?? '',
          tool.isError ?? false,
          headChars,
        );
        if ((tool.text ?? '') === text) return tool;
        const copy: ToolUse = {
          tool_use_id: tool.tool_use_id,
          tool: tool.tool,
          input: tool.input,
          text,
        };
        if (tool.isError) copy.isError = true;
        return copy;
      });
    const toolResults = (message.toolResults ?? [])
      .filter((result) => actions.get(result.tool_use_id) !== 'drop_call')
      .map((result) => {
        if (actions.get(result.tool_use_id) !== 'drop_result') return result;
        const text = truncatedResultText(result.text, result.isError ?? false, headChars);
        return text === result.text
          ? result
          : {
              tool_use_id: result.tool_use_id,
              text,
              isError: result.isError,
            };
      });
    if (
      !message.toolUses.some(
        (tool) => actions.get(tool.tool_use_id) === 'drop_call',
      ) &&
      !(message.toolResults ?? []).some(
        (result) => actions.get(result.tool_use_id) === 'drop_call',
      ) &&
      toolUses.every((tool, index) => tool === message.toolUses[index]) &&
      toolResults.every(
        (result, index) => result === message.toolResults?.[index],
      )
    ) {
      kept.push(message);
      continue;
    }
    if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
      continue;
    }
    const rebuilt: Message = { role: message.role, text: message.text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  return kept;
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

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

/**
 * Compacts a transcript by asking the decider, for every tool call outside
 * the pinned first and newest messages, whether the call and whether its
 * result must stay. Each call becomes one item: its focus block plus the
 * shared outline of the whole history (results omitted, fitted into
 * `maxStateTokens`). Throws when the decider fails or the history cannot be
 * fitted; the caller decides whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  decider: Decider,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const candidates = calls.filter((call) => !call.pinned);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  let fitted: { tokens: number; stage: string } = { tokens: 0, stage: '' };
  let batches: ToolCall[][] = [];
  let modelMs = 0;
  const answers = new Map<string, CallAnswer>();
  if (candidates.length > 0) {
    const state = fitState(messages, calls, resolved);
    fitted = state;
    const outline = renderState(state.state);
    batches = batchCalls(candidates, resolved);
    // Sequential: one server serialises a model anyway, and this keeps request sizes bounded.
    for (const batch of batches) {
      const answered = await askBatch(decider, messages, batch, outline, resolved);
      modelMs += answered.ms;
      for (const [id, answer] of answered.answers) answers.set(id, answer);
    }
  }

  const decisions = calls.map((call) =>
    decideCall(call, answers.get(call.id) ?? { keepCall: 1, keepResult: 1 }, resolved),
  );
  const kept = applyDecisions(
    messages,
    decisions,
    calls,
    resolved.truncateHeadChars,
  );
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      pinned: count(decisions, 'pinned'),
      stateTokens: fitted.tokens,
      stateStage: fitted.stage,
      requests: batches.length,
      modelMs,
      ms: Date.now() - started,
    },
  };
}
