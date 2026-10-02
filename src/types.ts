export type Role = 'user' | 'assistant';

/**
 * A tool_use block of an assistant message. `text` and `isError` mirror the
 * outcome once the transcript holds it (Claude Code attaches them).
 */
export interface ToolUse {
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  text?: string;
  isError?: boolean;
}

/** A tool_result block of a user message. */
export interface ToolResult {
  tool_use_id: string;
  text: string;
  isError?: boolean;
}

/**
 * One transcript message. The shape is a subset of Claude Code's
 * `SessionMessage`, so a session transcript can be passed in as is.
 */
export interface Message {
  role: Role;
  text: string;
  toolUses: ToolUse[];
  toolResults?: ToolResult[];
}

/** A tool call paired with its result by `tool_use_id`. */
export interface ToolCall {
  /** Short id used in the outline and focus block (`t1`, `t2`, ...). */
  id: string;
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  /** Index of the message holding the tool_use block. */
  callIndex: number;
  /** Index of the message holding the tool_result block. */
  resultIndex: number;
  resultChars: number;
  isError: boolean;
  /** In the first or the newest preserved messages; never a candidate. */
  pinned: boolean;
}

/** How costly it would be to get a call's output again by re-running it. */
export type RerunCost = 'cheap' | 'expensive' | 'unknown';

/** Removal-first: a result is dropped (head kept) unless it is spared. */
export type CallAction = 'keep' | 'drop_result';

export type DecisionReason =
  /** In the first or newest preserved messages. */
  | 'pinned'
  /** A rule says re-running is expensive. */
  | 'expensive'
  /** A rule says re-running is cheap. */
  | 'cheap'
  /** No rule applied; the model judged it expensive. */
  | 'model_spared'
  /** No rule applied; the model judged it cheap. */
  | 'model_dropped'
  /** No rule applied and the model could not be asked; kept to be safe. */
  | 'model_unavailable';

export interface CallDecision {
  id: string;
  tool: string;
  action: CallAction;
  reason: DecisionReason;
  cost: RerunCost;
  /** The model's P(expensive to re-run), when it was asked. */
  pExpensive?: number;
}

export interface CompactOptions {
  /** Newest messages never touched (the first message is always kept). Default 6. */
  preserveRecentMessages?: number;
  /** Minimum model P(expensive) for an unclassified result to be spared. Default 0.7. */
  spareThreshold?: number;
  /** Extra tool-name globs whose output is cheap to get again (win over defaults). */
  cheapTools?: readonly string[];
  /** Extra tool-name globs whose output is expensive to get again (win over defaults). */
  expensiveTools?: readonly string[];
  /** Characters of a call's result shown to the model. Default 1500. */
  focusResultChars?: number;
  /** Decision items sent per server request. Default 128. */
  maxRequestItems?: number;
  /** Characters of a dropped tool result to retain. Default 300. */
  truncateHeadChars?: number;
  /** Characters kept of each long string input field of a dropped call. Default 300. */
  truncateInputChars?: number;
}

export interface ResolvedCompactOptions {
  preserveRecentMessages: number;
  spareThreshold: number;
  cheapTools: readonly string[];
  expensiveTools: readonly string[];
  focusResultChars: number;
  maxRequestItems: number;
  truncateHeadChars: number;
  truncateInputChars: number;
}

export interface CompactResult {
  /** The compacted transcript; untouched messages are the input objects. */
  messages: Message[];
  decisions: CallDecision[];
  stats: {
    messagesBefore: number;
    messagesAfter: number;
    charsBefore: number;
    charsAfter: number;
    calls: number;
    /** Results left verbatim (spared or pinned). */
    kept: number;
    resultsDropped: number;
    pinned: number;
    /** Candidates a rule decided without the model. */
    byRule: number;
    /** Candidates the model was asked about. */
    byModel: number;
    /** Why the model could not be asked, when it could not. */
    modelError?: string;
    requests: number;
    /** Model time the server reported, summed over requests. */
    modelMs: number;
    ms: number;
  };
}

/** Yes/no questions by id; the same questions are asked of every item. */
export type DecisionQuestions = Record<string, string>;

/** P(yes) per question id, one row per item. */
export type DecisionAnswers = Record<string, number>[];

export interface DecideResponse {
  model?: string;
  answers: DecisionAnswers;
  ms?: number;
}

/** Anything that answers yes/no questions over texts: `GlinerClient`, or a host adapter. */
export interface Decider {
  decide(items: readonly string[], questions: DecisionQuestions): Promise<DecideResponse>;
}
