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

/**
 * Where a call's information lives, which decides how useful it stays:
 * - `workspace`: in the files, or a quick local lookup away (reads, edits, searches);
 * - `outcome`: the result of running something (tests, builds, scripts, remote);
 * - `research`: found outside the workspace (web, subagents, the user);
 * - `unknown`: no rule applies.
 */
export type CallCategory = 'workspace' | 'outcome' | 'research' | 'unknown';

/**
 * - `remove`: the call and its result go, leaving a one-line trace;
 * - `trim`: the call stays, its result is cut to its first and last lines;
 * - `keep`: untouched.
 */
export type CallAction = 'remove' | 'trim' | 'keep';

export type DecisionReason =
  /** In the first or newest preserved messages. */
  | 'pinned'
  /** Matched the user's `keepTools`. */
  | 'user_keep'
  /** Matched the user's `removeTools`. */
  | 'user_remove'
  /** A later call reads, edits, re-runs or re-fetches the same target. */
  | 'superseded'
  /** The information is in the workspace. */
  | 'workspace'
  /** The latest result of running something: trimmed to its head and tail. */
  | 'outcome'
  /** Found outside the workspace and not repeated since: kept. */
  | 'research'
  /** No rule applied; the model judged the details still useful. */
  | 'model_keep'
  /** No rule applied; the model judged them not needed. */
  | 'model_trim'
  /** No rule applied and the model could not be asked: trimmed. */
  | 'model_unavailable';

export interface CallDecision {
  id: string;
  tool: string;
  action: CallAction;
  reason: DecisionReason;
  category: CallCategory;
  /** The model's P(still useful), when it was asked. */
  pUseful?: number;
}

export interface CompactOptions {
  /** Newest messages never touched (the first message is always kept). Default 6. */
  preserveRecentMessages?: number;
  /** Minimum model P(still useful) for an unclassified result to be kept whole. Default 0.85. */
  keepThreshold?: number;
  /** Tool-name globs always removed (win over the rules). */
  removeTools?: readonly string[];
  /** Tool-name globs always kept (win over the rules and `removeTools`). */
  keepTools?: readonly string[];
  /** Leave a one-line trace where calls were removed. Default true. */
  breadcrumbs?: boolean;
  /** Characters kept from the start of a trimmed result. Default 300. */
  trimHeadChars?: number;
  /** Characters kept from the end of a trimmed result. Default 300. */
  trimTailChars?: number;
  /** Characters kept of each long string input field of a trimmed call. Default 300. */
  trimInputChars?: number;
  /** Characters of a call's result shown to the model. Default 1500. */
  focusResultChars?: number;
  /** Decision items sent per server request. Default 128. */
  maxRequestItems?: number;
}

export interface ResolvedCompactOptions {
  preserveRecentMessages: number;
  keepThreshold: number;
  removeTools: readonly string[];
  keepTools: readonly string[];
  breadcrumbs: boolean;
  trimHeadChars: number;
  trimTailChars: number;
  trimInputChars: number;
  focusResultChars: number;
  maxRequestItems: number;
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
    /** Calls left untouched (kept or pinned). */
    kept: number;
    trimmed: number;
    removed: number;
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
