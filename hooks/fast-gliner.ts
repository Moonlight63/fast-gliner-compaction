import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import {
  buildDecideRequest,
  DEFAULT_MODEL,
  DEFAULT_SERVER_URL,
  parseDecideResponse,
} from '../src/request.js';
import type {
  CompactOptions,
  CompactResult,
  Decider,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
  serverUrl: DEFAULT_SERVER_URL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  serverUrl: string;
  serverToken?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** A comma-separated tool glob list option, or undefined when unset. */
function optionList(options: PluginOptions, key: string): string[] | undefined {
  const list = optionString(options, key)
    ?.split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  return list && list.length > 0 ? list : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Pick<
    CompactOptions,
    | 'keepThreshold'
    | 'preserveRecentMessages'
    | 'focusResultChars'
    | 'maxRequestItems'
    | 'trimHeadChars'
    | 'trimTailChars'
    | 'trimInputChars'
  > = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'focusResultChars',
    'maxRequestItems',
    'trimHeadChars',
    'trimTailChars',
    'trimInputChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    serverUrl: optionString(options, 'serverUrl') ?? HOOK_DEFAULTS.serverUrl,
  };
  const serverToken = optionString(options, 'serverToken');
  if (serverToken) config.serverToken = serverToken;
  const removeTools = optionList(options, 'removeTools');
  if (removeTools) config.removeTools = removeTools;
  const keepTools = optionList(options, 'keepTools');
  if (keepTools) config.keepTools = keepTools;
  if (typeof options.breadcrumbs === 'boolean') config.breadcrumbs = options.breadcrumbs;
  return config;
}

/** A `Decider` over the engine's `$.http.fetch`. */
export function serverDecider(
  fetchFn: HookFetch,
  params: { serverUrl: string; token?: string; model: string },
): Decider {
  return {
    async decide(items, questions) {
      const request = buildDecideRequest(params, items, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseDecideResponse(
        response.status,
        response.ok,
        response.text,
        items.length,
        questions,
      );
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/**
 * Runs the library over a session transcript. A failing decision server does
 * not throw: unplaced results are trimmed and `result.stats.modelError` says why.
 */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
): Promise<SessionCompaction> {
  const decider = serverDecider(fetchFn, {
    serverUrl: config.serverUrl,
    token: config.serverToken,
    model: config.model,
  });
  const result = await compact(messages, decider, config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const kept = stats.kept - stats.pinned;
  const parts = [
    stats.removed > 0 ? `${stats.removed} removed` : '',
    stats.trimmed > 0 ? `${stats.trimmed} trimmed` : '',
    kept > 0 ? `${kept} kept` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  const model = stats.modelError
    ? `model unavailable, ${stats.calls - stats.pinned - stats.byRule} unplaced trimmed (${stats.modelError})`
    : `${stats.byRule} by rule, ${stats.byModel} by model in ${stats.modelMs}ms`;
  return `${percent(reductionRatio(result))} reduction; ${parts.join(', ') || 'no tool calls'}; ${model}`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/${d.reason}${d.pUseful === undefined ? '' : `=${d.pUseful.toFixed(2)}`}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

type EnvAccess = {
  env: { get: (name: string) => Promise<string | undefined> };
  settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
};

/** A settings.json `env` value, for variables the process was started without. */
function settingsEnv(settings: Readonly<Record<string, unknown>>, name: string): string | undefined {
  const env = settings['env'];
  if (!env || typeof env !== 'object') return undefined;
  const value = (env as Record<string, unknown>)[name];
  return typeof value === 'string' && value ? value : undefined;
}

/**
 * Plugin options win; `FGC_SERVER_URL` / `FGC_TOKEN` fill in what they leave
 * unset (process environment, then settings.json `env`), so a remote GPU host
 * can be configured once. `$.env.get` takes literal names only: the engine
 * lists the variables a module reads from them.
 */
async function withServerEnv(
  $: EnvAccess,
  options: PluginOptions,
  config: HookConfig,
): Promise<HookConfig> {
  const resolved = { ...config };
  // Only a fallback source: compaction must not fail because settings are unreadable.
  const settings = await $.settings.read().catch(() => ({}));
  if (!optionString(options, 'serverUrl')) {
    resolved.serverUrl =
      (await $.env.get('FGC_SERVER_URL')) ?? settingsEnv(settings, 'FGC_SERVER_URL') ?? config.serverUrl;
  }
  if (!config.serverToken) {
    const token = (await $.env.get('FGC_TOKEN')) ?? settingsEnv(settings, 'FGC_TOKEN');
    if (token) resolved.serverToken = token;
  }
  return resolved;
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;

  on('session.compact', async ($, event, next) => {
    try {
      const config = await withServerEnv($, options, configured);
      const { result, messages } = await compactSession(event.messages, config, async (url, init) => {
        const response = await $.http.fetch(url, init);
        return { status: response.status, ok: response.ok, text: response.text };
      });
      for (const line of decisionLogLines(result)) $.ui.log(line);
      if (reductionRatio(result) < config.minReductionRatio) {
        notify(
          $,
          `fallback to built-in summary (below ${percent(config.minReductionRatio)} minimum: ${summarize(result)})`,
        );
        return next(event);
      }
      notify(
        $,
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
      );
      return { messages };
    } catch (error) {
      notify(
        $,
        `fallback to built-in summary (${error instanceof Error ? error.message : String(error)})`,
      );
      return next(event);
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const { context } = await $.session.usage();
      if ((context.percent ?? 0) < configured.compactAtPercent) return next(event);
      compacting = true;
      await $.session.compact();
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
