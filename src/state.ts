import type { Message, ToolCall, ToolResult } from './types.js';

const INPUT_CHARS = 1000;

export function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, Math.max(0, limit - 1))}…`;
}

export function isPinned(
  index: number,
  total: number,
  preserveRecentMessages: number,
): boolean {
  return index === 0 || index >= total - preserveRecentMessages;
}

/**
 * Pairs every tool_use with its tool_result by `tool_use_id`. Calls without a
 * result are not candidates (there is nothing to drop yet).
 */
export function collectToolCalls(
  messages: readonly Message[],
  preserveRecentMessages: number,
): ToolCall[] {
  const results = new Map<string, { index: number; result: ToolResult }>();
  messages.forEach((message, index) => {
    for (const result of message.toolResults ?? []) {
      results.set(result.tool_use_id, { index, result });
    }
  });
  const calls: ToolCall[] = [];
  messages.forEach((message, callIndex) => {
    for (const tool of message.toolUses) {
      const found = results.get(tool.tool_use_id);
      if (!found) continue;
      calls.push({
        id: `t${calls.length + 1}`,
        tool_use_id: tool.tool_use_id,
        tool: tool.tool,
        input: tool.input,
        callIndex,
        resultIndex: found.index,
        resultChars: found.result.text.length,
        isError: found.result.isError ?? false,
        pinned:
          isPinned(callIndex, messages.length, preserveRecentMessages) ||
          isPinned(found.index, messages.length, preserveRecentMessages),
      });
    }
  });
  return calls;
}

function inputText(input: Record<string, unknown>, limit: number): string {
  let json = '';
  try {
    json = JSON.stringify(input);
  } catch {
    json = '[unserializable input]';
  }
  return truncate(json, limit);
}

/** The text of a call's tool result, '' when it cannot be found. */
export function resultText(messages: readonly Message[], call: ToolCall): string {
  return (
    messages[call.resultIndex]?.toolResults?.find(
      (result) => result.tool_use_id === call.tool_use_id,
    )?.text ?? ''
  );
}

/** What the model sees about one call: the tool, its input, and the head of its output. */
export function focusText(call: ToolCall, result: string, resultChars: number): string {
  const head =
    result.length <= resultChars
      ? result
      : `${result.slice(0, resultChars)}\n[… ${result.length - resultChars} more chars]`;
  return [
    `[tool call] ${call.tool} input=${inputText(call.input, INPUT_CHARS)}`,
    `[output] ${call.isError ? 'error' : 'ok'}, ${call.resultChars} chars:`,
    head,
  ].join('\n');
}
