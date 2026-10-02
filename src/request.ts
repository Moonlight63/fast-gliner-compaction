import type { DecideResponse, DecisionQuestions } from './types.js';

export const DEFAULT_SERVER_URL = 'http://127.0.0.1:8765';
export const DEFAULT_MODEL = 'gliner-decide';

export interface DecideRequest {
  url: string;
  method: 'POST';
  headers: Record<string, string>;
  body: string;
}

/** The HTTP request for one decision batch, for any fetch-like transport. */
export function buildDecideRequest(
  params: {
    serverUrl?: string;
    token?: string;
    model?: string;
  },
  items: readonly string[],
  questions: DecisionQuestions,
): DecideRequest {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (params.token) headers.authorization = `Bearer ${params.token}`;
  return {
    url: `${(params.serverUrl ?? DEFAULT_SERVER_URL).replace(/\/+$/, '')}/v1/decide`,
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: params.model ?? DEFAULT_MODEL,
      questions,
      items,
    }),
  };
}

/**
 * Validates a decision response: one row per item, a finite probability in
 * [0, 1] for every question. Throws on anything else.
 */
export function parseDecideResponse(
  status: number,
  ok: boolean,
  text: string,
  itemCount: number,
  questions: DecisionQuestions,
): DecideResponse {
  if (!ok) {
    throw new Error(`decision server request failed (${status}): ${text.slice(0, 200)}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error('decision server returned malformed JSON');
  }
  if (
    parsed === null ||
    typeof parsed !== 'object' ||
    !('answers' in parsed) ||
    !Array.isArray(parsed.answers)
  ) {
    throw new Error('decision server response is missing answers');
  }
  const answers = parsed.answers as unknown[];
  if (answers.length !== itemCount) {
    throw new Error(`decision server answered ${answers.length} of ${itemCount} items`);
  }
  for (const row of answers) {
    for (const id of Object.keys(questions)) {
      const value = (row as Record<string, unknown> | null)?.[id];
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
        throw new Error(`invalid decision server answer for ${id}`);
      }
    }
  }
  return parsed as DecideResponse;
}
