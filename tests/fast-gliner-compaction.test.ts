import { describe, expect, it } from 'vitest';
import {
  applyDecisions,
  batchCalls,
  buildDecideRequest,
  collectToolCalls,
  compact,
  compactMessages,
  focusText,
  GlinerClient,
  parseDecideResponse,
  QUESTIONS,
  reductionRatio,
  resolveOptions,
  resultText,
  truncatedInput,
  type CallDecision,
  type Decider,
  type Message,
  type ToolCall,
} from '../src/index.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): Message {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

const fileA = 'export const a = 1;\n'.repeat(50);
const testLog = 'PASS a.test.ts\n'.repeat(80);
const scriptOut = 'row\n'.repeat(300);

/** t1 Read (cheap), t2 Bash npm test (expensive), t3 Bash node script (unknown), t4 WebFetch (expensive). */
function transcript(): Message[] {
  return [
    message('user', 'Never edit anything under src/generated. Fix the failing test.'),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'cd app && npm test' }, testLog),
    result('tool-2', testLog),
    call('tool-3', 'Bash', { command: 'node scripts/report.mjs' }, scriptOut),
    result('tool-3', scriptOut),
    call('tool-4', 'WebFetch', { url: 'https://example.com/docs' }, 'docs '.repeat(200)),
    result('tool-4', 'docs '.repeat(200)),
    message('assistant', 'The failure is in b.test.ts; fixing now.'),
    message('user', 'go ahead'),
  ];
}

type Seen = { items: readonly string[]; questions: string[] };

function fakeDecider(answer: (item: string) => number, seen: Seen[] = []): Decider {
  return {
    async decide(items, questions) {
      seen.push({ items, questions: Object.keys(questions) });
      return { answers: items.map((item) => ({ expensive: answer(item) })), ms: 7 };
    },
  };
}

describe('options', () => {
  it('fills in removal-first defaults and ignores non-finite values', () => {
    expect(resolveOptions()).toEqual({
      preserveRecentMessages: 6,
      spareThreshold: 0.85,
      cheapTools: [],
      expensiveTools: [],
      focusResultChars: 1_500,
      maxRequestItems: 128,
      truncateHeadChars: 300,
      truncateInputChars: 300,
    });
    expect(
      resolveOptions({ spareThreshold: Number.NaN, preserveRecentMessages: 2.7, truncateHeadChars: -1.2 }),
    ).toMatchObject({ spareThreshold: 0.85, preserveRecentMessages: 2, truncateHeadChars: 0 });
  });
});

describe('tool call collection', () => {
  it('pairs each tool call with its result and pins the first and recent messages', () => {
    const calls = collectToolCalls(transcript(), 3);
    expect(calls.map((c) => [c.id, c.tool, c.pinned])).toEqual([
      ['t1', 'Read', false],
      ['t2', 'Bash', false],
      ['t3', 'Bash', false],
      ['t4', 'WebFetch', true],
    ]);
    expect(resultText(transcript(), calls[0]!)).toBe(fileA);
  });

  it('ignores calls without a result', () => {
    const messages = [message('user', 'hi'), call('tool-9', 'Read', { file_path: 'x' }, '')];
    expect(collectToolCalls(messages, 0)).toEqual([]);
  });
});

describe('focus text', () => {
  it('shows the tool, its input and the head of its output', () => {
    const calls = collectToolCalls(transcript(), 0);
    const text = focusText(calls[2]!, scriptOut, 40);
    expect(text).toMatch(/^\[tool call\] Bash input=\{"command":"node scripts\/report.mjs"\}\n\[output\] ok, 1200 chars:\n/);
    expect(text).toContain(`${scriptOut.slice(0, 40)}\n[… ${scriptOut.length - 40} more chars]`);
  });
});

describe('batching', () => {
  const calls = Array.from({ length: 10 }, (_, i) => ({ id: `t${i + 1}` })) as ToolCall[];

  it('splits calls into requests of at most maxRequestItems, in order', () => {
    expect(batchCalls(calls, { maxRequestItems: 128 })).toHaveLength(1);
    const batches = batchCalls(calls, { maxRequestItems: 4 });
    expect(batches.map((b) => b.length)).toEqual([4, 4, 2]);
    expect(batches.flat().map((c) => c.id)).toEqual(calls.map((c) => c.id));
  });
});

describe('applying decisions', () => {
  const decide = (id: string, action: CallDecision['action']): CallDecision => ({
    id,
    tool: 'Read',
    action,
    reason: action === 'keep' ? 'expensive' : 'cheap',
    cost: action === 'keep' ? 'expensive' : 'cheap',
  });

  it('truncates dropped results to a head and a note, keeps the call, reuses untouched messages', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const out = applyDecisions(messages, [decide('t1', 'drop_result'), decide('t2', 'keep')], calls, 50);

    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]).not.toBe(messages[1]);
    expect(out[1]?.toolUses[0]).toMatchObject({ tool_use_id: 'tool-1', input: { file_path: 'src/a.ts' } });
    const note = `${fileA.slice(0, 50)}\n[fast-gliner-compaction truncated ${fileA.length - 50} chars of this tool result; re-run the tool if needed]`;
    expect(out[1]?.toolUses[0]?.text).toBe(note);
    expect(out[2]?.toolResults?.[0]?.text).toBe(note);
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('leaves short results and their messages untouched, and honours a zero head', () => {
    const messages = [message('user', 'x'), call('s', 'Read', { file_path: 'a' }, 'short'), result('s', 'short')];
    const calls = collectToolCalls(messages, 0);
    const out = applyDecisions(messages, [decide('t1', 'drop_result')], calls, 300);
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);

    const long = transcript();
    const noHead = applyDecisions(long, [decide('t1', 'drop_result')], collectToolCalls(long, 0), 0);
    expect(noHead[2]?.toolResults?.[0]?.text).toBe(
      `[fast-gliner-compaction truncated ${fileA.length} chars of this tool result; re-run the tool if needed]`,
    );
  });

  it("shortens long string inputs of dropped calls but keeps short fields like paths", () => {
    const content = 'line\n'.repeat(400);
    const input = { file_path: 'src/big.ts', content };
    expect(truncatedInput({ file_path: 'a' }, 300)).toEqual({ file_path: 'a' });
    const short = truncatedInput(input, 100);
    expect(short.file_path).toBe('src/big.ts');
    expect(short.content).toBe(`${content.slice(0, 100)}\n[fast-gliner-compaction truncated ${content.length - 100} chars]`);

    const messages = [message('user', 'x'), call('w', 'Write', input, 'ok'), result('w', 'ok')];
    const calls = collectToolCalls(messages, 0);
    const out = applyDecisions(messages, [{ ...decide('t1', 'drop_result'), tool: 'Write' }], calls, 300, 100);
    expect(out[1]?.toolUses[0]?.input).toEqual(short);
    expect(out[1]?.toolUses[0]?.text).toBe('ok');
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compact', () => {
  it('drops cheap results by rule, spares expensive ones, and asks the model only about the rest', async () => {
    const seen: Seen[] = [];
    const output = await compact(transcript(), fakeDecider(() => 0.2, seen), { preserveRecentMessages: 2 });

    expect(seen).toHaveLength(1);
    expect(seen[0]!.questions).toEqual(Object.keys(QUESTIONS));
    expect(seen[0]!.items).toHaveLength(1);
    expect(seen[0]!.items[0]).toMatch(/^\[tool call\] Bash input=\{"command":"node scripts\/report.mjs"\}/);
    expect(output.decisions.map((d) => [d.id, d.action, d.reason, d.cost])).toEqual([
      ['t1', 'drop_result', 'cheap', 'cheap'],
      ['t2', 'keep', 'expensive', 'expensive'],
      ['t3', 'drop_result', 'model_dropped', 'unknown'],
      ['t4', 'keep', 'expensive', 'expensive'],
    ]);
    expect(output.decisions[2]?.pExpensive).toBe(0.2);
    expect(output.stats).toMatchObject({
      calls: 4,
      kept: 2,
      resultsDropped: 2,
      pinned: 0,
      byRule: 3,
      byModel: 1,
      requests: 1,
      modelMs: 7,
    });
    expect(output.stats.modelError).toBeUndefined();
    expect(reductionRatio(output)).toBeGreaterThan(0.3);
  });

  it('spares an unclassified result when the model reaches spareThreshold', async () => {
    const output = await compact(transcript(), fakeDecider(() => 0.9), { preserveRecentMessages: 2 });
    expect(output.decisions[2]).toMatchObject({ action: 'keep', reason: 'model_spared', pExpensive: 0.9 });
    const strict = await compact(transcript(), fakeDecider(() => 0.9), { preserveRecentMessages: 2, spareThreshold: 0.95 });
    expect(strict.decisions[2]).toMatchObject({ action: 'drop_result', reason: 'model_dropped' });
  });

  it('lets user tool globs override the default rules', async () => {
    const seen: Seen[] = [];
    const output = await compact(transcript(), fakeDecider(() => 0, seen), {
      preserveRecentMessages: 2,
      cheapTools: ['WebFetch'],
      expensiveTools: ['Read'],
    });
    expect(output.decisions.map((d) => d.reason)).toEqual(['expensive', 'expensive', 'model_dropped', 'cheap']);
  });

  it('works on rules alone and keeps unclassified results when the model is unavailable', async () => {
    const rulesOnly = await compact(transcript(), null, { preserveRecentMessages: 2 });
    expect(rulesOnly.decisions[2]).toMatchObject({ action: 'keep', reason: 'model_unavailable' });
    expect(rulesOnly.stats).toMatchObject({ byRule: 3, byModel: 0, requests: 0, modelError: 'no decision model configured' });

    const failing: Decider = { decide: async () => Promise.reject(new Error('connect ECONNREFUSED')) };
    const down = await compact(transcript(), failing, { preserveRecentMessages: 2 });
    expect(down.decisions[2]).toMatchObject({ action: 'keep', reason: 'model_unavailable' });
    expect(down.stats.modelError).toBe('connect ECONNREFUSED');
    expect(down.decisions[0]).toMatchObject({ action: 'drop_result', reason: 'cheap' });
  });

  it('does not call the model when the rules decide everything', async () => {
    const seen: Seen[] = [];
    const messages = [message('user', 'hi'), call('a', 'Read', { file_path: 'x' }, fileA), result('a', fileA), message('assistant', 'ok')];
    const output = await compact(messages, fakeDecider(() => 0, seen), { preserveRecentMessages: 1 });
    expect(seen).toHaveLength(0);
    expect(output.stats).toMatchObject({ byRule: 1, byModel: 0, requests: 0 });
  });

  it('rejects an answer count that does not match the items as a model error', async () => {
    const broken: Decider = { decide: async () => ({ answers: [] }) };
    const output = await compact(transcript(), broken, { preserveRecentMessages: 2 });
    expect(output.stats.modelError).toMatch(/answered 0 of 1/);
  });
});

describe('HTTP client', () => {
  it('builds a decide request against the server URL', () => {
    const request = buildDecideRequest({ serverUrl: 'http://gpu:8765/', token: 'k' }, ['a'], { q: 'x?' });
    expect(request.url).toBe('http://gpu:8765/v1/decide');
    expect(request.headers.authorization).toBe('Bearer k');
    expect(JSON.parse(request.body)).toEqual({ model: 'gliner-decide-1b', questions: { q: 'x?' }, items: ['a'] });

    const local = buildDecideRequest({}, [], { q: 'x?' });
    expect(local.url).toBe('http://127.0.0.1:8765/v1/decide');
    expect(local.headers.authorization).toBeUndefined();
  });

  it('rejects failed, malformed, short and out-of-range responses', () => {
    const q = { q: 'x?' };
    expect(() => parseDecideResponse(500, false, 'boom', 1, q)).toThrow(/500/);
    expect(() => parseDecideResponse(200, true, 'not json', 1, q)).toThrow(/malformed/);
    expect(() => parseDecideResponse(200, true, '{}', 1, q)).toThrow(/missing answers/);
    expect(() => parseDecideResponse(200, true, '{"answers":[]}', 1, q)).toThrow(/answered 0 of 1/);
    expect(() => parseDecideResponse(200, true, '{"answers":[{"q":1.5}]}', 1, q)).toThrow(/invalid/);
    expect(() => parseDecideResponse(200, true, '{"answers":[{}]}', 1, q)).toThrow(/invalid/);
    expect(parseDecideResponse(200, true, '{"answers":[{"q":0.25}],"ms":3}', 1, q)).toEqual({
      answers: [{ q: 0.25 }],
      ms: 3,
    });
  });

  it('asks over fetch and reports server errors through stats', async () => {
    const bodies: string[] = [];
    const client = new GlinerClient({
      serverUrl: 'http://gpu:8765',
      model: 'gliner25-multi',
      fetch: (async (_url: string | URL | Request, init?: RequestInit) => {
        bodies.push(String(init?.body));
        return new Response(JSON.stringify({ answers: [{ q: 0.4 }] }), { status: 200 });
      }) as typeof fetch,
    });
    const response = await client.decide(['state'], { q: 'x?' });
    expect(response.answers).toEqual([{ q: 0.4 }]);
    expect(JSON.parse(bodies[0]!).model).toBe('gliner25-multi');

    const down = (async () => new Response('model not loaded', { status: 409 })) as typeof fetch;
    const output = await compactMessages(transcript(), { fetch: down, preserveRecentMessages: 2 });
    expect(output.stats.modelError).toMatch(/409/);
  });
});
