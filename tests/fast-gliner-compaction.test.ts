import { describe, expect, it } from 'vitest';
import {
  applyDecisions,
  batchCalls,
  buildDecideRequest,
  collectToolCalls,
  compact,
  compactMessages,
  decideCall,
  decisionItems,
  estimateTokens,
  fitState,
  GlinerClient,
  parseDecideResponse,
  QUESTIONS,
  reductionRatio,
  renderState,
  resolveOptions,
  type Decider,
  type HistoryToolCall,
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
const fileB = 'export const b = 2;\n'.repeat(50);

function transcript(): Message[] {
  return [
    message('user', 'Never edit anything under src/generated. Fix the failing test.'),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    message('assistant', 'a.ts looks fine; checking b.ts'),
    call('tool-2', 'Read', { file_path: 'src/b.ts' }, fileB),
    result('tool-2', fileB),
    call('tool-3', 'Bash', { command: 'npm test' }, 'FAIL b.test.ts'),
    result('tool-3', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'The failure is in b.test.ts; fixing now.'),
    message('user', 'go ahead'),
  ];
}

type Seen = { items: readonly string[]; questions: string[] };

function fakeDecider(answer: (question: string, item: string) => number, seen: Seen[] = []): Decider {
  return {
    async decide(items, questions) {
      seen.push({ items, questions: Object.keys(questions) });
      return {
        answers: items.map((item) =>
          Object.fromEntries(Object.keys(questions).map((key) => [key, answer(key, item)])),
        ),
        ms: 7,
      };
    },
  };
}

const fit = {
  maxStateTokens: 25_000,
  preserveRecentMessages: 0,
  goal: 'fix the test',
};

describe('options', () => {
  it('fills in defaults and ignores non-finite values', () => {
    expect(resolveOptions()).toMatchObject({
      keepThreshold: 0.5,
      preserveRecentMessages: 6,
      maxStateTokens: 3_000,
      focusResultChars: 1_500,
      maxRequestItems: 128,
      truncateHeadChars: 300,
    });
    expect(resolveOptions({
      keepThreshold: Number.NaN,
      preserveRecentMessages: 2.7,
      truncateHeadChars: -1.2,
    })).toMatchObject({
      keepThreshold: 0.5,
      preserveRecentMessages: 2,
      truncateHeadChars: 0,
    });
  });
});

describe('token estimate', () => {
  it('charges words, digits and symbols separately and never undercounts JSON badly', () => {
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens('hello world')).toBe(2);
    expect(estimateTokens('internationalization')).toBe(4);
    expect(estimateTokens('12345678')).toBe(4);
    const json = JSON.stringify({ file_path: '/Users/x/src/a.ts', old_string: 'a = 1;', n: 42 });
    expect(estimateTokens(json)).toBeGreaterThanOrEqual(Math.ceil(json.length / 3));
  });
});

describe('tool call collection', () => {
  it('pairs each tool call with its result and pins recent ones', () => {
    const calls = collectToolCalls(transcript(), 3);
    expect(calls.map((c) => [c.id, c.tool, c.callIndex, c.resultIndex, c.pinned])).toEqual([
      ['t1', 'Read', 1, 2, false],
      ['t2', 'Read', 4, 5, false],
      ['t3', 'Bash', 6, 7, true],
    ]);
    expect(calls[2]?.isError).toBe(true);
    expect(calls[0]?.resultChars).toBe(fileA.length);
  });

  it('ignores calls without a result', () => {
    expect(collectToolCalls([message('user', 'hi'), call('x', 'Read', {}, '')], 0)).toHaveLength(0);
  });
});

describe('state fitting', () => {
  it('sends the whole history with tool results replaced by a note', () => {
    const messages = transcript();
    const { state, stage } = fitState(messages, collectToolCalls(messages, 0), fit);
    expect(stage).toBe('full');
    const json = JSON.stringify(state);
    expect(json).not.toContain('export const a = 1;');
    expect(json).toContain('Never edit anything under src/generated');
    expect(json).toContain('go ahead');
    expect(state.history.map((entry) => entry.i)).toEqual([0, 1, 3, 4, 6, 8, 9]);
    expect(state.history[1]?.tool_calls?.[0]).toMatchObject({
      id: 't1',
      tool: 'Read',
      result: `ok, ${fileA.length} chars (omitted)`,
    });
    expect((state.history[4]?.tool_calls?.[0] as HistoryToolCall).result).toMatch(/^error, /);
  });

  it('defaults the goal to the latest user prompts', () => {
    const { state } = fitState(transcript(), [], { ...fit, goal: '' });
    expect(state.goal).toContain('Fix the failing test');
    expect(state.goal).toContain('go ahead');
  });

  it('truncates tool inputs before touching message text', () => {
    const messages = [
      message('user', 'start'),
      call('w', 'Write', { file_path: 'x.ts', content: 'x'.repeat(5000) }, 'ok'),
      result('w', 'ok'),
      message('assistant', 'written'),
    ];
    const { state, stage, tokens } = fitState(messages, collectToolCalls(messages, 0), {
      ...fit,
      maxStateTokens: 300,
    });
    expect(stage).toBe('inputs<=200');
    expect(tokens).toBeLessThanOrEqual(300);
    expect(state.history[0]?.text).toBe('start');
    expect((state.history[1]?.tool_calls?.[0] as HistoryToolCall).input.length).toBeLessThanOrEqual(200);
  });

  it('shrinks old tool calls to one line each when nothing else is left to cut', () => {
    const messages = [message('user', 'start')];
    for (let i = 0; i < 40; i += 1) {
      messages.push(call(`c${i}`, 'Read', { file_path: `/repo/src/module-${i}.ts` }, 'x'), result(`c${i}`, 'x'));
    }
    messages.push(message('assistant', 'done'));
    const calls = collectToolCalls(messages, 1);
    const full = fitState(messages, calls, { ...fit, preserveRecentMessages: 1 });
    const compacted = fitState(messages, calls, {
      ...fit,
      preserveRecentMessages: 1,
      maxStateTokens: Math.floor(full.tokens * 0.8),
    });
    expect(compacted.stage).toBe('old calls compacted');
    expect(compacted.tokens).toBeLessThanOrEqual(Math.floor(full.tokens * 0.8));
    expect(compacted.tokens).toBeGreaterThanOrEqual(estimateTokens(JSON.stringify(compacted.state)));
    expect(compacted.state.history[1]?.tool_calls?.[0]).toBe(
      't1 Read file_path=/repo/src/module-0.ts → ok 1ch',
    );
    expect(compacted.state.history.at(-1)?.text).toBe('done');

    const merged = fitState(messages, calls, {
      ...fit,
      preserveRecentMessages: 1,
      maxStateTokens: Math.floor(full.tokens * 0.45),
    });
    expect(merged.stage).toBe('old calls merged');
    expect(merged.tokens).toBeLessThanOrEqual(Math.floor(full.tokens * 0.45));
    expect(merged.state.history).toHaveLength(3);
    expect(merged.state.history[1]?.tool_calls).toHaveLength(40);
    expect(merged.state.history[1]?.tool_calls?.[39]).toMatch(/^t40 Read /);
    expect(merged.state.history[0]?.text).toBe('start');
    expect(merged.state.history[2]?.text).toBe('done');
  });

  it('abridges long texts oldest-first and collapses old messages last', () => {
    const long = (n: number) => `${n} ` + 'lorem ipsum '.repeat(300);
    const messages = [
      message('user', long(0)),
      message('assistant', long(1)),
      message('user', long(2)),
      message('assistant', long(3)),
      message('user', 'latest'),
    ];
    const abridged = fitState(messages, [], { ...fit, maxStateTokens: 1800, preserveRecentMessages: 1 });
    expect(abridged.stage).toBe('texts abridged');
    expect(abridged.tokens).toBeLessThanOrEqual(1800);
    expect(abridged.state.history[1]?.text).toContain('chars omitted');
    expect(abridged.state.history[0]?.text).toBe(long(0));
    expect(abridged.state.history[4]?.text).toBe('latest');

    const collapsed = fitState(messages, [], { ...fit, maxStateTokens: 420, preserveRecentMessages: 1 });
    expect(collapsed.stage).toBe('old messages collapsed');
    expect(collapsed.tokens).toBeLessThanOrEqual(420);
    expect(collapsed.state.history[1]?.text).toMatch(/^\[… \d+ chars omitted …\]$/);
    expect(collapsed.state.history[0]?.text).toContain('lorem');
    expect(collapsed.state.history[4]?.text).toBe('latest');
  });

  it('leaves the oldest entries out behind a note when even one line per call is too much', () => {
    const messages = [message('user', 'start')];
    for (let i = 0; i < 400; i += 1) {
      messages.push(
        message('assistant', `step ${i}`),
        call(`c${i}`, 'Read', { file_path: `/repo/src/module-${i}.ts` }, 'x'),
        result(`c${i}`, 'x'),
      );
    }
    messages.push(message('user', 'latest'));
    const calls = collectToolCalls(messages, 2);
    const fitted = fitState(messages, calls, { ...fit, preserveRecentMessages: 2, maxStateTokens: 3000 });

    expect(fitted.stage).toBe('oldest left out');
    expect(fitted.tokens).toBeLessThanOrEqual(3000);
    expect(fitted.tokens).toBeGreaterThanOrEqual(estimateTokens(JSON.stringify(fitted.state)));
    expect(fitted.state.history[0]?.text).toBe('start');
    expect(fitted.state.history[1]?.text).toMatch(/^\[… \d+ older tool calls and \d+ messages omitted …\]$/);
    expect(fitted.state.history.at(-1)?.text).toBe('latest');
    const outline = renderState(fitted.state);
    expect(outline).toContain('t400 Read');
    expect(outline).not.toContain('t1 Read');
    // Trimmed line by line, so the budget is used rather than emptied.
    expect(fitted.tokens).toBeGreaterThan(2500);
    expect(outline.match(/^ {2}t\d+ Read/gm)!.length).toBeGreaterThan(50);
  });

  it('throws when the history cannot be fitted', () => {
    const messages = [message('user', 'a'.repeat(2000)), message('assistant', 'b')];
    expect(() => fitState(messages, [], { ...fit, maxStateTokens: 50 })).toThrow(/too large/);
  });
});

describe('decision items', () => {
  const calls: ToolCall[] = Array.from({ length: 10 }, (_, i) => ({
    id: `t${i + 1}`,
    tool_use_id: `tool-${i + 1}`,
    tool: 'Read',
    input: {},
    callIndex: i * 2 + 1,
    resultIndex: i * 2 + 2,
    resultChars: 100,
    isError: false,
    pinned: false,
  }));

  it('puts everything in one request when it fits', () => {
    expect(batchCalls(calls, { maxRequestItems: 128 })).toHaveLength(1);
  });

  it('splits calls into requests of at most maxRequestItems, in order', () => {
    const batches = batchCalls(calls, { maxRequestItems: 4 });
    expect(batches.map((b) => b.length)).toEqual([4, 4, 2]);
    expect(batches.flat().map((c) => c.id)).toEqual(calls.map((c) => c.id));
  });

  it('puts the focus block first, with the call input and the head of its output', () => {
    const messages = transcript();
    const found = collectToolCalls(messages, 0);
    const outline = renderState(fitState(messages, found, fit).state);
    const [first, , third] = decisionItems(messages, found, outline, { focusResultChars: 40 });

    expect(first).toMatch(/^\[focus\] tool call t1 Read input=\{"file_path":"src\/a.ts"\}/);
    expect(first).toContain(`${fileA.slice(0, 40)}\n[… ${fileA.length - 40} more chars]`);
    expect(first!.endsWith(outline)).toBe(true);
    expect(third).toContain('[focus output] error, 34 chars:\nFAIL b.test.ts: expected 2 to be 3');
  });

  it('renders the outline as transcript text with one line per call', () => {
    const messages = transcript();
    const outline = renderState(fitState(messages, collectToolCalls(messages, 0), fit).state);
    expect(outline).toContain('[goal] fix the test');
    expect(outline).toContain('[user] Never edit anything under src/generated. Fix the failing test.');
    expect(outline).toContain('  t3 Bash {"command":"npm test"} → error, 34 chars (omitted)');
    expect(outline).not.toContain('export const a');
  });
});

describe('decisions', () => {
  const options = { keepThreshold: 0.5 };
  const unpinned = { id: 't1', tool: 'Read', pinned: false };

  it('keeps, drops the result, or drops the call based on the keep probabilities', () => {
    expect(decideCall(unpinned, { keepCall: 0.9, keepResult: 0.7 }, options).action).toBe('keep');
    expect(decideCall(unpinned, { keepCall: 0.9, keepResult: 0.2 }, options).action).toBe('drop_result');
    expect(decideCall(unpinned, { keepCall: 0.1, keepResult: 0.2 }, options).action).toBe('drop_call');
    expect(decideCall({ ...unpinned, pinned: true }, { keepCall: 0, keepResult: 0 }, options)).toMatchObject({
      action: 'keep',
      reason: 'pinned',
    });
  });

  it('removes dropped calls and truncates dropped results', () => {
    const messages = transcript();
    messages[4]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[5]!.toolResults![0]!.text = 'x'.repeat(2000);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.1, keepResult: 0.1 }, options),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.1 }, options),
      decideCall(calls[2]!, { keepCall: 0.9, keepResult: 0.9 }, options),
    ];
    const kept = applyDecisions(messages, decisions, calls, 300);

    expect(kept.map((m) => m.text || m.toolUses[0]?.tool_use_id || m.toolResults?.[0]?.tool_use_id)).toEqual([
      'Never edit anything under src/generated. Fix the failing test.',
      'a.ts looks fine; checking b.ts',
      'tool-2',
      'tool-2',
      'tool-3',
      'tool-3',
      'The failure is in b.test.ts; fixing now.',
      'go ahead',
    ]);
    expect(kept[0]).toBe(messages[0]);
    expect(kept[2]).not.toBe(messages[4]);
    expect(kept[2]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-gliner-compaction truncated 1700 chars`),
    );
    expect(kept[3]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-gliner-compaction truncated 1700 chars`),
    );
    expect(kept[2]).not.toBe(messages[4]);
    expect(kept[3]).not.toBe(messages[5]);
    expect(kept[4]).toBe(messages[6]);
    expect(kept[5]?.toolResults?.[0]?.text).toContain('expected 2 to be 3');

    const shortMessages = transcript();
    shortMessages[4]!.toolUses[0]!.text = 'y'.repeat(100);
    shortMessages[5]!.toolResults![0]!.text = 'y'.repeat(100);
    const shortKept = applyDecisions(shortMessages, decisions, calls, 300);
    expect(shortKept[2]).toBe(shortMessages[4]);
    expect(shortKept[3]).toBe(shortMessages[5]);
  });

  it('honours truncateHeadChars, including a zero head', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 })];
    const original = messages[2]!.toolResults![0]!.text;
    const total = original.length;

    const kept = applyDecisions(messages, decisions, calls, 50);
    expect(kept[2]?.toolResults?.[0]?.text).toBe(
      `${original.slice(0, 50)}\n[fast-gliner-compaction truncated ${total - 50} chars of this tool result; re-run the tool if needed]`,
    );
    expect(kept[1]?.toolUses[0]?.text).toBe(kept[2]?.toolResults?.[0]?.text);

    const noHead = applyDecisions(messages, decisions, calls, 0);
    expect(noHead[2]?.toolResults?.[0]?.text).toBe(
      `[fast-gliner-compaction truncated ${total} chars of this tool result; re-run the tool if needed]`,
    );
  });
});

describe('compact', () => {
  it('sends one item per candidate with the shared questions and merges the answers in order', async () => {
    const seen: Seen[] = [];
    const messages = transcript();
    const output = await compact(
      messages,
      fakeDecider((question, item) =>
        question === 'keep_call' ? 0.9 : item.startsWith('[focus] tool call t2 ') ? 0.8 : 0.1,
        seen,
      ),
      { preserveRecentMessages: 1, maxRequestItems: 2 },
    );

    expect(output.stats.requests).toBe(seen.length);
    expect(seen.map((r) => r.items.length)).toEqual([2, 1]);
    for (const request of seen) expect(request.questions).toEqual(Object.keys(QUESTIONS));
    expect(seen.flatMap((r) => r.items.map((item) => item.slice(0, 22)))).toEqual([
      '[focus] tool call t1 R',
      '[focus] tool call t2 R',
      '[focus] tool call t3 B',
    ]);
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_result', 'keep', 'drop_result']);
    expect(output.messages).toHaveLength(messages.length);
    expect(output.stats).toMatchObject({ resultsDropped: 2, kept: 1, callsDropped: 0, pinned: 0, modelMs: 14 });
    expect(reductionRatio(output)).toBeGreaterThan(0);
  });

  it('keeps everything without calling the decider when no tool call is a candidate', async () => {
    const seen: Seen[] = [];
    const messages = [message('user', 'hello'), message('assistant', 'hi')];
    const output = await compact(messages, fakeDecider(() => 0, seen));
    expect(seen).toHaveLength(0);
    expect(output.stats).toMatchObject({ requests: 0, stateStage: '', calls: 0 });
    expect(output.messages).toEqual(messages);
  });

  it('reports a tiny reduction when the model wants everything kept', async () => {
    const output = await compact(transcript(), fakeDecider(() => 0.95), { preserveRecentMessages: 1 });
    expect(output.decisions.every((d) => d.action === 'keep')).toBe(true);
    expect(reductionRatio(output)).toBe(0);
  });

  it('rejects an answer count that does not match the items', async () => {
    const broken: Decider = { decide: async () => ({ answers: [{ keep_call: 0.5, keep_result: 0.5 }] }) };
    await expect(compact(transcript(), broken, { preserveRecentMessages: 1 })).rejects.toThrow(
      /answered 1 of 3/,
    );
  });
});

describe('HTTP client', () => {
  it('builds a decide request against the server URL', () => {
    const request = buildDecideRequest({ serverUrl: 'http://gpu:8765/', token: 'k' }, ['a'], { q: 'x?' });
    expect(request.url).toBe('http://gpu:8765/v1/decide');
    expect(request.headers.authorization).toBe('Bearer k');
    expect(JSON.parse(request.body)).toEqual({ model: 'gliner-decide', questions: { q: 'x?' }, items: ['a'] });

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

  it('asks over fetch and surfaces server errors', async () => {
    const bodies: string[] = [];
    const client = new GlinerClient({
      serverUrl: 'http://gpu:8765',
      model: 'gliner-decide-1b',
      fetch: (async (_url: string | URL | Request, init?: RequestInit) => {
        bodies.push(String(init?.body));
        return new Response(JSON.stringify({ answers: [{ q: 0.4 }] }), { status: 200 });
      }) as typeof fetch,
    });
    const response = await client.decide(['state'], { q: 'x?' });
    expect(response.answers).toEqual([{ q: 0.4 }]);
    expect(JSON.parse(bodies[0]!).model).toBe('gliner-decide-1b');

    const down = (async () => new Response('model not loaded', { status: 409 })) as typeof fetch;
    await expect(
      compactMessages(transcript(), { fetch: down, preserveRecentMessages: 1 }),
    ).rejects.toThrow(/409/);
  });
});
