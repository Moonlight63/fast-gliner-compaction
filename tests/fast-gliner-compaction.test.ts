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
  supersededCalls,
  traceText,
  trimmedText,
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
const testLog = `${'PASS a.test.ts\n'.repeat(80)}Tests: 80 passed\n`;
const scriptOut = 'row\n'.repeat(300);
const docs = 'docs '.repeat(200);

/**
 * t1 Read (workspace), t2 Edit (workspace), t3 npm test (outcome, superseded
 * by t6), t4 node script (unknown), t5 WebFetch (research), t6 npm test (outcome).
 */
function transcript(): Message[] {
  return [
    message('user', 'Never edit anything under src/generated. Fix the failing test.'),
    message('assistant', 'Reading the module first.'),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Edit', { file_path: 'src/a.ts', old_string: 'a = 1', new_string: 'a = 2' }, 'ok'),
    result('tool-2', 'ok'),
    call('tool-3', 'Bash', { command: 'npm test' }, testLog),
    result('tool-3', testLog),
    call('tool-4', 'Bash', { command: 'node scripts/report.mjs' }, scriptOut),
    result('tool-4', scriptOut),
    call('tool-5', 'WebFetch', { url: 'https://example.com/docs' }, docs),
    result('tool-5', docs),
    call('tool-6', 'Bash', { command: 'npm test' }, testLog),
    result('tool-6', testLog),
    message('assistant', 'Tests pass.'),
    message('user', 'go ahead'),
  ];
}

type Seen = { items: readonly string[]; questions: string[] };

function fakeDecider(answer: (item: string) => number, seen: Seen[] = []): Decider {
  return {
    async decide(items, questions) {
      seen.push({ items, questions: Object.keys(questions) });
      return { answers: items.map((item) => ({ useful: answer(item) })), ms: 7 };
    },
  };
}

const decision = (id: string, action: CallDecision['action']): CallDecision => ({
  id,
  tool: 'Read',
  action,
  reason: action === 'keep' ? 'research' : action === 'trim' ? 'outcome' : 'workspace',
  category: action === 'keep' ? 'research' : action === 'trim' ? 'outcome' : 'workspace',
});

const applyOptions = { trimHeadChars: 50, trimTailChars: 20, trimInputChars: 300, breadcrumbs: true };

describe('options', () => {
  it('fills in removal-first defaults and ignores non-finite values', () => {
    expect(resolveOptions()).toEqual({
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
    });
    expect(
      resolveOptions({ keepThreshold: Number.NaN, preserveRecentMessages: 2.7, trimHeadChars: -1.2, breadcrumbs: false }),
    ).toMatchObject({ keepThreshold: 0.85, preserveRecentMessages: 2, trimHeadChars: 0, breadcrumbs: false });
  });
});

describe('tool call collection', () => {
  it('pairs each tool call with its result and pins the first and recent messages', () => {
    const calls = collectToolCalls(transcript(), 4);
    expect(calls.map((c) => [c.id, c.tool, c.pinned])).toEqual([
      ['t1', 'Read', false],
      ['t2', 'Edit', false],
      ['t3', 'Bash', false],
      ['t4', 'Bash', false],
      ['t5', 'WebFetch', false],
      ['t6', 'Bash', true],
    ]);
    expect(resultText(transcript(), calls[0]!)).toBe(fileA);
  });

  it('ignores calls without a result', () => {
    const messages = [message('user', 'hi'), call('tool-9', 'Read', { file_path: 'x' }, '')];
    expect(collectToolCalls(messages, 0)).toEqual([]);
  });
});

describe('superseded calls', () => {
  it('marks every call a later call repeats on the same target', () => {
    expect([...supersededCalls(collectToolCalls(transcript(), 0))].sort()).toEqual(['t1', 't3']);
  });
});

describe('focus text', () => {
  it('shows the tool, its input and the head of its output', () => {
    const calls = collectToolCalls(transcript(), 0);
    const text = focusText(calls[3]!, scriptOut, 40);
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

describe('trimming and traces', () => {
  it('keeps the head and the tail of a long result around a note', () => {
    expect(trimmedText('short', false, 10, 10)).toBe('short');
    const trimmed = trimmedText(testLog, false, 15, 17);
    expect(trimmed).toBe(
      `PASS a.test.ts\n\n[fast-gliner-compaction trimmed ${testLog.length - 32} chars of this tool result; re-run the tool if needed]\nTests: 80 passed\n`,
    );
    expect(trimmedText(testLog, true, 0, 0)).toBe(
      `[fast-gliner-compaction trimmed ${testLog.length} chars of this tool result (error); re-run the tool if needed]`,
    );
  });

  it('shortens long string inputs but keeps short fields like paths', () => {
    const content = 'line\n'.repeat(400);
    expect(truncatedInput({ file_path: 'a' }, 300)).toEqual({ file_path: 'a' });
    expect(truncatedInput({ file_path: 'src/big.ts', content }, 100)).toEqual({
      file_path: 'src/big.ts',
      content: `${content.slice(0, 100)}\n[fast-gliner-compaction truncated ${content.length - 100} chars]`,
    });
  });

  it('names removed calls by target and folds repeats', () => {
    const tools = collectToolCalls(transcript(), 0).map((c) => ({
      tool_use_id: c.tool_use_id,
      tool: c.tool,
      input: c.input,
    }));
    expect(traceText([tools[0]!, tools[1]!, tools[1]!, tools[2]!])).toBe(
      '[fast-gliner-compaction removed 4 tool calls: Read src/a.ts, Edit src/a.ts ×2, Bash npm test]',
    );
    expect(traceText([{ tool_use_id: 'g', tool: 'Glob', input: { pattern: '**/*.ts' } }])).toBe(
      '[fast-gliner-compaction removed 1 tool call: Glob **/*.ts]',
    );
  });
});

describe('applying decisions', () => {
  it('removes calls with their results behind one trace, trims, and reuses untouched messages', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const out = applyDecisions(
      messages,
      [decision('t1', 'remove'), decision('t2', 'remove'), decision('t3', 'trim'), decision('t5', 'keep')],
      calls,
      applyOptions,
    );

    expect(out.map((m) => m.text || m.toolUses[0]?.tool_use_id || m.toolResults?.[0]?.tool_use_id)).toEqual([
      'Never edit anything under src/generated. Fix the failing test.',
      'Reading the module first.',
      '[fast-gliner-compaction removed 2 tool calls: Read src/a.ts, Edit src/a.ts]',
      'tool-3',
      'tool-3',
      'tool-4',
      'tool-4',
      'tool-5',
      'tool-5',
      'tool-6',
      'tool-6',
      'Tests pass.',
      'go ahead',
    ]);
    expect(out[0]).toBe(messages[0]);
    expect(out[2]).toMatchObject({ role: 'assistant', toolUses: [] });
    expect(out[3]?.toolUses[0]?.text).toMatch(/^PASS a.test.ts\n.*\n\[fast-gliner-compaction trimmed \d+ chars/s);
    expect(out[4]?.toolResults?.[0]?.text).toMatch(/\nTests: 80 passed\n$/);
    expect(out[5]).toBe(messages[8]);
    expect(out[7]).toBe(messages[10]);
  });

  it('never puts a trace between a tool_use and its tool_result', () => {
    const messages = [
      message('user', 'go'),
      message('assistant', '', {
        toolUses: [
          { tool_use_id: 'a', tool: 'Read', input: { file_path: 'x' }, text: fileA },
          { tool_use_id: 'b', tool: 'WebFetch', input: { url: 'u' }, text: docs },
        ],
      }),
      message('user', '', {
        toolResults: [
          { tool_use_id: 'a', text: fileA },
          { tool_use_id: 'b', text: docs },
        ],
      }),
      message('assistant', 'done'),
    ];
    const calls = collectToolCalls(messages, 0);
    const out = applyDecisions(messages, [decision('t1', 'remove'), decision('t2', 'keep')], calls, applyOptions);
    expect(out.map((m) => m.text || `${m.role}:${m.toolUses.map((t) => t.tool_use_id)}${(m.toolResults ?? []).map((r) => r.tool_use_id)}`)).toEqual([
      'go',
      'assistant:b',
      'user:b',
      '[fast-gliner-compaction removed 1 tool call: Read x]',
      'done',
    ]);
  });

  it('drops removed calls silently when breadcrumbs are off', () => {
    const messages = transcript();
    const out = applyDecisions(messages, [decision('t1', 'remove')], collectToolCalls(messages, 0), {
      ...applyOptions,
      breadcrumbs: false,
    });
    expect(out).toHaveLength(messages.length - 2);
    expect(out.some((m) => m.text.includes('fast-gliner-compaction removed'))).toBe(false);
  });
});

describe('compact', () => {
  it('removes workspace and superseded calls, trims outcomes, keeps research, and asks about the rest', async () => {
    const seen: Seen[] = [];
    const output = await compact(transcript(), fakeDecider(() => 0.2, seen), { preserveRecentMessages: 2 });

    expect(seen).toHaveLength(1);
    expect(seen[0]!.questions).toEqual(Object.keys(QUESTIONS));
    expect(seen[0]!.items).toHaveLength(1);
    expect(seen[0]!.items[0]).toMatch(/^\[tool call\] Bash input=\{"command":"node scripts\/report.mjs"\}/);
    expect(output.decisions.map((d) => [d.id, d.action, d.reason, d.category])).toEqual([
      ['t1', 'remove', 'superseded', 'workspace'],
      ['t2', 'remove', 'workspace', 'workspace'],
      ['t3', 'remove', 'superseded', 'outcome'],
      ['t4', 'trim', 'model_trim', 'unknown'],
      ['t5', 'keep', 'research', 'research'],
      ['t6', 'trim', 'outcome', 'outcome'],
    ]);
    expect(output.decisions[3]?.pUseful).toBe(0.2);
    expect(output.stats).toMatchObject({
      calls: 6,
      removed: 3,
      trimmed: 2,
      kept: 1,
      pinned: 0,
      byRule: 5,
      byModel: 1,
      requests: 1,
      modelMs: 7,
    });
    expect(output.stats.modelError).toBeUndefined();
    expect(output.messages.some((m) => m.text.startsWith('[fast-gliner-compaction removed 3 tool calls'))).toBe(true);
    expect(reductionRatio(output)).toBeGreaterThan(0.5);
  });

  it('keeps an unplaced result whole when the model reaches keepThreshold', async () => {
    const output = await compact(transcript(), fakeDecider(() => 0.9), { preserveRecentMessages: 2 });
    expect(output.decisions[3]).toMatchObject({ action: 'keep', reason: 'model_keep', pUseful: 0.9 });
    const strict = await compact(transcript(), fakeDecider(() => 0.9), { preserveRecentMessages: 2, keepThreshold: 0.95 });
    expect(strict.decisions[3]).toMatchObject({ action: 'trim', reason: 'model_trim' });
  });

  it('lets keepTools and removeTools override the rules', async () => {
    const output = await compact(transcript(), fakeDecider(() => 0), {
      preserveRecentMessages: 2,
      keepTools: ['Read'],
      removeTools: ['WebFetch', 'Read'],
    });
    expect(output.decisions.map((d) => d.reason)).toEqual([
      'user_keep',
      'workspace',
      'superseded',
      'model_trim',
      'user_remove',
      'outcome',
    ]);
  });

  it('works on rules alone and trims unplaced results when the model is unavailable', async () => {
    const rulesOnly = await compact(transcript(), null, { preserveRecentMessages: 2 });
    expect(rulesOnly.decisions[3]).toMatchObject({ action: 'trim', reason: 'model_unavailable' });
    expect(rulesOnly.stats).toMatchObject({ byRule: 5, byModel: 0, requests: 0, modelError: 'no decision model configured' });

    const failing: Decider = { decide: async () => Promise.reject(new Error('connect ECONNREFUSED')) };
    const down = await compact(transcript(), failing, { preserveRecentMessages: 2 });
    expect(down.decisions[3]).toMatchObject({ action: 'trim', reason: 'model_unavailable' });
    expect(down.stats.modelError).toBe('connect ECONNREFUSED');
  });

  it('does not call the model when the rules place everything', async () => {
    const seen: Seen[] = [];
    const messages = [message('user', 'hi'), call('a', 'Read', { file_path: 'x' }, fileA), result('a', fileA), message('assistant', 'ok')];
    const output = await compact(messages, fakeDecider(() => 0, seen), { preserveRecentMessages: 1 });
    expect(seen).toHaveLength(0);
    expect(output.stats).toMatchObject({ byRule: 1, byModel: 0, requests: 0, removed: 1 });
  });

  it('reports an answer count that does not match the items as a model error', async () => {
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
