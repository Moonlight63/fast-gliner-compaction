import { describe, expect, it } from 'vitest';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  resolveHookConfig,
  summarize,
  toSessionMessages,
} from '../hooks/fast-gliner.ts';
import { applyDecisions, collectToolCalls, type CallDecision, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);
const scriptOut = 'row\n'.repeat(300);

/** t1 Read (cheap by rule), t2 node script (left to the model). */
function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'node scripts/report.mjs' }, scriptOut),
    result('tool-2', scriptOut),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

/** A fake decision server answering P(expensive) for every item. */
function serverFetch(p: number, calls: { url: string; body: string }[] = []) {
  return async (url: string, init?: { body?: string }) => {
    calls.push({ url, body: init?.body ?? '' });
    const { items } = JSON.parse(init?.body ?? '{}') as { items: string[] };
    return { status: 200, ok: true, text: JSON.stringify({ answers: items.map(() => ({ expensive: p })), ms: 5 }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      model: 'gliner-decide-1b',
      serverUrl: 'http://127.0.0.1:8765',
    });
    expect(
      resolveHookConfig({
        serverUrl: 'http://172.16.16.125:8765',
        serverToken: 't',
        spareThreshold: 0.9,
        truncateInputChars: 200,
        cheapTools: ' mcp__db__query, Foo ',
        expensiveTools: 'Bar',
        model: 'gliner25-multi',
        compactAtPercent: 'no',
      }),
    ).toEqual({
      serverUrl: 'http://172.16.16.125:8765',
      serverToken: 't',
      spareThreshold: 0.9,
      truncateInputChars: 200,
      cheapTools: ['mcp__db__query', 'Foo'],
      expensiveTools: ['Bar'],
      model: 'gliner25-multi',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
    });
    expect(resolveHookConfig({ serverUrl: '', cheapTools: ' , ' })).toMatchObject({
      serverUrl: 'http://127.0.0.1:8765',
    });
    expect(resolveHookConfig({ cheapTools: ' , ' }).cheapTools).toBeUndefined();
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions: CallDecision[] = [
      { id: 't1', tool: 'Read', action: 'drop_result', reason: 'cheap', cost: 'cheap' },
      { id: 't2', tool: 'Bash', action: 'keep', reason: 'model_spared', cost: 'unknown', pExpensive: 0.9 },
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(/\n\[fast-gliner-compaction truncated \d+ chars of this tool result/);
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const calls: { url: string; body: string }[] = [];
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 2, serverUrl: 'http://gpu:8765' }),
      serverToken: 't',
    };
    const { result: output, messages } = await compactSession(transcript(), config, serverFetch(0.95, calls));
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe('http://gpu:8765/v1/decide');
    expect(JSON.parse(calls[0]!.body)).toMatchObject({ model: 'gliner-decide-1b' });
    expect(output.decisions.map((d) => d.reason)).toEqual(['cheap', 'model_spared']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', undefined, undefined, 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(
      /^\d+% reduction; 1 results truncated, 1 spared as expensive; 1 by rule, 1 by model in 5ms$/,
    );
    expect(decisionLog(output)).toBe('t1:Read:drop_result/cheap t2:Bash:keep/model_spared=0.95');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 2 });
    const { result: output } = await compactSession(transcript(), config, serverFetch(0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_result/cheap',
      'decisions (2/2): t2:Bash:drop_result/model_dropped=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('still compacts by rule when the server fails, keeping what it could not classify', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 2 });
    const { result: output } = await compactSession(transcript(), config, async () => ({
      status: 401,
      ok: false,
      text: 'invalid token',
    }));
    expect(output.decisions.map((d) => d.reason)).toEqual(['cheap', 'model_unavailable']);
    expect(output.stats.modelError).toMatch(/401/);
    expect(summarize(output)).toMatch(/model unavailable, 1 unclassified kept \(decision server request failed \(401\)/);
  });
});
