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

/** t1 Read (workspace: removed), t2 node script (left to the model). */
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

/** A fake decision server answering P(useful) for every item. */
function serverFetch(p: number, calls: { url: string; body: string }[] = []) {
  return async (url: string, init?: { body?: string }) => {
    calls.push({ url, body: init?.body ?? '' });
    const { items } = JSON.parse(init?.body ?? '{}') as { items: string[] };
    return { status: 200, ok: true, text: JSON.stringify({ answers: items.map(() => ({ useful: p })), ms: 5 }) };
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
        keepThreshold: 0.9,
        trimTailChars: 500,
        removeTools: ' mcp__db__query, Foo ',
        keepTools: 'Bar',
        breadcrumbs: false,
        model: 'gliner25-multi',
        compactAtPercent: 'no',
      }),
    ).toEqual({
      serverUrl: 'http://172.16.16.125:8765',
      serverToken: 't',
      keepThreshold: 0.9,
      trimTailChars: 500,
      removeTools: ['mcp__db__query', 'Foo'],
      keepTools: ['Bar'],
      breadcrumbs: false,
      model: 'gliner25-multi',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
    });
    expect(resolveHookConfig({ serverUrl: '' }).serverUrl).toBe('http://127.0.0.1:8765');
    expect(resolveHookConfig({ removeTools: ' , ' }).removeTools).toBeUndefined();
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions: CallDecision[] = [
      { id: 't1', tool: 'Read', action: 'remove', reason: 'workspace', category: 'workspace' },
      { id: 't2', tool: 'Bash', action: 'trim', reason: 'model_trim', category: 'unknown', pUseful: 0.1 },
    ];
    const out = toSessionMessages(
      messages,
      applyDecisions(messages, decisions, calls, { trimHeadChars: 300, trimTailChars: 300, trimInputChars: 300, breadcrumbs: true }),
    );
    expect(out.map((m) => m.handle)).toEqual(['h-0', undefined, undefined, undefined, 'h-5', 'h-6']);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.text).toBe('[fast-gliner-compaction removed 1 tool call: Read src/a.ts]');
    expect(out[2]?.toolUses[0]?.text).toMatch(/\[fast-gliner-compaction trimmed \d+ chars of this tool result/);
    expect(out[3]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-2', isError: false });
    expect(out[4]).toBe(messages[5]);
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
    expect(JSON.parse(calls[0]!.body)).toMatchObject({ model: 'gliner-decide-1b', questions: { useful: expect.any(String) } });
    expect(output.decisions.map((d) => d.reason)).toEqual(['workspace', 'model_keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', undefined, 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 removed, 1 kept; 1 by rule, 1 by model in 5ms$/);
    expect(decisionLog(output)).toBe('t1:Read:remove/workspace t2:Bash:keep/model_keep=0.95');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 2 });
    const { result: output } = await compactSession(transcript(), config, serverFetch(0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:remove/workspace',
      'decisions (2/2): t2:Bash:trim/model_trim=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('still compacts by rule when the server fails, trimming what it could not place', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 2 });
    const { result: output } = await compactSession(transcript(), config, async () => ({
      status: 401,
      ok: false,
      text: 'invalid token',
    }));
    expect(output.decisions.map((d) => d.reason)).toEqual(['workspace', 'model_unavailable']);
    expect(output.stats.modelError).toMatch(/401/);
    expect(summarize(output)).toMatch(/model unavailable, 1 unplaced trimmed \(decision server request failed \(401\)/);
  });
});
