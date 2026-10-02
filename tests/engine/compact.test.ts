// Engine test: `claude plugin test .` loads the plugin the way a session does
// and raises session.compact through the engine. Beneath the plugin, the test
// stands in for the decision server (http.fetch), the UI (ui.log, ui.toast)
// and core's own compaction (session.compact, reached only on a fallback).
import { expect, mock, test } from 'claude-code/testing';
import type { HttpResponse, SessionMessage } from 'claude-code';

const file = 'export const a = 1;\n'.repeat(50);
const rows = 'row\n'.repeat(300);

function transcript(): SessionMessage[] {
  return [
    { role: 'user', text: 'Fix the failing test.', toolUses: [] },
    {
      role: 'assistant',
      text: '',
      toolUses: [{ tool_use_id: 'r1', tool: 'Read', input: { file_path: 'src/a.ts' }, text: file }],
    },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'r1', text: file, isError: false }] },
    {
      role: 'assistant',
      text: '',
      toolUses: [{ tool_use_id: 'n1', tool: 'Bash', input: { command: 'node scripts/report.mjs' }, text: rows }],
    },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'n1', text: rows, isError: false }] },
    { role: 'assistant', text: 'Fixing now.', toolUses: [] },
    { role: 'user', text: 'go ahead', toolUses: [] },
  ];
}

const answer = (useful: number): HttpResponse => ({
  status: 200,
  ok: true,
  headers: {},
  text: JSON.stringify({ answers: [{ useful }], ms: 3 }),
});

const options = { serverUrl: 'http://gpu.test:8765', preserveRecentMessages: 2 };
const builtIn: SessionMessage[] = [{ role: 'assistant', text: 'built-in summary', toolUses: [] }];

test('replaces the compaction with the removal-first transcript', { options }, async ($, on) => {
  const urls: string[] = [];
  const logs: string[] = [];
  on('http.fetch', async (_$, e) => {
    urls.push(e.url);
    return { value: answer(0.1) };
  });
  on('ui.log', async (_$, e) => {
    logs.push(e.text);
    return {};
  });
  on('ui.toast', async (_$, e) => {
    logs.push(`toast: ${e.text}`);
    return {};
  });
  on('session.compact', async () => ({ messages: builtIn }));
  on('settings.read', async () => ({ value: {} }));
  mock.env(on, {});

  const result = await $.session.compact({ trigger: 'manual', messages: transcript() });

  expect(urls).toEqual(['http://gpu.test:8765/v1/decide']);
  if (!result.messages) throw new Error(`compaction skipped: ${result.skip}`);
  const texts = result.messages.map((m) => m.text);
  expect(texts).not.toContain('built-in summary');
  expect(texts).toContain('[fast-gliner-compaction removed 1 tool call: Read src/a.ts]');
  const trimmed = result.messages.find((m) => m.toolResults?.[0]?.tool_use_id === 'n1');
  expect(trimmed?.toolResults?.[0]?.text).toMatch(/\[fast-gliner-compaction trimmed \d+ chars/);
  expect(logs.some((l) => /^toast: .*\d+% reduction; 1 removed, 1 trimmed; 1 by rule, 1 by model/.test(l))).toBe(true);
});

test('still compacts by rule when the decision server fails', { options }, async ($, on) => {
  const logs: string[] = [];
  on('http.fetch', async () => ({ value: { status: 401, ok: false, headers: {}, text: 'invalid token' } }));
  on('ui.log', async (_$, e) => {
    logs.push(e.text);
    return {};
  });
  on('ui.toast', async (_$, e) => {
    logs.push(`toast: ${e.text}`);
    return {};
  });
  on('session.compact', async () => ({ messages: builtIn }));
  on('settings.read', async () => ({ value: {} }));
  mock.env(on, {});

  const result = await $.session.compact({ trigger: 'manual', messages: transcript() });

  if (!result.messages) throw new Error(`compaction skipped: ${result.skip}`);
  expect(result.messages.map((m) => m.text)).toContain('[fast-gliner-compaction removed 1 tool call: Read src/a.ts]');
  expect(logs.some((l) => /model unavailable, 1 unplaced trimmed \(decision server request failed \(401\)/.test(l))).toBe(true);
});
