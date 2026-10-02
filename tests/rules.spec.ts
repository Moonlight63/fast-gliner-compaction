import { describe, expect, it } from 'vitest';
import { callTarget, categorize, categoryRules, commandCategory, splitCommands, toolMatcher } from '../src/index.js';

describe('shell command splitting', () => {
  it('splits on unquoted separators only and drops heredoc bodies', () => {
    expect(splitCommands(`grep -n "a\\|b" f.ts | head -3 && echo 'x; y'`)).toEqual([
      ['grep', '-n', 'a|b', 'f.ts'],
      ['head', '-3'],
      ['echo', 'x; y'],
    ]);
    expect(splitCommands(`cat > v.ts <<'EOF'\nnpm test\nEOF\nwc -l v.ts`)).toEqual([
      ['cat', '>', 'v.ts'],
      ['wc', '-l', 'v.ts'],
    ]);
    expect(splitCommands('docker logs x 2>&1 | tail')).toEqual([['docker', 'logs', 'x', '2>&1'], ['tail']]);
    expect(splitCommands('n=$(grep -c foo f); echo $n').map((w) => w[0])).toEqual(['n=', 'grep', 'echo']);
  });
});

describe('command category', () => {
  it.each([
    ['grep -rn "useAuth" app --include=*.ts | head -20', 'workspace'],
    ['cd /repo && git -C auth log --oneline -4; git status -sb', 'workspace'],
    ['for f in a b; do sed -n 1,5p $f; done', 'workspace'],
    ['git add -A && git commit -m "x"', 'workspace'],
    ['docker logs --since 30m api 2>&1 | grep -iE "error|warn"', 'workspace'],
    ["sed -i \\\n  -e 's/a/b/' f.vue", 'workspace'],
    ['cd app && npm test', 'outcome'],
    ['npx vitest run', 'outcome'],
    ['bun run build', 'outcome'],
    ["ssh gpu 'nvidia-smi'", 'outcome'],
    ['git push origin main', 'outcome'],
    ['docker compose up -d', 'outcome'],
    ['ls && npm test', 'outcome'],
    ['curl -s https://api.example.com/v1/status', 'research'],
    ['gh pr view 12', 'research'],
    ['npm test && curl -s https://x', 'research'],
    ['node -e "console.log(1)"', 'unknown'],
    ['python3 scripts/report.py', 'unknown'],
    ['P=db; q(){ docker exec $P psql -c "$1"; }; q "select 1"', 'unknown'],
  ])('%s -> %s', (command, expected) => {
    expect(commandCategory(command)).toBe(expected);
  });
});

describe('tool categories', () => {
  const rules = categoryRules();
  const category = (tool: string, input: Record<string, unknown> = {}) => categorize({ tool, input }, rules);

  it('places named tools, shell-running tools and network code', () => {
    expect(category('Read')).toBe('workspace');
    expect(category('Write')).toBe('workspace');
    expect(category('Edit')).toBe('workspace');
    expect(category('WebFetch')).toBe('research');
    expect(category('Agent')).toBe('research');
    expect(category('AskUserQuestion')).toBe('research');
    expect(category('mcp__plugin_x_y__ctx_fetch_and_index')).toBe('research');
    expect(category('Bash', { command: 'ls -la' })).toBe('workspace');
    expect(category('mcp__x__ctx_batch_execute', { commands: [{ command: 'ls' }, { command: 'npm test' }] })).toBe('outcome');
    expect(category('mcp__x__ctx_execute', { language: 'shell', code: 'grep -n x f' })).toBe('workspace');
    expect(category('mcp__x__ctx_execute', { language: 'javascript', code: 'const r = await fetch(u)' })).toBe('research');
    expect(category('mcp__x__ctx_execute', { language: 'javascript', code: 'console.log(1)' })).toBe('unknown');
    expect(category('mcp__someone__do_thing')).toBe('unknown');
  });

  it('matches globs as whole names with * wildcards', () => {
    const match = toolMatcher(['mcp__*fetch*', 'Read']);
    expect(match('mcp__a__web_fetch')).toBe(true);
    expect(match('Read')).toBe(true);
    expect(match('ReadMore')).toBe(false);
    expect(match('mcp.fetch')).toBe(false);
  });
});

describe('call targets', () => {
  it('identifies a call by file, normalised command, URL or query', () => {
    expect(callTarget({ input: { file_path: 'src/a.ts', offset: 10 } })).toBe('file:src/a.ts');
    expect(callTarget({ input: { command: 'npm   test\n' } })).toBe('cmd:npm test');
    expect(callTarget({ input: { url: 'https://x' } })).toBe('url:https://x');
    expect(callTarget({ input: { query: 'oauth' } })).toBe('query:oauth');
    expect(callTarget({ input: { pattern: '*.ts' } })).toBeUndefined();
  });
});
