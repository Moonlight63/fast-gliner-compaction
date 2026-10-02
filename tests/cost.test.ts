import { describe, expect, it } from 'vitest';
import { commandCost, costRules, ruleCost, splitCommands, toolMatcher } from '../src/index.js';

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
    expect(splitCommands('n=$(grep -c foo f); echo $n')).toEqual([['n='], ['grep', '-c', 'foo', 'f)'], ['echo', '$n']]);
  });
});

describe('command cost', () => {
  it.each([
    ['grep -rn "useAuth" app --include=*.ts | head -20', 'cheap'],
    ['cd /repo && git -C auth log --oneline -4; git status -sb', 'cheap'],
    ['for f in a b; do sed -n 1,5p $f; done', 'cheap'],
    ['git add -A && git commit -m "x"', 'cheap'],
    ['docker logs --since 30m api 2>&1 | grep -iE "error|warn"', 'cheap'],
    ["sed -i \\\n  -e 's/a/b/' f.vue", 'cheap'],
    ['cd app && npm test', 'expensive'],
    ['npx vitest run', 'expensive'],
    ['bun run build', 'expensive'],
    ["ssh gpu 'nvidia-smi'", 'expensive'],
    ['git push origin main', 'expensive'],
    ['docker compose up -d', 'expensive'],
    ['ls && npm test', 'expensive'],
    ['node -e "console.log(1)"', 'unknown'],
    ['python3 scripts/report.py', 'unknown'],
    ['P=db; q(){ docker exec $P psql -c "$1"; }; q "select 1"', 'unknown'],
    ['docker exec api env', 'unknown'],
  ])('%s -> %s', (command, expected) => {
    expect(commandCost(command)).toBe(expected);
  });
});

describe('tool rules', () => {
  const rules = costRules();
  const cost = (tool: string, input: Record<string, unknown> = {}) => ruleCost({ tool, input }, rules);

  it('classifies named tools, shell-running tools and network code', () => {
    expect(cost('Read')).toBe('cheap');
    expect(cost('Edit')).toBe('cheap');
    expect(cost('WebFetch')).toBe('expensive');
    expect(cost('Agent')).toBe('expensive');
    expect(cost('AskUserQuestion')).toBe('expensive');
    expect(cost('mcp__plugin_x_y__ctx_fetch_and_index')).toBe('expensive');
    expect(cost('Bash', { command: 'ls -la' })).toBe('cheap');
    expect(cost('mcp__x__ctx_batch_execute', { commands: [{ command: 'ls' }, { command: 'npm test' }] })).toBe('expensive');
    expect(cost('mcp__x__ctx_execute', { language: 'shell', code: 'grep -n x f' })).toBe('cheap');
    expect(cost('mcp__x__ctx_execute', { language: 'javascript', code: 'const r = await fetch(u)' })).toBe('expensive');
    expect(cost('mcp__x__ctx_execute', { language: 'javascript', code: 'console.log(1)' })).toBe('unknown');
    expect(cost('mcp__someone__do_thing')).toBe('unknown');
  });

  it('lets user globs win over the defaults', () => {
    const custom = costRules(['WebFetch'], ['Read', 'mcp__db__*']);
    expect(ruleCost({ tool: 'WebFetch', input: {} }, custom)).toBe('cheap');
    expect(ruleCost({ tool: 'Read', input: {} }, custom)).toBe('expensive');
    expect(ruleCost({ tool: 'mcp__db__query', input: {} }, custom)).toBe('expensive');
  });

  it('matches globs as whole names with * wildcards', () => {
    const match = toolMatcher(['mcp__*fetch*', 'Read']);
    expect(match('mcp__a__web_fetch')).toBe(true);
    expect(match('Read')).toBe(true);
    expect(match('ReadMore')).toBe(false);
    expect(match('mcp.fetch')).toBe(false);
  });
});
