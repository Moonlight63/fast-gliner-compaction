import type { RerunCost, ToolCall } from './types.js';

/**
 * Tools whose output is cheap to get again: local lookups, and edits whose
 * result is only a confirmation. Globs: `*` matches any run of characters.
 */
export const DEFAULT_CHEAP_TOOLS: readonly string[] = [
  'Read',
  'Glob',
  'Grep',
  'LS',
  'Edit',
  'MultiEdit',
  'Write',
  'NotebookRead',
  'NotebookEdit',
  'TodoRead',
  'TodoWrite',
  'ToolSearch',
  'Skill',
  'ListMcpResourcesTool',
  'ReadMcpResourceTool',
  'mcp__*__read_file',
  'mcp__*__get_repo_structure',
  'mcp__*__ctx_search',
  'mcp__*__ctx_stats',
  'mcp__*__resolve-library-id',
  'mcp__*__hindsight_search_knowledge_pages',
  'mcp__*__hindsight_read_knowledge_page',
  'mcp__*__hindsight_list_knowledge_pages',
  'mcp__*__hindsight_ingest_document',
  'mcp__*__hindsight_sync_status',
];

/** Tools whose output is slow, external, or impossible to reproduce. */
export const DEFAULT_EXPENSIVE_TOOLS: readonly string[] = [
  'WebFetch',
  'WebSearch',
  'Task',
  'Agent',
  'AskUserQuestion',
  'mcp__*fetch*',
  'mcp__*web_search*',
  'mcp__*webReader*',
  'mcp__*web_reader*',
  'mcp__*query-docs',
  'mcp__*__hindsight_reflect',
];

const CHEAP_PROGRAMS = new Set([
  'ls', 'cat', 'head', 'tail', 'grep', 'egrep', 'fgrep', 'rg', 'find', 'fd', 'tree', 'wc',
  'sort', 'uniq', 'cut', 'tr', 'diff', 'stat', 'file', 'du', 'df', 'pwd', 'echo', 'printf',
  'which', 'whereis', 'type', 'env', 'printenv', 'date', 'whoami', 'hostname', 'uname', 'id',
  'ps', 'pgrep', 'ss', 'jq', 'yq', 'sed', 'awk', 'basename', 'dirname', 'realpath', 'readlink',
  'test', 'true', 'false', 'free', 'nproc', 'nvidia-smi', 'mkdir', 'touch', 'cp', 'mv', 'rm',
  'rmdir', 'ln', 'chmod', 'kill', 'pkill',
]);
const EXPENSIVE_PROGRAMS = new Set([
  'pytest', 'vitest', 'jest', 'mocha', 'playwright', 'tsc', 'vue-tsc', 'eslint', 'cargo', 'make',
  'gradle', 'mvn', 'dotnet', 'curl', 'wget', 'ssh', 'scp', 'rsync', 'gh', 'sleep',
]);
const PACKAGE_MANAGERS = new Set(['npm', 'pnpm', 'yarn', 'bun']);
const PACKAGE_EXPENSIVE = new Set(['test', 'run', 'install', 'i', 'ci', 'build', 'exec', 'x', 'add', 'dlx']);
// Read-only git, plus local writes whose output is only a confirmation.
const CHEAP_GIT = new Set([
  'status', 'diff', 'log', 'show', 'branch', 'remote', 'rev-parse', 'ls-files', 'blame', 'tag',
  'describe', 'add', 'commit', 'reset', 'checkout', 'switch', 'restore', 'stash', 'mv', 'rm',
  'init', 'config', 'grep', 'shortlog', 'reflog',
]);
/** Keywords that prefix a command without deciding what runs (`if grep …`). */
const SKIP_WORDS = new Set([
  'while', 'until', 'do', 'done', 'if', 'then', 'else', 'elif', 'fi', 'esac', '{', '}', '!',
]);
/** Builtins whose whole command is bookkeeping, arguments included. */
const BUILTINS = new Set([
  'cd', 'pushd', 'popd', 'set', 'export', 'local', 'return', 'exit', 'source', '.', 'unset',
  'shopt', 'trap', 'wait', 'read', 'shift',
]);
const EXPENSIVE_GIT = new Set(['push', 'pull', 'fetch', 'clone']);
const CHEAP_DOCKER = new Set(['ps', 'logs', 'inspect', 'images', 'version', 'info']);
const EXPENSIVE_DOCKER = new Set(['build', 'compose', 'run', 'pull', 'push']);
const PIP = new Set(['pip', 'pip3', 'uv', 'poetry']);

/** Compiles tool-name globs into one matcher. */
export function toolMatcher(globs: readonly string[]): (tool: string) => boolean {
  const patterns = globs.map(
    (glob) => new RegExp(`^${glob.split('*').map((part) => part.replace(/[.+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`),
  );
  return (tool) => patterns.some((pattern) => pattern.test(tool));
}

/** Removes heredoc bodies, keeping the line that opens each one. */
function stripHeredocs(command: string): string {
  return command.replace(
    /<<-?\s*(['"]?)([A-Za-z_][A-Za-z0-9_]*)\1([^\n]*)\n[\s\S]*?\n[ \t]*\2[ \t]*(?=\n|$)/g,
    '$3',
  );
}

/**
 * Splits a command line into simple commands at unquoted `&&`, `||`, `;`,
 * `|`, `&`, newlines, and the start of `$(…)` / backtick substitutions, so
 * substituted commands are judged too. Quotes are kept out of the words.
 */
export function splitCommands(command: string): string[][] {
  const text = stripHeredocs(command).replace(/\\\n/g, ' ');
  const commands: string[][] = [];
  let words: string[] = [];
  let word = '';
  let quoted = false;
  let quote: "'" | '"' | null = null;
  const endWord = (): void => {
    if (word || quoted) words.push(word);
    word = '';
    quoted = false;
  };
  const endCommand = (): void => {
    endWord();
    if (words.length > 0) commands.push(words);
    words = [];
  };
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    if (quote === "'") {
      if (ch === "'") quote = null;
      else word += ch;
      continue;
    }
    if (ch === '\\' && i + 1 < text.length) {
      word += text[++i];
      continue;
    }
    if (ch === '$' && text[i + 1] === '(' && text[i + 2] !== '(') {
      i++;
      endCommand();
      continue;
    }
    if (ch === '`') {
      endCommand();
      continue;
    }
    if (quote === '"') {
      if (ch === '"') quote = null;
      else word += ch;
      continue;
    }
    if (ch === "'" || ch === '"') {
      quote = ch;
      quoted = true;
      continue;
    }
    // `2>&1`, `&>` and `>&` are redirections, not separators.
    if (ch === '&' && (word.endsWith('>') || text[i + 1] === '>')) {
      word += ch;
      continue;
    }
    if (ch === ';' || ch === '|' || ch === '&' || ch === '\n') {
      endCommand();
      continue;
    }
    if (ch === ' ' || ch === '\t') {
      endWord();
      continue;
    }
    if ((ch === '(' || ch === ')') && !word) {
      endCommand();
      continue;
    }
    word += ch;
  }
  endCommand();
  return commands;
}

/** The cost of one simple command (no pipes or separators). */
function segmentCost(input: readonly string[]): RerunCost | 'skip' {
  const words = [...input];
  // `for x in …` / `case … in` headers list words, not commands.
  if (words[0] === 'for' || words[0] === 'case' || words[0] === 'select') return 'skip';
  // Leading VAR=value assignments, keywords and wrappers do not change what runs.
  for (;;) {
    const first = words[0];
    if (first === undefined) return 'skip';
    if (/^[A-Za-z_][A-Za-z0-9_]*=/.test(first) || SKIP_WORDS.has(first) || first === 'sudo' || first === 'command') {
      words.shift();
    } else if (first === 'timeout' || first === 'nice') {
      words.splice(0, first === 'timeout' ? 2 : 1);
    } else break;
  }
  const raw = words[0]!;
  if (BUILTINS.has(raw)) return 'skip';
  // A function definition (`name(){`) or a call to one is judged by its body.
  if (/^[A-Za-z_][A-Za-z0-9_-]*\(\)\{?$/.test(raw) || raw.endsWith('(){')) return 'skip';
  const program = raw.split('/').pop() ?? '';
  let sub = words[1] ?? '';
  if (program === 'git' && sub === '-C') sub = words[3] ?? '';
  if (EXPENSIVE_PROGRAMS.has(program)) return 'expensive';
  if (program === 'npx' || program === 'bunx') return 'expensive';
  if (PACKAGE_MANAGERS.has(program)) return PACKAGE_EXPENSIVE.has(sub) ? 'expensive' : 'unknown';
  if (PIP.has(program)) return ['install', 'sync', 'add'].includes(sub) ? 'expensive' : 'unknown';
  if (program === 'git') {
    if (EXPENSIVE_GIT.has(sub)) return 'expensive';
    return CHEAP_GIT.has(sub) ? 'cheap' : 'unknown';
  }
  if (program === 'docker') {
    if (EXPENSIVE_DOCKER.has(sub)) return 'expensive';
    return CHEAP_DOCKER.has(sub) ? 'cheap' : 'unknown';
  }
  return CHEAP_PROGRAMS.has(program) ? 'cheap' : 'unknown';
}

/**
 * The re-run cost of a shell command: expensive when any simple command in
 * it is, cheap when every one is, otherwise unknown. Loop and function
 * bodies and substitutions are judged command by command; a call to a
 * function defined in the same line is judged by that function's body.
 */
export function commandCost(command: string): RerunCost {
  const segments = splitCommands(command);
  const functions = new Set(
    segments.flatMap((words) => words.filter((w) => /\(\)\{?$/.test(w)).map((w) => w.replace(/\(\)\{?$/, ''))),
  );
  let sawCheap = false;
  let sawUnknown = false;
  for (const words of segments) {
    const first = words.find((w) => !/^[A-Za-z_][A-Za-z0-9_]*=/.test(w) && !SKIP_WORDS.has(w));
    if (first && functions.has(first)) continue;
    const cost = segmentCost(words);
    if (cost === 'expensive') return 'expensive';
    if (cost === 'cheap') sawCheap = true;
    if (cost === 'unknown') sawUnknown = true;
  }
  return sawCheap && !sawUnknown ? 'cheap' : 'unknown';
}

const NETWORK_CODE = /\bfetch\s*\(|\bhttps?\.(get|request)\s*\(|\brequests\.(get|post)\s*\(|\burllib\b|\baxios\b/;

/** Shell commands a tool runs, from `command`, `commands[]`, or shell `code`. */
function commandsOf(input: Record<string, unknown>): string[] | undefined {
  if (typeof input.command === 'string') return [input.command];
  if (typeof input.code === 'string' && (input.language === 'shell' || input.language === 'bash')) return [input.code];
  if (Array.isArray(input.commands)) {
    const commands = input.commands.map((c) => (typeof c === 'string' ? c : (c as { command?: unknown })?.command));
    if (commands.every((c): c is string => typeof c === 'string')) return commands;
  }
  return undefined;
}

export interface CostRules {
  cheap: (tool: string) => boolean;
  expensive: (tool: string) => boolean;
}

/** User globs come first, so they win over the defaults on a conflict. */
export function costRules(cheapTools: readonly string[] = [], expensiveTools: readonly string[] = []): CostRules {
  const userCheap = toolMatcher(cheapTools);
  const userExpensive = toolMatcher(expensiveTools);
  const defaultCheap = toolMatcher(DEFAULT_CHEAP_TOOLS);
  const defaultExpensive = toolMatcher(DEFAULT_EXPENSIVE_TOOLS);
  return {
    cheap: (tool) => userCheap(tool) || (!userExpensive(tool) && defaultCheap(tool)),
    expensive: (tool) => userExpensive(tool) || (!userCheap(tool) && defaultExpensive(tool)),
  };
}

/**
 * How costly it would be to get a call's output again, by rule. `unknown`
 * calls are the ones a model is asked about. Shell-running tools (Bash, or
 * any tool whose input carries `command`/`commands`) are judged by command.
 */
export function ruleCost(call: Pick<ToolCall, 'tool' | 'input'>, rules: CostRules): RerunCost {
  if (rules.expensive(call.tool)) return 'expensive';
  if (rules.cheap(call.tool)) return 'cheap';
  // Code that reaches the network (e.g. a sandboxed `fetch`) cannot be re-run for free.
  if (typeof call.input.code === 'string' && NETWORK_CODE.test(call.input.code)) return 'expensive';
  const commands = commandsOf(call.input);
  if (!commands) return 'unknown';
  const costs = commands.map(commandCost);
  if (costs.includes('expensive')) return 'expensive';
  return costs.every((cost) => cost === 'cheap') ? 'cheap' : 'unknown';
}
