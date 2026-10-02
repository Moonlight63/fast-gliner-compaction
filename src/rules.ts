import type { CallCategory, ToolCall } from './types.js';

/**
 * Tools whose information lives in the workspace (or is a quick lookup away):
 * reads, searches, and edits, whose content is in the files themselves.
 * Globs: `*` matches any run of characters.
 */
export const DEFAULT_WORKSPACE_TOOLS: readonly string[] = [
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

/** Tools whose output is found information that is not in the workspace. */
export const DEFAULT_RESEARCH_TOOLS: readonly string[] = [
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

// Shell programs that only read, or change local files (their effect is in the workspace).
const WORKSPACE_PROGRAMS = new Set([
  'ls', 'cat', 'head', 'tail', 'grep', 'egrep', 'fgrep', 'rg', 'find', 'fd', 'tree', 'wc',
  'sort', 'uniq', 'cut', 'tr', 'diff', 'stat', 'file', 'du', 'df', 'pwd', 'echo', 'printf',
  'which', 'whereis', 'type', 'env', 'printenv', 'date', 'whoami', 'hostname', 'uname', 'id',
  'ps', 'pgrep', 'ss', 'jq', 'yq', 'sed', 'awk', 'basename', 'dirname', 'realpath', 'readlink',
  'test', 'true', 'false', 'free', 'nproc', 'nvidia-smi', 'mkdir', 'touch', 'cp', 'mv', 'rm',
  'rmdir', 'ln', 'chmod', 'kill', 'pkill', 'tee', 'xargs',
]);
// Programs that run something: their result is an outcome (pass/fail, output of a run).
const OUTCOME_PROGRAMS = new Set([
  'pytest', 'vitest', 'jest', 'mocha', 'playwright', 'tsc', 'vue-tsc', 'eslint', 'cargo', 'make',
  'gradle', 'mvn', 'dotnet', 'ssh', 'scp', 'rsync', 'sleep', 'npx', 'bunx',
]);
// Programs that fetch information from elsewhere.
const RESEARCH_PROGRAMS = new Set(['curl', 'wget', 'gh']);
const PACKAGE_MANAGERS = new Set(['npm', 'pnpm', 'yarn', 'bun']);
const PACKAGE_RUNS = new Set(['test', 'run', 'install', 'i', 'ci', 'build', 'exec', 'x', 'add', 'dlx']);
// Read-only git, plus local writes whose effect is in the repository.
const WORKSPACE_GIT = new Set([
  'status', 'diff', 'log', 'show', 'branch', 'remote', 'rev-parse', 'ls-files', 'blame', 'tag',
  'describe', 'add', 'commit', 'reset', 'checkout', 'switch', 'restore', 'stash', 'mv', 'rm',
  'init', 'config', 'grep', 'shortlog', 'reflog',
]);
const OUTCOME_GIT = new Set(['push', 'pull', 'fetch', 'clone']);
const WORKSPACE_DOCKER = new Set(['ps', 'logs', 'inspect', 'images', 'version', 'info']);
const OUTCOME_DOCKER = new Set(['build', 'compose', 'run', 'pull', 'push']);
const PIP = new Set(['pip', 'pip3', 'uv', 'poetry']);
/** Keywords that prefix a command without deciding what runs (`if grep …`). */
const SKIP_WORDS = new Set([
  'while', 'until', 'do', 'done', 'if', 'then', 'else', 'elif', 'fi', 'esac', '{', '}', '!',
]);
/** Builtins whose whole command is bookkeeping, arguments included. */
const BUILTINS = new Set([
  'cd', 'pushd', 'popd', 'set', 'export', 'local', 'return', 'exit', 'source', '.', 'unset',
  'shopt', 'trap', 'wait', 'read', 'shift',
]);

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

/** The category of one simple command (no pipes or separators). */
function segmentCategory(input: readonly string[]): CallCategory | 'skip' {
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
  // A function definition (`name(){ body`) is judged by its body.
  if (/^[A-Za-z_][A-Za-z0-9_-]*\(\)\{?$/.test(raw)) return segmentCategory(words.slice(1));
  const program = raw.split('/').pop() ?? '';
  let sub = words[1] ?? '';
  if (program === 'git' && sub === '-C') sub = words[3] ?? '';
  if (RESEARCH_PROGRAMS.has(program)) return 'research';
  if (OUTCOME_PROGRAMS.has(program)) return 'outcome';
  if (PACKAGE_MANAGERS.has(program)) return PACKAGE_RUNS.has(sub) ? 'outcome' : 'unknown';
  if (PIP.has(program)) return ['install', 'sync', 'add'].includes(sub) ? 'outcome' : 'unknown';
  if (program === 'git') {
    if (OUTCOME_GIT.has(sub)) return 'outcome';
    return WORKSPACE_GIT.has(sub) ? 'workspace' : 'unknown';
  }
  if (program === 'docker') {
    if (OUTCOME_DOCKER.has(sub)) return 'outcome';
    return WORKSPACE_DOCKER.has(sub) ? 'workspace' : 'unknown';
  }
  return WORKSPACE_PROGRAMS.has(program) ? 'workspace' : 'unknown';
}

const STRENGTH: Record<CallCategory, number> = { workspace: 0, unknown: 1, outcome: 2, research: 3 };

/**
 * The category of a shell command: the strongest of its simple commands
 * (research > outcome > unknown > workspace), so `ls && npm test` is an
 * outcome. Loop and function bodies and substitutions are judged command by
 * command; a call to a function defined in the same line is judged by its body.
 */
export function commandCategory(command: string): CallCategory {
  const segments = splitCommands(command);
  const functions = new Set(
    segments.flatMap((words) => words.filter((w) => /\(\)\{?$/.test(w)).map((w) => w.replace(/\(\)\{?$/, ''))),
  );
  let strongest: CallCategory | undefined;
  for (const words of segments) {
    const first = words.find((w) => !/^[A-Za-z_][A-Za-z0-9_]*=/.test(w) && !SKIP_WORDS.has(w));
    if (first && functions.has(first)) continue;
    const category = segmentCategory(words);
    if (category === 'skip') continue;
    if (!strongest || STRENGTH[category] > STRENGTH[strongest]) strongest = category;
  }
  return strongest ?? 'workspace';
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

export interface CategoryRules {
  remove: (tool: string) => boolean;
  keep: (tool: string) => boolean;
  workspace: (tool: string) => boolean;
  research: (tool: string) => boolean;
}

export function categoryRules(removeTools: readonly string[] = [], keepTools: readonly string[] = []): CategoryRules {
  return {
    remove: toolMatcher(removeTools),
    keep: toolMatcher(keepTools),
    workspace: toolMatcher(DEFAULT_WORKSPACE_TOOLS),
    research: toolMatcher(DEFAULT_RESEARCH_TOOLS),
  };
}

/**
 * Where a call's information lives, by rule:
 * - `workspace`: in the files or a quick local lookup (reads, edits, searches);
 * - `outcome`: the result of running something (tests, builds, scripts, remote);
 * - `research`: found elsewhere (web, subagents, the user, network code);
 * - `unknown`: none of the above; a model is asked.
 */
export function categorize(call: Pick<ToolCall, 'tool' | 'input'>, rules: CategoryRules): CallCategory {
  if (rules.research(call.tool)) return 'research';
  if (rules.workspace(call.tool)) return 'workspace';
  if (typeof call.input.code === 'string' && NETWORK_CODE.test(call.input.code)) return 'research';
  const commands = commandsOf(call.input);
  if (!commands) return 'unknown';
  return commands
    .map(commandCategory)
    .reduce((a, b) => (STRENGTH[b] > STRENGTH[a] ? b : a), 'workspace' as CallCategory);
}

/**
 * What a call is about, for spotting a later call that supersedes it: a file
 * path, a command, a URL, or a query. Undefined when nothing identifies it.
 */
export function callTarget(call: Pick<ToolCall, 'input'>): string | undefined {
  const input = call.input;
  for (const key of ['file_path', 'notebook_path', 'path']) {
    if (typeof input[key] === 'string') return `file:${input[key]}`;
  }
  const commands = commandsOf(input);
  if (commands) return `cmd:${commands.join(' ; ').replace(/\s+/g, ' ').trim()}`;
  if (typeof input.url === 'string') return `url:${input.url}`;
  if (typeof input.query === 'string') return `query:${input.query}`;
  return undefined;
}
