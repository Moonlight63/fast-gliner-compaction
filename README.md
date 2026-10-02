# fast-gliner-compaction

A Claude Code plugin for removal-first compaction. Old tool output is
truncated by default and kept only when getting it again would be expensive.
Rules decide the clear cases, and a self-hosted GLiNER2.5 classifier decides
the rest. Nothing is summarized or rewritten. User and assistant text stays
verbatim, and every tool call stays in place.

Forked from [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction).
That plugin asks TypeSafe's hosted Jev model which history is still needed;
this one asks a simpler, more checkable question about each tool call.

## Why removal-first

In our sessions, about 95% of old tool output was never referenced again (see
"Evaluation"). Removing it is also cheap to undo. A removed result leaves its
call and the first 300 characters in place, and the assistant can re-run the
tool. The real risk is losing output that cannot be re-run for free: a long
test run, a web fetch, a subagent's report, a user's answer, or a one-off
database snapshot. So the plugin asks one question: **would it be expensive to
get this output again?**

## How it works

1. Every `tool_use` is paired with its `tool_result`. Calls in the first
   message or the newest `preserveRecentMessages` messages are pinned and left
   alone.
2. Rules classify each remaining call:
   - **Cheap.** `Read`, `Grep`, `Glob`, `Edit`, `Write`, and similar local
     lookups and edit confirmations. Also shell commands made only of
     read-only or local programs (`grep`, `sed`, `cat`, `ls`, `git status`,
     `git log`, `docker logs`, and so on).
   - **Expensive.** `WebFetch`, `WebSearch`, `Agent`/`Task`,
     `AskUserQuestion`, MCP fetch and search tools, and code that calls
     `fetch`. Also shell commands with any test, build, install, network, or
     remote step (`npm test`, `npx`, `bun run`, `pytest`, `tsc`, `curl`,
     `ssh`, `git push`, `docker build`, …).
   - **Unknown.** Everything else, such as ad-hoc `node -e`/`python3`
     scripts, `docker exec`, and unfamiliar MCP tools.

   The shell parser understands quoting, pipes, heredocs, loops, `$(…)`, and
   functions. In our sessions the rules left only about 15% of calls to the
   model.
3. For **unknown** calls only, the model sees the tool, its input, and the
   first `focusResultChars` characters of its output. It is asked whether
   re-running would be expensive. The output is kept when P(expensive) ≥
   `spareThreshold`.
4. Each removed result keeps `truncateHeadChars` characters and a note. Long
   string inputs of its call (for example a `Write`'s file content) are
   shortened to `truncateInputChars`. Paths and commands stay whole.
5. If the decision server is down, compaction still runs on the rules. Unknown
   outputs are kept, and the toast says why. If the reduction is below
   `minReductionRatio`, the built-in summary runs instead.

Add your own rules with the `cheapTools` and `expensiveTools` options. They
take comma-separated tool-name globs such as `mcp__db__*` and win over the
built-in rules.

## Evaluation

`npm run eval -- <session.jsonl…>` writes `eval-report.md`. The report has
four parts:

- **Rule coverage.** How many calls the rules decide.
- **Model agreement on rule-labelled calls.** The model sees each call
  without its label. The report says whether the model agrees with the rules,
  with shell commands separated out, because there the model has to read the
  command.
- **A threshold sweep.**
- **Every call left to the model, with its score,** so you can review the
  model's judgement on exactly the calls it decides.

Results on 4 of our sessions (885 tool calls). Two of the sessions overlap,
because one resumes the other.

| | |
| --- | --- |
| Decided by rules | 85% (625 cheap, 131 expensive) |
| Left to the model | 15% (129 calls) |
| `gliner-decide-1b` agreement, AUC (all / shell / named tools) | 0.73 / 0.65 / 0.82 |
| At threshold 0.85: cheap correctly removed / expensive correctly kept | 93% / 18% |
| Characters freed, two long sessions | 53% and 58% |
| Characters freed, two short sessions | 76% |

What remains after compaction is mostly assistant text, which is never
touched, and the commands themselves.

The model is a weak veto. It removes nearly everything the rules would call
cheap, but keeps only a minority of what they would call expensive. The rules
do most of the work. On this question, the general `gliner2.5-base` and
`-multi` models did worse than chance. An earlier attempt to predict "will
this output be reused later" found no model that beat simply keeping recent
calls. That attempt is why the plugin asks about re-run cost instead.

## Models

The server exposes these names, and the plugin's `model` option picks one.

| Name | Checkpoint | Reads | Notes |
| --- | --- | --- | --- |
| `gliner-decide-1b` (default) | `fastino/GLiNER2.5-Decide-1B` (Ettin 1B) | 4096 (capped) | Best on the re-run-cost question; needs `server/compat.py` |
| `gliner-decide` | `fastino/GLiNER2.5-Decide` (340M, DeBERTa-v3-large) | 8192 | Slow on long inputs (eager attention) |
| `gliner-multi-decide` | `fastino/GLiNER2.5-multi-Decide` | 4096 | Multilingual |
| `gliner25-base` / `-multi` / `-small` | `fastino/gliner2.5-{base,multi,small}-v1` | 4096 | General-purpose GLiNER2.5; need `compat.py` |
| `laya` / `laya-multilingual` | `convaiinnovations/laya*` | 512 / 8192 | Failed the long-input needle test |

`python server/needle_test.py <model>` checks that a model reads to the end of
long inputs on your hardware.

### Checkpoints saved by transformers 5

gliner2 2.0 pins `transformers<5`, but Decide-1B and the `gliner2.5-*-v1`
checkpoints were saved by transformers 5. `server/compat.py` handles two
incompatibilities:

- The tokenizer configs list `extra_special_tokens` in the 5.x format, and
  Decide-1B's names a 5.x-only class.
- Decide-1B's ModernBERT `rope_parameters` are ignored by 4.x, so its
  sliding-window layers run at `rope_theta=10000` instead of 160000. The model
  then loads without error and answers ~0.5 to everything.

## Running the decision server

The server is optional. Without it, the plugin runs on rules alone and keeps
unknown outputs.

### On a GPU host (Docker)

```sh
cd server
FGC_TOKEN=$(openssl rand -hex 24) docker compose up -d --build
```

The compose file binds the server to the host's loopback interface. Reach it
through an SSH tunnel:

```sh
ssh -N -L 8765:127.0.0.1:8765 gpu-host
```

To serve the LAN instead, set `FGC_BIND=0.0.0.0` and always set a token.
`FGC_GPU` picks the device, and `FGC_MODELS` lists the models to preload.

### Virtualenv (CPU or GPU)

```sh
cd server
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
FGC_MODELS=gliner-decide-1b ./start.sh   # http://127.0.0.1:8765
```

`./start.sh` (re)starts the server detached and writes `server.pid` and
`server.log`. Items are short (one call each), so a CPU is workable.

| Server env | Default | Meaning |
| --- | --- | --- |
| `FGC_MODELS` | `gliner-decide-1b` | Models loaded at startup |
| `FGC_LAZY` | `1` | Load other known models on first request |
| `FGC_DEVICE` | `cuda` if available | `cuda`, `cuda:1`, or `cpu` |
| `FGC_HALF` | `1` | fp16 weights on CUDA |
| `FGC_BATCH_SIZE` | `16` | Items per forward pass |
| `FGC_TOKEN` | — | Bearer token required on `/v1/*` |
| `FGC_HOST` / `FGC_PORT` | `127.0.0.1` / `8765` | Bind address (virtualenv mode) |

API: `GET /health`, and `POST /v1/decide` with
`{model, questions: {id: text}, items: [text]}`, which returns
`{answers: [{id: P(yes)}], ms}`.

## Install in Claude Code

Function hooks are early access and require Claude Code 2.1.274+:

```json
{ "env": {
    "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1",
    "FGC_SERVER_URL": "http://127.0.0.1:8765",
    "FGC_TOKEN": "<token, if the server sets one>"
} }
```

```sh
claude plugin marketplace add Moonlight63/fast-gliner-compaction
claude plugin install fast-gliner-compaction@fast-gliner-compaction
```

The plugin's `serverUrl` and `serverToken` options override the environment.
To develop from a checkout: `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .`

| Option | Default | Description |
| --- | --- | --- |
| `serverUrl` | `FGC_SERVER_URL`, then `http://127.0.0.1:8765` | Decision server |
| `serverToken` | `FGC_TOKEN` | Bearer token |
| `model` | `gliner-decide-1b` | Model for calls the rules leave unknown |
| `spareThreshold` | `0.85` | Minimum P(expensive) to keep an unknown output |
| `cheapTools` / `expensiveTools` | — | Extra tool-name globs; win over built-in rules |
| `preserveRecentMessages` | `6` | Newest messages never touched |
| `compactAtPercent` | `60` | Context % that triggers compaction |
| `minReductionRatio` | `0.25` | Below this, fall back to the built-in summary |
| `truncateHeadChars` | `300` | Characters kept from a removed result |
| `truncateInputChars` | `300` | Characters kept of each long input field of its call |
| `focusResultChars` | `1500` | Result characters shown to the model |
| `maxRequestItems` | `128` | Items per server request |

## Development

```sh
npm install
npm run typecheck
npm test                 # fake decider, no server needed
npm run validate:plugin
```
