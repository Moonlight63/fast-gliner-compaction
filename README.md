# fast-gliner-compaction

A Claude Code plugin for removal-first compaction. Each old tool call is
judged by **how useful its information still is going forward**. Calls whose
information lives in the workspace, or that a later call has superseded, are
removed entirely. The latest outcome of each run is trimmed. Research is
kept. A self-hosted GLiNER2.5 classifier judges what the rules can't place.
Nothing is summarized or rewritten, and user and assistant text stays
verbatim.

Forked from [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction),
which uses TypeSafe's hosted Jev model to decide what to drop.

## The idea

What matters is not how expensive a call was, but whether its information is
still needed and whether the workspace already holds it. A `Write` may have
taken effort to produce, but its content is now in the file, and reading the
file back is cheap. In our sessions, 0 of 231 old `Write`/`Edit` outputs were
ever referenced again.

| Kind of call | What happens | Why |
| --- | --- | --- |
| **Workspace:** `Read`, `Write`, `Edit`, `Grep`, `Glob`, read-only or local shell (`grep`, `sed`, `cat`, `ls`, `git status/add/commit`, `cp`…) | **Removed**, call and result | The files hold the information; re-read when needed |
| **Superseded:** a later call targets the same file, command, URL or query | **Removed** | The newer call has the current state |
| **Outcome:** running something, such as tests, builds, installs, `npx`, `ssh`, `git push`, `docker build` | Latest run **trimmed** to its first and last 300 characters | Pass/fail and the summary are usually at the ends |
| **Research:** `WebFetch`, `WebSearch`, subagents, `AskUserQuestion`, MCP fetch/search, `curl`, `gh`, code that calls `fetch` | **Kept** | Not in the workspace; expensive or impossible to re-derive |
| **Unknown:** ad-hoc `node`/`python3` scripts, `docker exec`, unfamiliar MCP tools | **Model decides:** kept if P(still useful) ≥ `keepThreshold`, otherwise trimmed | — |

The first message and the newest `preserveRecentMessages` messages are never
touched.

Each run of removed calls leaves a single trace line, so the assistant still
knows what it touched:

```
[fast-gliner-compaction removed 7 tool calls: Read src/auth/session.ts, Edit src/auth/session.ts ×3, Write src/auth/new.ts, Bash git add -A]
```

The shell parser understands quoting, pipes, heredocs, loops, `$(…)`, and
functions. A command takes the strongest category of its parts, so
`ls && npm test` counts as an outcome. Add your own rules with `removeTools`
and `keepTools`, which take comma-separated tool-name globs such as
`mcp__db__*`.

If the decision server is down, compaction still runs on the rules, and
unknown outputs are trimmed.

## Evaluation

`npm run eval -- <session.jsonl…>` cuts each session at 40/60/80% and
compacts what came before the cut.

- **Used later.** An output counts as "used later" when one of its
  distinctive lines reappears after the cut in the assistant's text or tool
  inputs. For example, the assistant quoted it, or edited code it had read.
- **Loss.** For each used-later output, the eval checks whether that line
  survived compaction. That is the loss that matters.
- **Report.** `eval-report.md` lists every lost output and every call the
  model decided.

Results on 3 of our sessions × 3 cut points (827 tool outputs):

| | |
| --- | --- |
| Characters freed | **76–78%** on a long session, **78–94%** on short ones |
| Outputs used later | 39 (4.7%) |
| …whose used information did not survive | 32 |
| …of which re-readable file content (`Read`, `sed -n`, `cat`, `grep`, `Glob`) | 30, plus 2 directory listings the model trimmed |
| Research or outcome information lost | 0 |

| Decision | Outputs | Used later | Lost |
| --- | ---: | ---: | ---: |
| remove: workspace | 502 | 33 | 27 |
| remove: superseded | 171 | 3 | 3 |
| trim: outcome | 72 | 0 | 0 |
| trim: model | 68 | 3 | 2 |
| keep: research | 14 | 0 | 0 |

So the price is roughly one extra file read per compaction, in exchange for
removing about four-fifths of the context. The label only sees text that was
reused verbatim, so treat these losses as a lower bound.

Earlier attempts are recorded in the git history. Asking models "will this
output be needed later?" directly gave chance-level results. "Is it
expensive to re-run?" was objective but kept the wrong things.

## Models

The server exposes these names, and the plugin's `model` option picks one.
The model is consulted only for calls the rules can't place, about 10% of
calls in our sessions.

| Name | Checkpoint | Reads | Notes |
| --- | --- | --- | --- |
| `gliner-decide-1b` (default) | `fastino/GLiNER2.5-Decide-1B` (Ettin 1B) | 4096 (capped) | Needs `server/compat.py` |
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

The server is optional. Without it, the plugin runs on rules alone and trims
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
`server.log`. Each item is a single call, so a CPU is workable.

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
| `model` | `gliner-decide-1b` | Model for calls the rules can't place |
| `keepThreshold` | `0.85` | Minimum P(still useful) to keep an unknown output whole |
| `removeTools` / `keepTools` | — | Tool-name globs that override the rules |
| `breadcrumbs` | `true` | Leave a trace line where calls were removed |
| `preserveRecentMessages` | `6` | Newest messages never touched |
| `compactAtPercent` | `60` | Context % that triggers compaction |
| `minReductionRatio` | `0.25` | Below this, fall back to the built-in summary |
| `trimHeadChars` / `trimTailChars` | `300` / `300` | Kept from each end of a trimmed result |
| `trimInputChars` | `300` | Kept of each long input field of a trimmed call |
| `focusResultChars` | `1500` | Result characters shown to the model |
| `maxRequestItems` | `128` | Items per server request |

## Development

```sh
npm install
npm run typecheck
npm test                 # vitest specs (tests/*.spec.ts), fake decider, no server needed
npm run test:engine      # tests/engine/*.test.ts inside Claude Code's own plugin runtime
npm run validate:plugin
```
