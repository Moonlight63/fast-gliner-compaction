# fast-gliner-compaction

Claude Code plugin that replaces the compaction summary with keep/drop
decisions from a self-hosted GLiNER2.5 classifier. Every tool call is scored;
stale results are truncated or dropped, and everything kept stays verbatim.
Forked from [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction),
with TypeSafe's hosted Jev model replaced by a decision server you run yourself,
on a CPU or a GPU host.

## Why

A summary is lossy: a file path, exact error, or constraint can disappear even
when it matters later. This plugin never rewrites anything. It only removes
tool calls and tool results that the model scores as no longer needed. User and
assistant text stays verbatim and in order.

## How it works

1. Every `tool_use` is paired with its `tool_result`. Calls in the first message
   or the newest `preserveRecentMessages` messages are pinned and never touched.
2. An **outline** of the whole conversation is built once. Tool results are
   replaced by a one-line note, and the outline is shrunk in stages to fit
   `maxStateTokens` (3,000 by default; the stages are unchanged from upstream).
3. Each candidate call becomes one **item**: a focus block with the call's input
   and the first `focusResultChars` characters of its output, followed by the
   outline. The focus block comes first, so a model that truncates loses
   outline, never the call.
4. Every item is asked the same two yes/no questions, `keep_call` and
   `keep_result`. Because the questions are identical, the server can batch all
   items through one forward pass.
5. Decisions against `keepThreshold`: keep the result if it scores at or above
   the threshold; otherwise keep the call and truncate the result to
   `truncateHeadChars`; otherwise drop the call and its result.

Why one item per call instead of fast-jev's single 25k-token state: these
encoders read up to 4k–8k tokens, and cost grows quadratically with length.

If the server fails, the response is malformed, or the reduction is below
`minReductionRatio`, the hook falls back to Claude Code's built-in summary.

## Models

The server exposes these names. The plugin's `model` option picks one, and the
server loads it on first use.

| Name | Checkpoint | Reads | Notes |
| --- | --- | --- | --- |
| `gliner-decide` (default) | `fastino/GLiNER2.5-Decide` (340M, DeBERTa-v3-large) | 8192 | Sharpest in the long-context needle test; ~2 GB VRAM in fp16 |
| `gliner-decide-1b` | `fastino/GLiNER2.5-Decide-1B` (Ettin 1B) | 4096 (capped) | Needs `server/compat.py` (see below); ~2.5× slower, softer past ~4k tokens |
| `gliner-multi-decide` | `fastino/GLiNER2.5-multi-Decide` | 4096 | Multilingual transcripts |
| `gliner25-base` / `gliner25-multi` / `gliner25-small` | `fastino/gliner2.5-{base,multi,small}-v1` | 4096 | General-purpose GLiNER2.5; also need `compat.py` |
| `laya` | `convaiinnovations/laya` (ModernBERT 421M) | 512 | Failed the needle test even at 50 words |
| `laya-multilingual` | `laya-multilingual` (mmBERT 322M) | 8192 | Could not separate yes from no at 300 words or more |

Run `python server/needle_test.py <model>` to check a model on your hardware.
`npm run eval -- <transcripts.jsonl…>` compares models on real sessions. It
labels a dropped result as "needed" when its content reappears later in the
session, and reports AUC against recency and size baselines.

### First results (zero-shot, 2026-10-01)

The run covered 4 Claude Code sessions, each cut at 40/60/80%, for about 1,480
decisions, of which 64–68 were "needed." Scores are AUC: 0.5 is chance. AUC is
controlled for result size and position, because the proxy label favors long
outputs. The size column is controlled for size only.

| Model | Size + position | Size only | Median P(keep) |
| --- | ---: | ---: | ---: |
| baseline: keep newer calls | — | 0.66 | — |
| `gliner25-multi` | **0.64** | 0.67 | 0.15 |
| `laya` | 0.59 | 0.54 | 0.79 |
| `gliner25-base` | 0.53 | 0.52 | 0.68 |
| `gliner-decide-1b` | 0.50 | 0.51 | 0.46 |
| `gliner25-small` | 0.47 | 0.45 | 0.85 |
| `laya-multilingual` | 0.44 | 0.48 | 0.95 |
| `gliner-decide` | 0.42 | 0.43 | 0.23 |

- Only `gliner25-multi` carries a signal independent of recency. Its score has
  no correlation with position (Spearman 0.04).
- No model's 0.5 threshold is usable as is: the medians range from 0.15 to
  0.95, so each needs recalibration.
- With so few positives, treat differences under about 0.07 as noise.

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
FGC_MODELS=gliner-decide python app.py   # http://127.0.0.1:8765
```

`./start.sh` (re)starts the server detached, using the same environment
variables. The server writes its PID to `server.pid` and its output to
`server.log`.

On a CPU, lower `maxStateTokens` (for example to 1500), because cost grows
quadratically with item length.

| Server env | Default | Meaning |
| --- | --- | --- |
| `FGC_MODELS` | `gliner-decide` | Models loaded at startup |
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
claude plugin marketplace add ~/Documents/fast-gliner-compaction
claude plugin install fast-gliner-compaction@fast-gliner-compaction
```

The plugin's `serverUrl` and `serverToken` options override the environment.
To develop from a checkout: `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .`

| Option | Default | Description |
| --- | --- | --- |
| `serverUrl` | `FGC_SERVER_URL`, then `http://127.0.0.1:8765` | Decision server |
| `serverToken` | `FGC_TOKEN` | Bearer token |
| `model` | `gliner-decide` | Server model name |
| `keepThreshold` | `0.5` | Minimum keep probability |
| `preserveRecentMessages` | `6` | Newest messages never touched |
| `compactAtPercent` | `60` | Context % that triggers compaction |
| `minReductionRatio` | `0.25` | Below this, fall back to the built-in summary |
| `maxStateTokens` | `3000` | Outline budget per item |
| `focusResultChars` | `1500` | Result characters shown in the focus block |
| `maxRequestItems` | `128` | Items per server request |
| `truncateHeadChars` | `300` | Characters kept from a dropped result |

## Development

```sh
npm install
npm run typecheck
npm test                 # fake decider, no server needed
npm run validate:plugin
```

## Limitations

- Zero-shot, no model has yet beaten a "keep newer calls" baseline on its
  own, and none is calibrated (see "First results"). Use the eval harness
  before trusting any default.
- Only tool calls and results are candidates; text messages are never removed.
- Token sizes are estimates; the server truncates at each model's limit.
