"""HTTP decision server for fast-gliner-compaction.

    POST /v1/decide
      {"model": "gliner-decide-1b",
       "questions": {"expensive": "..."},
       "items": ["state text 1", "state text 2", ...]}
    -> {"model": "...", "answers": [{"expensive": 0.91}, ...], "ms": 41}

    GET /health -> loaded models, devices, known models

Environment:
    FGC_MODELS      models to load at startup (comma list, default gliner-decide-1b)
    FGC_LAZY        load other known models on first request (default 1)
    FGC_DEVICE      cuda, cuda:1, cpu (default: cuda when available)
    FGC_HALF        fp16 weights on CUDA (default 1)
    FGC_BATCH_SIZE  items per forward pass (default 16)
    FGC_TOKEN       bearer token required on /v1/* when set
    FGC_MAX_ITEMS   largest accepted request (default 2048)
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from typing import Dict, List

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import backends


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _device() -> str:
    configured = os.environ.get("FGC_DEVICE", "").strip()
    if configured:
        return configured
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _device()
HALF = _flag("FGC_HALF", "1")
LAZY = _flag("FGC_LAZY", "1")
BATCH_SIZE = int(os.environ.get("FGC_BATCH_SIZE", "16"))
MAX_ITEMS = int(os.environ.get("FGC_MAX_ITEMS", "2048"))
TOKEN = os.environ.get("FGC_TOKEN", "")

loaded: Dict[str, backends.Backend] = {}
loading = threading.Lock()


def get_backend(name: str) -> backends.Backend:
    backend = loaded.get(name)
    if backend:
        return backend
    if name not in backends.MODELS:
        raise HTTPException(404, f"unknown model {name!r}; known: {', '.join(backends.MODELS)}")
    if not LAZY:
        raise HTTPException(409, f"model {name!r} is not loaded and FGC_LAZY=0")
    with loading:
        if name not in loaded:
            loaded[name] = backends.load(name, DEVICE, BATCH_SIZE, HALF)
    return loaded[name]


def require_token(authorization: str = Header(default="")) -> None:
    if not TOKEN:
        return
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, TOKEN):
        raise HTTPException(401, "invalid or missing bearer token")


class DecideRequest(BaseModel):
    model: str = "gliner-decide-1b"
    questions: Dict[str, str] = Field(min_length=1)
    items: List[str]


app = FastAPI(title="fast-gliner-compaction server")


@app.on_event("startup")
def preload() -> None:
    for name in filter(None, (n.strip() for n in os.environ.get("FGC_MODELS", "gliner-decide-1b").split(","))):
        loaded[name] = backends.load(name, DEVICE, BATCH_SIZE, HALF)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "device": DEVICE,
        "half": HALF,
        "lazy": LAZY,
        "auth": bool(TOKEN),
        "loaded": {name: {"kind": b.kind, "source": b.source, "max_tokens": b.max_tokens} for name, b in loaded.items()},
        "known": list(backends.MODELS),
    }


@app.post("/v1/decide", dependencies=[Depends(require_token)])
async def decide(request: DecideRequest) -> dict:
    if len(request.items) > MAX_ITEMS:
        raise HTTPException(413, f"{len(request.items)} items exceeds FGC_MAX_ITEMS={MAX_ITEMS}")
    backend = await run_in_threadpool(get_backend, request.model)
    started = time.perf_counter()
    answers = await run_in_threadpool(backend.decide, request.items, request.questions) if request.items else []
    return {
        "model": backend.name,
        "answers": answers,
        "ms": round((time.perf_counter() - started) * 1000),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("FGC_HOST", "127.0.0.1"), port=int(os.environ.get("FGC_PORT", "8765")))
