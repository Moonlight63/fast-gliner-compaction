"""Decision backends: each turns (texts, yes/no questions) into P(yes) per question.

Every backend answers the same shape so the plugin and the eval harness can
swap models by name.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List

import compat

# name -> (kind, hub id or laya checkpoint, max input tokens the model reads well)
MODELS: Dict[str, tuple] = {
    "gliner-decide": ("gliner", "fastino/GLiNER2.5-Decide", 8192),
    "gliner-decide-1b": ("gliner", "fastino/GLiNER2.5-Decide-1B", 4096),
    "gliner-multi-decide": ("gliner", "fastino/GLiNER2.5-multi-Decide", 4096),
    # General-purpose GLiNER2.5 checkpoints (extraction + classification heads).
    "gliner25-base": ("gliner", "fastino/gliner2.5-base-v1", 4096),
    "gliner25-multi": ("gliner", "fastino/gliner2.5-multi-v1", 4096),
    "gliner25-small": ("gliner", "fastino/gliner2.5-small-v1", 4096),
    "laya": ("laya", "english", 512),
    "laya-multilingual": ("laya", "multilingual", 8192),
}


@dataclass
class Backend:
    name: str
    kind: str
    source: str
    max_tokens: int
    device: str

    def decide(self, texts: List[str], questions: Dict[str, str]) -> List[Dict[str, float]]:
        raise NotImplementedError


class GlinerBackend(Backend):
    def __init__(self, name, kind, source, max_tokens, device, batch_size: int, half: bool):
        super().__init__(name, kind, source, max_tokens, device)
        compat.apply()
        from gliner2 import AutoExtractor

        kwargs = {"map_location": device}
        if half and device.startswith("cuda"):
            kwargs["quantize"] = True
        self.model = AutoExtractor.from_pretrained(source, **kwargs)
        self.batch_size = batch_size
        self.lock = threading.Lock()

    def decide(self, texts, questions):
        tasks = {qid: {"labels": ["yes", "no"], "prompt": prompt} for qid, prompt in questions.items()}
        with self.lock:
            results = self.model.batch_classify_text(
                texts,
                tasks,
                batch_size=self.batch_size,
                include_confidence=True,
                max_len=self.max_tokens,
            )
        answers = []
        for result in results:
            row = {}
            for qid in questions:
                answer = result[qid]
                confidence = float(answer["confidence"])
                # Two labels share one softmax, so P(yes) is the complement when "no" wins.
                row[qid] = confidence if answer["label"] == "yes" else 1.0 - confidence
            answers.append(row)
        return answers


class LayaBackend(Backend):
    def __init__(self, name, kind, source, max_tokens, device, batch_size: int, half: bool):
        super().__init__(name, kind, source, max_tokens, device)
        from laya import Router

        self.router = Router(preload=False)
        self.batch_size = batch_size
        self.lock = threading.Lock()

    def decide(self, texts, questions):
        typed = {qid: {"type": "noul", "instructions": prompt} for qid, prompt in questions.items()}
        requests = [
            {"state": text, "questions": typed, "model": self.source, "max_len": self.max_tokens}
            for text in texts
        ]
        with self.lock:
            results = self.router.predict_batch(requests, batch_size=self.batch_size)
        return [
            {qid: float(result["answers"][qid]["noul"]) for qid in questions} for result in results
        ]


def load(name: str, device: str, batch_size: int, half: bool) -> Backend:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    kind, source, max_tokens = MODELS[name]
    cls = GlinerBackend if kind == "gliner" else LayaBackend
    return cls(name, kind, source, max_tokens, device, batch_size, half)
