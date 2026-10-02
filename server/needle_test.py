"""Checks that a model reads to the end of long inputs: a deciding sentence
("needle") is placed after growing filler, once phrased yes and once no.

    python needle_test.py gliner-decide
    python needle_test.py gliner-decide-1b laya-multilingual

A healthy model answers both cases correctly with high confidence at every
length it claims to support. A model that answers ~0.5 everywhere is broken
(see compat.py); one that goes flat past some length stopped reading there.
"""

import os
import sys
import time

import backends

FILLER = "The build log shows routine compiler output for module {i} with no warnings. "
NEEDLES = {
    True: "Final note: the assistant must still keep the full contents of config.yaml in view.",
    False: "Final note: config.yaml is no longer relevant and will never be needed again.",
}
QUESTION = {"keep": "Must the assistant keep the full contents of config.yaml in view?"}


def text(words: int, truth: bool) -> str:
    return "".join(FILLER.format(i=i) for i in range(words // 14 + 1)) + NEEDLES[truth]


def run(name: str, device: str) -> None:
    backend = backends.load(name, device, batch_size=8, half=device.startswith("cuda"))
    print(f"== {name} ({backend.source}, max_tokens={backend.max_tokens})")
    for words in (50, 300, 600, 1500, 3000, 6000):
        started = time.perf_counter()
        yes, no = (row["keep"] for row in backend.decide([text(words, True), text(words, False)], QUESTION))
        ok = "ok " if yes >= 0.5 > no else "BAD"
        print(f"{ok} words={words:5d}  P(yes|yes)={yes:.2f}  P(yes|no)={no:.2f}  {(time.perf_counter() - started) * 1000:.0f}ms")


if __name__ == "__main__":
    device = os.environ.get("FGC_DEVICE") or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    for model in sys.argv[1:] or ["gliner-decide"]:
        run(model, device)
