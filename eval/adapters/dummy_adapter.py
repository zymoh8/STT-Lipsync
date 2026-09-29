"""
Fake engines for smoke-testing the harness with no models installed.

`dummy` reads the reference from a sidecar .txt next to the wav and
corrupts it at a configurable rate, so you can prove the whole
pipeline - manifest, adapter, scoring, report - works before you spend
an hour of Pi time on a real model.

If a real run produces numbers that look odd, come back here: run the
dummy at error_rate 0.0 and confirm the harness reports 0.00% WER. If it
does not, the bug is in normalisation or the manifest, not the engine.
"""
from __future__ import annotations

import os
import random

from .base import Adapter, register


@register
class DummyAdapter(Adapter):
    name = "dummy"

    def __init__(self, error_rate: float = 0.0, seed: int = 0,
                 style: str = "plain", **kw):
        super().__init__(**kw)
        self.error_rate = error_rate
        self.seed = seed
        # "plain" = lowercase bare words (vosk-like)
        # "rich"  = capitals, punctuation, digits (whisper-like)
        # Running both at error_rate 0 must give identical WER. If it
        # does not, normalisation is broken.
        self.style = style

    def _transcribe(self, wav_path: str) -> str:
        side = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.exists(side):
            return ""
        with open(side, encoding="utf-8") as fh:
            words = fh.read().strip().split()
        rng = random.Random(self.seed + hash(wav_path) % 10000)
        out = []
        for w in words:
            if rng.random() < self.error_rate:
                choice = rng.random()
                if choice < 0.5:
                    out.append("zzz")      # substitution
                elif choice < 0.8:
                    continue               # deletion
                else:
                    out.extend([w, "zzz"])  # insertion
            else:
                out.append(w)
        text = " ".join(out)
        if self.style == "rich":
            text = text.capitalize() + "."
            text = text.replace(" seven ", " 7 ")
        return text
