"""
whisper.cpp via its CLI binary.

Build once on the Pi:
    git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
    bash ./models/download-ggml-model.sh base.en

Then point `binary` at build/bin/whisper-cli (older builds: ./main).

Model sizes worth testing on a 4GB Pi 4, smallest first:
    tiny.en   ~75 MB
    base.en   ~142 MB
    small.en  ~466 MB   <- viable because you are not streaming
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

from .base import Adapter, register

_TIMESTAMP = re.compile(r"^\[[\d:.\s\->]+\]\s*")


@register
class WhisperCppAdapter(Adapter):
    name = "whispercpp"

    def __init__(self, binary: str, model: str, threads: int = 4,
                 language: str = "en", extra_args: list[str] | None = None,
                 **kw):
        super().__init__(**kw)
        self.binary = binary
        self.model = model
        self.threads = threads
        self.language = language
        self.extra_args = extra_args or []

    def prepare(self) -> None:
        for p in (self.binary, self.model):
            if not os.path.exists(p):
                raise FileNotFoundError(p)

    def _transcribe(self, wav_path: str) -> str:
        with tempfile.TemporaryDirectory() as td:
            out_prefix = os.path.join(td, "out")
            cmd = [
                self.binary, "-m", self.model, "-f", wav_path,
                "-t", str(self.threads), "-l", self.language,
                "-otxt", "-of", out_prefix, "-np", "-nt",
            ] + self.extra_args
            proc = subprocess.run(cmd, capture_output=True, text=True)
            txt_path = out_prefix + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, encoding="utf-8", errors="ignore") as fh:
                    raw = fh.read()
            else:
                raw = proc.stdout
            lines = [_TIMESTAMP.sub("", ln).strip() for ln in raw.splitlines()]
            return " ".join(ln for ln in lines if ln).strip()
