"""
Model adapter interface.

Adding an engine = writing ONE file in this folder that subclasses
Adapter and registers itself. Nothing else in the harness changes.

Every adapter must:
  * accept 16 kHz mono WAV paths
  * return plain text (punctuation and casing are fine - the normaliser
    strips them; do NOT pre-normalise inside an adapter, or you will hide
    a real difference between engines)
  * report peak RSS and wall time for the transcription only, not model
    loading - load happens once in prepare()
"""
from __future__ import annotations

import time
import wave
from dataclasses import dataclass


@dataclass
class Result:
    text: str
    seconds: float          # wall time for transcription
    audio_seconds: float    # duration of the input
    peak_rss_mb: float = 0.0

    @property
    def rtf(self) -> float:
        """Real-time factor. <1.0 means faster than real time.

        Your use case is utterance-at-a-time dictation, not streaming, so
        an RTF of 2-3 is usable (speak a sentence, wait a beat). Report it
        anyway - it is half of the accuracy-per-resource story that makes
        this project a contribution rather than a leaderboard."""
        return self.seconds / self.audio_seconds if self.audio_seconds else 0.0


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def peak_rss_mb() -> float:
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS bytes
        import sys
        return kb / 1024.0 if sys.platform.startswith("linux") else kb / 1048576.0
    except Exception:
        return 0.0


class Adapter:
    name = "base"

    def __init__(self, **kwargs):
        self.config = kwargs

    def prepare(self) -> None:
        """Load the model. Called once. Not timed."""

    def _transcribe(self, wav_path: str) -> str:
        raise NotImplementedError

    def transcribe(self, wav_path: str) -> Result:
        dur = wav_duration(wav_path)
        t0 = time.perf_counter()
        text = self._transcribe(wav_path)
        dt = time.perf_counter() - t0
        return Result(text=text or "", seconds=dt, audio_seconds=dur,
                      peak_rss_mb=peak_rss_mb())


REGISTRY: dict[str, type[Adapter]] = {}


def register(cls: type[Adapter]) -> type[Adapter]:
    REGISTRY[cls.name] = cls
    return cls


def build(spec: dict) -> Adapter:
    """spec = {"engine": "vosk", "label": "vosk-0.22", ...kwargs}"""
    spec = dict(spec)
    engine = spec.pop("engine")
    label = spec.pop("label", engine)
    if engine not in REGISTRY:
        raise KeyError(f"unknown engine {engine!r}; have {sorted(REGISTRY)}")
    a = REGISTRY[engine](**spec)
    a.label = label  # type: ignore[attr-defined]
    return a
