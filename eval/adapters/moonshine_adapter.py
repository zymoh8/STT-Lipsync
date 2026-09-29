"""
Moonshine (Useful Sensors) - built for edge devices, short utterances.

    pip install useful-moonshine-onnx

Models: moonshine/tiny, moonshine/base
Designed for exactly your workload: short dictated utterances, low
latency, small footprint. Strong candidate if objective 2 pushes you to a
Zero-class board.
"""
from __future__ import annotations

from .base import Adapter, register


@register
class MoonshineAdapter(Adapter):
    name = "moonshine"

    def __init__(self, model: str = "moonshine/base", **kw):
        super().__init__(**kw)
        self.model = model
        self._mod = None

    def prepare(self) -> None:
        try:
            import moonshine_onnx as m  # type: ignore
        except ImportError:
            import moonshine as m  # type: ignore
        self._mod = m

    def _transcribe(self, wav_path: str) -> str:
        out = self._mod.transcribe(wav_path, self.model)
        if isinstance(out, (list, tuple)):
            out = " ".join(str(x) for x in out)
        return str(out).strip()
