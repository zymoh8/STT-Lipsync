"""Vosk (Kaldi). pip install vosk"""
from __future__ import annotations

import json
import wave

from .base import Adapter, register


@register
class VoskAdapter(Adapter):
    name = "vosk"

    def __init__(self, model_path: str, sample_rate: int = 16000,
                 grammar: list[str] | None = None, **kw):
        super().__init__(**kw)
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.grammar = grammar
        self._model = None

    def prepare(self) -> None:
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        self._model = Model(self.model_path)

    def _transcribe(self, wav_path: str) -> str:
        from vosk import KaldiRecognizer
        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                raise ValueError(
                    f"{wav_path}: need 16-bit mono. Convert the corpus once "
                    f"with tools/convert_audio.sh rather than resampling "
                    f"inside the loop - resampling cost would pollute RTF.")
            rate = wf.getframerate()
            if self.grammar:
                rec = KaldiRecognizer(self._model, rate,
                                      json.dumps(self.grammar))
            else:
                rec = KaldiRecognizer(self._model, rate)
            rec.SetWords(False)
            out = []
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    out.append(json.loads(rec.Result()).get("text", ""))
            out.append(json.loads(rec.FinalResult()).get("text", ""))
        return " ".join(x for x in out if x).strip()
