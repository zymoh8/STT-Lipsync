"""Adapters share transcribe(audio, sample_id) -> str; add models here."""

import json
from pathlib import Path
from typing import Protocol

from .audio import Audio


class STTModel(Protocol):
    name: str
    metadata: dict

    def transcribe(self, audio: Audio, sample_id: str) -> str: ...


class VoskModel:
    def __init__(self, name: str, model_path: Path):
        if not model_path.is_dir():
            raise ValueError(f"Vosk model directory does not exist: {model_path}")
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError('Install Vosk with: python3 -m pip install -e ".[vosk]"') from exc
        from importlib.metadata import version
        self.name = name
        self.metadata = {"adapter": "vosk", "model_path": str(model_path.resolve()),
                         "vosk_version": version("vosk")}
        self._recognizer = vosk.KaldiRecognizer
        self._model = vosk.Model(str(model_path.resolve()))

    def transcribe(self, audio: Audio, sample_id: str) -> str:
        recognizer = self._recognizer(self._model, audio.sample_rate)
        segments = []
        for chunk in audio.chunks():
            if recognizer.AcceptWaveform(chunk):
                segments.append(json.loads(recognizer.Result()).get("text", ""))
        segments.append(json.loads(recognizer.FinalResult()).get("text", ""))
        return " ".join(segment for segment in segments if segment).strip()


class PredictionsModel:
    """Score saved outputs or exercise the pipeline without an installed model."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.metadata = {"adapter": "predictions", "path": str(path.resolve()),
                         "timing_is_inference": False}
        self.predictions = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(self.predictions, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.predictions.items()
        ):
            raise ValueError("Predictions must be a JSON object mapping sample IDs to text")

    def transcribe(self, audio: Audio, sample_id: str) -> str:
        if sample_id not in self.predictions:
            raise ValueError(f"No prediction for sample {sample_id!r}")
        return self.predictions[sample_id]
