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


class TransformersModel:
    """Local float32 CPU inference, using the benchmark's decoded PCM.

    CPU is explicit so timings are synchronous and comparable on this Mac.
    Dependencies are optional and weights are never downloaded during evaluation.
    """

    model_class: str
    family: str

    def __init__(self, name: str, model_path: Path):
        model_path = model_path.resolve()
        if not model_path.is_dir():
            raise ValueError(f"Model directory does not exist: {model_path}")
        try:
            import transformers
            import torch
        except ImportError as exc:
            raise RuntimeError('Install adapters with: python -m pip install -e ".[transformers]"') from exc
        from importlib.metadata import version
        processor = transformers.AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = getattr(transformers, self.model_class).from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32,
        )
        model.eval()
        self._pipeline = transformers.pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            device="cpu",
        )
        self.name = name
        self.metadata = {
            "adapter": self.family, "model_path": str(model_path),
            "transformers_version": version("transformers"),
            "torch_version": version("torch"), "device": "cpu", "dtype": "float32",
        }

    def transcribe(self, audio: Audio, sample_id: str) -> str:
        import numpy as np
        if audio.sample_rate != 16000 or audio.channels != 1 or audio.sample_width != 2:
            raise ValueError("Transformers adapters require 16000 Hz, 16-bit mono PCM")
        # Signed little-endian PCM -> normalized floating point waveform.
        waveform = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768.0
        options = {}
        if self.family == "whisper":
            options = {"return_timestamps": True, "generate_kwargs": {"task": "transcribe"}}
        elif self.family == "wav2vec2":
            options = {"chunk_length_s": 30, "stride_length_s": 5}
        result = self._pipeline({"raw": waveform, "sampling_rate": audio.sample_rate}, **options)
        return result["text"].strip()


class WhisperModel(TransformersModel):
    family = "whisper"
    model_class = "AutoModelForSpeechSeq2Seq"


class Wav2Vec2Model(TransformersModel):
    family = "wav2vec2"
    model_class = "AutoModelForCTC"


class ParakeetModel(TransformersModel):
    family = "parakeet"
    model_class = "AutoModelForTDT"


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
