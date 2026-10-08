import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from .audio import load_audio
from .metrics import Counts, score
from .models import STTModel


def load_manifest(path: Path) -> list[dict]:
    """JSONL paths resolve relative to the manifest, independent of current cwd."""
    samples = []
    ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str)
                                            for k in ("id", "audio", "reference")):
            raise ValueError(f"{path}:{line_number}: id, audio, reference must be strings")
        if not row["id"].strip() or not row["audio"].strip() or row["id"] in ids:
            raise ValueError(f"{path}:{line_number}: empty id/audio or duplicate id")
        ids.add(row["id"])
        samples.append({**row, "audio": str((path.parent / row["audio"]).resolve())})
    if not samples:
        raise ValueError("Manifest contains no samples")
    return samples


def evaluate(samples: list[dict], models: list[STTModel], *, sample_rate: int = 16000,
             normalize_text: bool = True) -> dict:
    if not models or len({model.name for model in models}) != len(models):
        raise ValueError("Supply at least one model with unique names")
    rows = []
    # Load one recording at a time; every adapter receives identical decoded bytes.
    for sample in samples:
        base = {"sample_id": sample["id"], "audio": sample["audio"],
                "reference": sample["reference"]}
        try:
            audio = load_audio(sample["audio"], sample_rate)
        except Exception as exc:
            for model in models:
                rows.append({**base, "model": model.name, "status": "error",
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        base.update(duration_seconds=audio.duration_seconds,
                    pcm_sha256=hashlib.sha256(audio.pcm).hexdigest())
        for model in models:
            try:
                started = time.perf_counter()
                hypothesis = model.transcribe(audio, sample["id"])
                elapsed = time.perf_counter() - started
                if not isinstance(hypothesis, str):
                    raise TypeError("Model must return a transcript string")
                rows.append({**base, "model": model.name, "status": "ok",
                             "hypothesis": hypothesis, "transcription_seconds": elapsed,
                             "real_time_factor": elapsed / audio.duration_seconds,
                             **score(sample["reference"], hypothesis, normalize_text)})
            except Exception as exc:
                rows.append({**base, "model": model.name, "status": "error",
                             "error": f"{type(exc).__name__}: {exc}"})
    summaries = {}
    # Compare summary rates only when models completed the same samples.
    common_ids = set.intersection(*[
        {row["sample_id"] for row in rows if row["model"] == model.name and row["status"] == "ok"}
        for model in models
    ])
    for model in models:
        selected = [row for row in rows if row["model"] == model.name]
        successful = [row for row in selected if row["status"] == "ok"]
        summary = {"successful_samples": len(successful),
                   "failed_samples": len(selected) - len(successful)}
        for prefix, subset in (("successful", successful),
                               ("common", [r for r in successful if r["sample_id"] in common_ids])):
            for metric in ("word", "character"):
                total = Counts()
                for row in subset:
                    total += Counts(**{k: row[metric][k] for k in (
                        "substitutions", "deletions", "insertions", "reference_units")})
                summary[f"{prefix}_{metric}"] = total.to_dict() if subset else None
        summary["transcription_seconds"] = sum(r["transcription_seconds"] for r in successful)
        duration = sum(r["duration_seconds"] for r in successful)
        summary["real_time_factor"] = summary["transcription_seconds"] / duration if duration else None
        summaries[model.name] = summary
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(), "sample_rate": sample_rate,
            "normalization": "nfkc_casefold_punctuation_whitespace" if normalize_text else "none",
            "models": {model.name: model.metadata for model in models},
            "common_sample_ids": sorted(common_ids), "summary": summaries, "results": rows}
