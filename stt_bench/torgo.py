"""Build a reproducible manifest from an extracted TORGO archive."""

import json
import os
from pathlib import Path
import re
import wave

from .audio import load_audio


def import_torgo(root: Path, output: Path, microphone: str = "headMic") -> dict:
    """Keep one microphone; audit exclusions rather than treating them as silence.

    Prompt text is a supplied stimulus, not a verified verbatim transcript.
    The filters below cannot identify every non-word task; review the manifest.
    """
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Extracted TORGO directory does not exist: {root}")
    if microphone not in ("headMic", "arrayMic"):
        raise ValueError("Choose headMic or arrayMic")
    output = output.resolve()
    audit_path = output.with_suffix(".audit.json")
    if output.exists() or audit_path.exists():
        raise ValueError("Manifest or audit already exists; choose a new --output path")
    samples, skipped = [], []
    speakers = set()
    sessions = 0
    for prompts in sorted(root.rglob("prompts")):
        if not prompts.is_dir():
            continue
        session = prompts.parent
        speaker = session.parent.name
        if not re.fullmatch(r"M0[1-5]", speaker):
            continue
        sessions += 1
        audio_dir = session / f"wav_{microphone}"
        if not audio_dir.is_dir():
            skipped.append({"path": str(session), "reason": f"missing wav_{microphone}"})
            continue
        for audio in sorted(audio_dir.iterdir()):
            if not audio.is_file() or audio.suffix.lower() != ".wav":
                continue
            prompt = prompts / f"{audio.stem}.txt"
            reason = None
            reference = ""
            if not prompt.is_file():
                reason = "missing prompt"
            else:
                try:
                    reference = " ".join(prompt.read_text(encoding="utf-8-sig").split())
                except UnicodeError:
                    reason = "prompt is not UTF-8; review encoding"
                if reason is None:
                    if not reference:
                        reason = "empty prompt; not assumed to be silence"
                    elif re.search(r"\bxxx\b", reference, re.IGNORECASE):
                        reason = "prompt marked xxx (unusable)"
                    elif re.search(r"\.(?:jpg|jpeg|png)\b", reference, re.IGNORECASE):
                        reason = "image prompt requires a verbatim transcript"
                    elif re.search(r"[\[\]<>/]", reference):
                        reason = "annotated or phonetic prompt requires review"
            if reason is None:
                try:
                    load_audio(audio)
                except (OSError, ValueError, EOFError, wave.Error) as exc:
                    reason = f"invalid benchmark audio: {exc}"
            if reason:
                skipped.append({"path": str(audio), "prompt": str(prompt),
                                "reference": reference, "reason": reason})
                continue
            samples.append({
                "id": f"{speaker}/{session.name}/{microphone}/{audio.stem}",
                "audio": os.path.relpath(audio, output.parent), "reference": reference,
                "speaker": speaker, "session": session.name, "microphone": microphone,
                "reference_source": str(prompt), "reference_type": "supplied_prompt",
            })
            speakers.add(speaker)
    if not sessions:
        raise ValueError("No M01–M05 speaker/session/prompts folders found; extract M.tar.bz2 first")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Duplicate speaker/session/utterance IDs; select a single extracted dataset root")
    audit = {"dataset": "TORGO", "root": str(root), "microphone": microphone,
             "speakers": sorted(speakers), "sessions_found": sessions,
             "included_samples": len(samples), "skipped": skipped,
             "reference_note": "Supplied prompts need review against the actual speech; "
                               "lexical non-word tasks may remain. These are not silence tests."}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Even an empty selection has an audit explaining why; the runner rejects empty manifests.
    with output.open("x", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with audit_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return audit
