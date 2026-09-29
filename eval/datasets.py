"""
Dataset loading.

Design note: everything downstream consumes a MANIFEST - a CSV with one
row per utterance. Corpus-specific quirks live only in the builder
functions here. If a builder guesses your directory layout wrong, you fix
one function and regenerate the manifest; nothing else in the harness
changes.

Manifest columns:
    utt_id      unique string
    audio_path  absolute or manifest-relative path to a .wav
    reference   ground truth transcript
    speaker_id  e.g. "M01"
    severity    one of: control, mild, moderate, severe, unknown
    dataset     e.g. "torgo"
    prompt_type "word" | "sentence" | "unknown"

IMPORTANT: the TORGO and UA-Speech builders below are written from the
published descriptions of those corpora. Neither was run against the real
data when written - verify the first manifest by hand before trusting a
single number. Open five rows, play the audio, check the reference
matches what you hear.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, asdict, fields

SEVERITIES = ("control", "mild", "moderate", "severe", "unknown")


@dataclass
class Utterance:
    utt_id: str
    audio_path: str
    reference: str
    speaker_id: str = ""
    severity: str = "unknown"
    dataset: str = ""
    prompt_type: str = "unknown"


def write_manifest(rows: list[Utterance], path: str) -> None:
    cols = [f.name for f in fields(Utterance)]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"wrote {len(rows)} utterances -> {path}")


def read_manifest(path: str) -> list[Utterance]:
    base = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ap = row["audio_path"]
            if not os.path.isabs(ap):
                ap = os.path.normpath(os.path.join(base, ap))
            row["audio_path"] = ap
            out.append(Utterance(**{k: row.get(k, "") for k in
                                    [f.name for f in fields(Utterance)]}))
    return out


# --------------------------------------------------------------------------
# TORGO
# --------------------------------------------------------------------------
# Published layout (verify against your copy):
#   <root>/<SPK>/Session<N>/wav_headMic/0001.wav
#                           wav_arrayMic/0001.wav
#                           prompts/0001.txt
# Speaker IDs: F01 F03 F04 M01..M05 (dysarthric), FC01.. MC01.. (control)
#
# Severity is NOT encoded in the directory tree. It comes from the paper.
# The mapping below is a PLACEHOLDER - fill it from the TORGO
# documentation that ships with your licensed copy, then delete this note.
TORGO_SEVERITY: dict[str, str] = {
    # "M01": "severe",
    # "M02": "severe",
    # "F01": "severe",
    # ... fill from corpus docs
}

_NON_PROMPT = re.compile(r"^\s*(xxx|\[.*\]|)\s*$", re.I)


def build_torgo(root: str, mic: str = "wav_headMic") -> list[Utterance]:
    """Walk a TORGO tree and emit utterances.

    mic: "wav_headMic" (close-talking, closer to your deployment) or
         "wav_arrayMic". Pick one and stay with it - mixing them makes
         results incomparable. headMic is the right default for a project
         that will mount a mic near the mouth.
    """
    rows: list[Utterance] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    for spk in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir):
            continue
        is_control = "C" in spk.upper()[:3] and spk.upper()[1] == "C"
        sev = "control" if is_control else TORGO_SEVERITY.get(spk, "unknown")
        for sess in sorted(os.listdir(spk_dir)):
            sess_dir = os.path.join(spk_dir, sess)
            wav_dir = os.path.join(sess_dir, mic)
            pr_dir = os.path.join(sess_dir, "prompts")
            if not (os.path.isdir(wav_dir) and os.path.isdir(pr_dir)):
                continue
            for wav in sorted(os.listdir(wav_dir)):
                if not wav.lower().endswith(".wav"):
                    continue
                stem = os.path.splitext(wav)[0]
                prompt_file = os.path.join(pr_dir, stem + ".txt")
                if not os.path.isfile(prompt_file):
                    continue
                with open(prompt_file, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read().strip()
                # TORGO prompts include non-speech instructions such as
                # image-description tasks; those have no usable reference.
                if _NON_PROMPT.match(text) or text.startswith("/"):
                    continue
                rows.append(Utterance(
                    utt_id=f"torgo_{spk}_{sess}_{stem}",
                    audio_path=os.path.join(wav_dir, wav),
                    reference=text,
                    speaker_id=spk,
                    severity=sev,
                    dataset="torgo",
                    prompt_type="word" if len(text.split()) == 1 else "sentence",
                ))
    return rows


# --------------------------------------------------------------------------
# UA-Speech
# --------------------------------------------------------------------------
# Filenames encode everything: <SPK>_<BLOCK>_<WORDID>_<MIC>.wav
# e.g. M04_B1_UW10_M3.wav
# Speakers prefixed C (CM/CF) are controls.
# Word IDs map to text via the corpus word list - supply it as a
# two-column CSV: word_id,text
UASPEECH_SEVERITY: dict[str, str] = {
    # fill from corpus docs, e.g. "M04": "severe", "M05": "moderate"
}

_UA_NAME = re.compile(r"^(?P<spk>[A-Z]{1,2}\d{2})_(?P<block>B\d)_"
                      r"(?P<word>[A-Z0-9]+)_(?P<mic>M\d)$", re.I)


def build_uaspeech(root: str, wordlist_csv: str,
                   mic: str | None = None) -> list[Utterance]:
    words: dict[str, str] = {}
    with open(wordlist_csv, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0].strip():
                words[row[0].strip().upper()] = row[1].strip()

    rows: list[Utterance] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            m = _UA_NAME.match(os.path.splitext(fn)[0])
            if not m:
                continue
            if mic and m.group("mic").upper() != mic.upper():
                continue
            spk = m.group("spk").upper()
            text = words.get(m.group("word").upper())
            if not text:
                continue
            sev = "control" if spk.startswith("C") else \
                UASPEECH_SEVERITY.get(spk, "unknown")
            rows.append(Utterance(
                utt_id=f"ua_{os.path.splitext(fn)[0]}",
                audio_path=os.path.join(dirpath, fn),
                reference=text,
                speaker_id=spk,
                severity=sev,
                dataset="uaspeech",
                prompt_type="word",
            ))
    return rows


# --------------------------------------------------------------------------
# Generic: your own recordings
# --------------------------------------------------------------------------
def build_folder(root: str, dataset: str = "local",
                 severity: str = "control") -> list[Utterance]:
    """Every <name>.wav paired with <name>.txt in the same folder."""
    rows = []
    for dirpath, _d, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            stem = os.path.splitext(fn)[0]
            txt = os.path.join(dirpath, stem + ".txt")
            if not os.path.isfile(txt):
                continue
            with open(txt, encoding="utf-8", errors="ignore") as fh:
                text = fh.read().strip()
            if not text:
                continue
            rows.append(Utterance(
                utt_id=f"{dataset}_{stem}",
                audio_path=os.path.join(dirpath, fn),
                reference=text,
                speaker_id=os.path.basename(dirpath),
                severity=severity,
                dataset=dataset,
                prompt_type="word" if len(text.split()) == 1 else "sentence",
            ))
    return rows


def split_holdout(rows: list[Utterance], test_fraction: float = 0.3,
                  seed: int = 1234) -> tuple[list[Utterance], list[Utterance]]:
    """Split BY SPEAKER, not by utterance.

    Splitting by utterance leaks: the same speaker appears in dev and
    test, so anything speaker-adaptive scores far better than it will in
    the field. Every deployment sees a speaker it has never heard.
    """
    import random
    spks = sorted({r.speaker_id for r in rows})
    rng = random.Random(seed)
    rng.shuffle(spks)
    n_test = max(1, round(len(spks) * test_fraction))
    test_spk = set(spks[:n_test])
    test = [r for r in rows if r.speaker_id in test_spk]
    dev = [r for r in rows if r.speaker_id not in test_spk]
    return dev, test
