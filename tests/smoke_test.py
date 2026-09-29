#!/usr/bin/env python3
"""
End-to-end smoke test. No models, no corpus, no dependencies.

    python tests/smoke_test.py

Generates silent wavs with sidecar transcripts, runs the dummy engines,
and asserts the whole chain behaves. Run this first on any new machine -
if it fails, the harness is broken and no benchmark number from that
machine means anything.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from eval.metrics import wer, word_errors, corpus_wer  # noqa: E402
from eval.normalize import normalize  # noqa: E402

SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "please call the nurse",
    "i would like some water",
    "turn the lights off",
    "seven",
    "my name is alex and i live in boston",
    "can you open the window",
    "it is twenty five degrees outside",
]


def make_wav(path: str, seconds: float = 1.0, rate: int = 16000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % int(rate * seconds),
                                  *([0] * int(rate * seconds))))


def check(label: str, cond: bool) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        raise AssertionError(label)


def main() -> None:
    print("1. normalisation")
    check("case and punctuation stripped",
          normalize("Hello, World!") == "hello world")
    check("digits expanded",
          normalize("i have 25 apples") == "i have twenty five apples")
    check("hyphen becomes a space",
          normalize("twenty-five") == "twenty five")
    check("contraction expanded",
          normalize("don't") == "do not")
    check("bracketed annotation dropped",
          normalize("hello [noise] world") == "hello world")
    check("vosk-style and whisper-style agree",
          normalize("i have 7 cats") == normalize("I have seven cats."))
    check("filler removed", normalize("um hello") == "hello")

    print("\n2. metrics")
    check("identical -> 0.0", wer("a b c", "a b c") == 0.0)
    check("one substitution of three -> 1/3",
          abs(wer("a b c", "a x c") - 1 / 3) < 1e-9)
    check("one deletion of three -> 1/3",
          abs(wer("a b c", "a c") - 1 / 3) < 1e-9)
    check("one insertion of three -> 1/3",
          abs(wer("a b c", "a b x c") - 1 / 3) < 1e-9)
    check("WER may exceed 1.0",
          wer("hello", "one two three four five") > 1.0)
    c = word_errors("the cat sat", "the dog sat")
    check("S/D/I breakdown", (c.substitutions, c.deletions, c.insertions)
          == (1, 0, 0))
    check("pooled != mean of per-utterance",
          abs(corpus_wer([word_errors("a b c d e f g h i j", "a b c d e f g h i x"),
                          word_errors("yes", "no")]) - 2 / 11) < 1e-9)

    print("\n3. end to end")
    tmp = tempfile.mkdtemp(prefix="stt_smoke_")
    try:
        audio = os.path.join(tmp, "audio")
        os.makedirs(audio)
        for i, s in enumerate(SENTENCES):
            make_wav(os.path.join(audio, f"u{i:02d}.wav"))
            with open(os.path.join(audio, f"u{i:02d}.txt"), "w") as fh:
                fh.write(s)

        manifest = os.path.join(tmp, "manifest.csv")
        subprocess.run([sys.executable, "tools/make_manifest.py", "folder",
                        audio, manifest], cwd=ROOT, check=True,
                       capture_output=True)
        check("manifest built", os.path.exists(manifest))

        cfg = os.path.join(tmp, "cfg.json")
        with open(cfg, "w") as fh:
            fh.write('{"models":['
                     '{"engine":"dummy","label":"perfect","error_rate":0.0},'
                     '{"engine":"dummy","label":"perfect-rich",'
                     '"error_rate":0.0,"style":"rich"},'
                     '{"engine":"dummy","label":"sloppy","error_rate":0.35}'
                     ']}')
        out = os.path.join(tmp, "res.csv")
        r = subprocess.run([sys.executable, "run_eval.py", "--manifest",
                            manifest, "--config", cfg, "--out", out],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout, r.stderr)
        check("runner exited cleanly", r.returncode == 0)

        from eval.report import by_model, load_rows, markdown
        rows = load_rows(out)
        stats = by_model(rows)
        check("three models scored", len(stats) == 3)
        check("perfect engine -> 0% WER", stats["perfect"]["wer"] == 0.0)
        check("rich formatting also 0% (normalisation works)",
              stats["perfect-rich"]["wer"] == 0.0)
        check("sloppy engine is worse", stats["sloppy"]["wer"] > 0.1)
        md = markdown(rows)
        check("report renders", "## By dysarthria severity" in md)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL PASSED - harness is sound on this machine.")


if __name__ == "__main__":
    main()
