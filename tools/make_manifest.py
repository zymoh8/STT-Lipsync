#!/usr/bin/env python3
"""
Build a manifest CSV from a corpus.

    python tools/make_manifest.py torgo   /path/to/TORGO      data/torgo.csv
    python tools/make_manifest.py ua      /path/to/UASpeech   data/ua.csv \
           --wordlist /path/to/wordlist.csv
    python tools/make_manifest.py folder  ./my_recordings     data/local.csv

ALWAYS verify a new manifest by hand before running a benchmark on it:

    python tools/make_manifest.py --verify data/torgo.csv

Open five rows, play the audio, confirm the reference is what you hear.
A silently wrong manifest produces a results table that looks completely
plausible and means nothing.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.datasets import (build_folder, build_torgo, build_uaspeech,  # noqa
                           read_manifest, write_manifest)


def verify(path: str, n: int = 5) -> None:
    rows = read_manifest(path)
    print(f"{len(rows)} utterances")
    spk = sorted({r.speaker_id for r in rows})
    print(f"{len(spk)} speakers: {', '.join(spk[:12])}"
          f"{' ...' if len(spk) > 12 else ''}")
    from collections import Counter
    print("severity:", dict(Counter(r.severity for r in rows)))
    print("prompt  :", dict(Counter(r.prompt_type for r in rows)))
    missing = [r for r in rows if not os.path.exists(r.audio_path)]
    print(f"missing audio files: {len(missing)}")
    for r in missing[:5]:
        print("   ", r.audio_path)
    unknown = sum(1 for r in rows if r.severity == "unknown")
    if unknown:
        print(f"\n!! {unknown} utterances have severity='unknown'. Fill the "
              f"severity map in eval/datasets.py from the corpus docs, or "
              f"your severity table will be one useless column.")
    print(f"\n--- {n} random rows: play these and check the text ---")
    for r in random.Random(0).sample(rows, min(n, len(rows))):
        print(f"[{r.speaker_id}/{r.severity}] {os.path.basename(r.audio_path)}")
        print(f"    {r.reference!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("kind", nargs="?", choices=["torgo", "ua", "folder"])
    p.add_argument("root", nargs="?")
    p.add_argument("out", nargs="?")
    p.add_argument("--wordlist", help="UA-Speech word id,text CSV")
    p.add_argument("--mic", help="torgo: wav_headMic|wav_arrayMic")
    p.add_argument("--verify", help="verify an existing manifest and exit")
    a = p.parse_args()

    if a.verify:
        verify(a.verify)
        return
    if not (a.kind and a.root and a.out):
        p.error("need: kind root out")

    if a.kind == "torgo":
        rows = build_torgo(a.root, a.mic or "wav_headMic")
    elif a.kind == "ua":
        if not a.wordlist:
            p.error("--wordlist is required for UA-Speech")
        rows = build_uaspeech(a.root, a.wordlist)
    else:
        rows = build_folder(a.root)

    if not rows:
        sys.exit("no utterances found - the directory layout probably differs "
                 "from what the builder expects. Fix build_* in "
                 "eval/datasets.py; that is the only place layout lives.")
    write_manifest(rows, a.out)
    print()
    verify(a.out)


if __name__ == "__main__":
    main()
