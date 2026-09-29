#!/usr/bin/env python3
"""
LipSync STT evaluation runner.

    python run_eval.py --manifest data/torgo.csv --config config.yaml \
                       --out results/torgo_run1.csv

Then:
    python run_eval.py --report results/torgo_run1.csv

Every row of the output CSV is one (model, utterance) pair with its
hypothesis kept verbatim. Keeping the raw hypothesis is deliberate: when
a number looks wrong you need to see what the engine actually said, and
re-running a Pi benchmark to find out is an hour you do not have.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

from eval import adapters
from eval.datasets import read_manifest, split_holdout
from eval.metrics import word_errors
from eval.report import load_rows, markdown, worst_utterances

FIELDS = ["model", "utt_id", "dataset", "speaker_id", "severity",
          "prompt_type", "reference", "hypothesis", "wer", "errors",
          "substitutions", "deletions", "insertions", "ref_len",
          "seconds", "audio_seconds", "rtf", "peak_rss_mb"]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        import json
        return json.loads(text)


def run(manifest_path: str, config_path: str, out_path: str,
        limit: int | None, holdout: str | None) -> None:
    cfg = load_config(config_path)
    utts = read_manifest(manifest_path)

    if holdout in ("dev", "test"):
        dev, test = split_holdout(utts, cfg.get("test_fraction", 0.3),
                                  cfg.get("split_seed", 1234))
        utts = dev if holdout == "dev" else test
        print(f"[split] {holdout}: {len(utts)} utts, "
              f"{len({u.speaker_id for u in utts})} speakers")

    if limit:
        utts = utts[:limit]
    if not utts:
        sys.exit("no utterances - check the manifest path")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()

        for spec in cfg["models"]:
            label = spec.get("label", spec["engine"])
            print(f"\n=== {label} ===")
            try:
                ad = adapters.build(spec)
                ad.prepare()
            except Exception as e:
                print(f"  SKIPPED - {type(e).__name__}: {e}")
                continue

            t0 = time.time()
            for i, u in enumerate(utts, 1):
                if not os.path.exists(u.audio_path):
                    print(f"  missing audio: {u.audio_path}")
                    continue
                try:
                    res = ad.transcribe(u.audio_path)
                except Exception as e:
                    print(f"  {u.utt_id}: {type(e).__name__}: {e}")
                    continue
                c = word_errors(u.reference, res.text)
                w.writerow({
                    "model": label, "utt_id": u.utt_id, "dataset": u.dataset,
                    "speaker_id": u.speaker_id, "severity": u.severity,
                    "prompt_type": u.prompt_type, "reference": u.reference,
                    "hypothesis": res.text, "wer": f"{c.wer:.4f}",
                    "errors": c.errors, "substitutions": c.substitutions,
                    "deletions": c.deletions, "insertions": c.insertions,
                    "ref_len": c.ref_len, "seconds": f"{res.seconds:.3f}",
                    "audio_seconds": f"{res.audio_seconds:.3f}",
                    "rtf": f"{res.rtf:.3f}",
                    "peak_rss_mb": f"{res.peak_rss_mb:.1f}",
                })
                if i % 25 == 0 or i == len(utts):
                    el = time.time() - t0
                    print(f"  {i}/{len(utts)}  {el:.0f}s elapsed  "
                          f"({el/i:.2f}s/utt)")
                    fh.flush()

    print(f"\nwrote {out_path}")
    print("\nNow run:  python run_eval.py --report " + out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="results/run.csv")
    p.add_argument("--limit", type=int,
                   help="only the first N utterances - use 10 for a smoke test")
    p.add_argument("--holdout", choices=["dev", "test"],
                   help="speaker-disjoint split; develop on dev, report on test")
    p.add_argument("--report", help="summarise an existing results CSV")
    p.add_argument("--worst", help="with --report: show worst utts for MODEL")
    a = p.parse_args()

    if a.report:
        rows = load_rows(a.report)
        print(markdown(rows))
        if a.worst:
            print(f"\n## Worst utterances - {a.worst}\n")
            for r in worst_utterances(rows, a.worst):
                print(f"WER {float(r['wer'])*100:6.1f}%  [{r['speaker_id']}] "
                      f"{r['utt_id']}")
                print(f"   ref: {r['reference']}")
                print(f"   hyp: {r['hypothesis']}\n")
        return

    if not a.manifest:
        p.error("--manifest is required unless --report is given")
    run(a.manifest, a.config, a.out, a.limit, a.holdout)


if __name__ == "__main__":
    main()
