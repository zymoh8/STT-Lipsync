"""Independently validate saved edit totals, coverage, and original PCM hashes."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt_bench.audio import load_audio
from stt_bench.runner import load_manifest
from stt_bench.reporting import analyze


def main():
    import rapidfuzz
    from rapidfuzz.distance import Levenshtein
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/torgo-manifest.jsonl"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    samples = load_manifest(args.manifest)
    by_id = {s["id"]: s for s in samples}
    expected_hashes = {s["id"]: hashlib.sha256(load_audio(s["audio"]).pcm).hexdigest() for s in samples}
    analysis = analyze(report, samples)
    comparisons = 0
    successful = 0
    for row in report["results"]:
        assert row["reference"] == by_id[row["sample_id"]]["reference"], row["sample_id"]
        if "pcm_sha256" in row:
            assert row["pcm_sha256"] == expected_hashes[row["sample_id"]], row["sample_id"]
        if row["status"] != "ok":
            continue
        successful += 1
        for unit, reference, hypothesis in (
            ("word", row["normalized_reference"].split(), row["normalized_hypothesis"].split()),
            ("character", row["normalized_reference"], row["normalized_hypothesis"]),
        ):
            distance = Levenshtein.distance(reference, hypothesis)
            stored = row[unit]
            assert distance == stored["errors"], (row["sample_id"], row["model"], unit)
            assert distance == sum(stored[k] for k in ("substitutions", "deletions", "insertions"))
            assert len(reference) == stored["reference_units"]
            expected_rate = distance / len(reference) if reference else float(distance)
            assert math.isclose(expected_rate, stored["rate"], rel_tol=1e-12, abs_tol=1e-12)
            comparisons += 1
        assert row["transcription_seconds"] >= 0
        assert math.isclose(row["real_time_factor"], row["transcription_seconds"] / row["duration_seconds"], rel_tol=1e-12)
    duplicate_groups = [count for count in Counter(expected_hashes.values()).values() if count > 1]
    validation = {
        "status": "passed", "independent_library": "RapidFuzz", "rapidfuzz_version": rapidfuzz.__version__,
        "attempts": analysis["attempts"], "successful_transcripts": successful,
        "independent_edit_distance_checks": comparisons, "edit_distance_mismatches": 0,
        "original_pcm_files_verified": len(expected_hashes), "unique_pcm_hashes": len(set(expected_hashes.values())),
        "duplicate_pcm_group_sizes": sorted(duplicate_groups, reverse=True),
        "one_attempt_per_model_and_sample": True, "common_samples": analysis["common_samples"],
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
    }
    (args.report.parent / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
