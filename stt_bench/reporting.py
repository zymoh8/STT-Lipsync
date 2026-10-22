"""Aggregate saved benchmark rows without rerunning inference."""

from collections import Counter
import statistics
import unicodedata


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * quantile
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def metrics(rows):
    """Compute corpus and utterance scores on an explicit successful-row subset."""
    if not rows:
        return {"samples": 0, "wer": None, "cer": None, "exact_match_rate": None}
    output = {"samples": len(rows)}
    for unit, label in (("word", "wer"), ("character", "cer")):
        counts = {key: sum(r[unit][key] for r in rows)
                  for key in ("substitutions", "deletions", "insertions", "reference_units")}
        errors = sum(counts[k] for k in ("substitutions", "deletions", "insertions"))
        output[label] = errors / counts["reference_units"] if counts["reference_units"] else errors
        output[unit] = {**counts, "errors": errors}
        output[f"mean_utterance_{label}"] = statistics.mean(r[unit]["rate"] for r in rows)
        output[f"median_utterance_{label}"] = statistics.median(r[unit]["rate"] for r in rows)
    output["exact_matches"] = sum(r["normalized_reference"] == r["normalized_hypothesis"] for r in rows)
    output["exact_match_rate"] = output["exact_matches"] / len(rows)
    output["empty_predictions"] = sum(not r["normalized_hypothesis"] for r in rows)
    output["expansion_flags"] = sum(
        len(r["normalized_hypothesis"].split()) >= max(10, 3 * len(r["normalized_reference"].split()))
        for r in rows)
    output["non_latin_flags"] = sum(any(
        char.isalpha() and "LATIN" not in unicodedata.name(char, "")
        for char in r["hypothesis"]) for r in rows)
    output["duration_seconds"] = sum(r["duration_seconds"] for r in rows)
    output["transcription_seconds"] = sum(r["transcription_seconds"] for r in rows)
    output["real_time_factor"] = output["transcription_seconds"] / output["duration_seconds"]
    for key in ("transcription_seconds", "real_time_factor"):
        values = [r[key] for r in rows]
        for quantile, label in ((0.5, "median"), (0.9, "p90"), (0.95, "p95")):
            output[f"{label}_{key}"] = percentile(values, quantile)
    return output


def analyze(report, samples):
    names = list(report["models"])
    ids = {s["id"] for s in samples}
    rows = report["results"]
    if len(ids) != len(samples):
        raise ValueError("Duplicate manifest IDs")
    seen = {(r["model"], r["sample_id"]) for r in rows}
    expected = {(name, sample_id) for name in names for sample_id in ids}
    if len(seen) != len(rows) or seen != expected:
        raise ValueError("Report does not contain exactly one attempt per model/sample")
    common = set.intersection(*[
        {r["sample_id"] for r in rows if r["model"] == name and r["status"] == "ok"}
        for name in names])
    pcm_hashes = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        previous = pcm_hashes.setdefault(row["sample_id"], row["pcm_sha256"])
        if previous != row["pcm_sha256"]:
            raise ValueError(f"Models received different PCM: {row['sample_id']}")
    speakers = sorted({sample.get("speaker", "unknown") for sample in samples})
    long_ids = sorted({r["sample_id"] for r in rows if r["status"] == "ok" and r["duration_seconds"] > 30})
    short_ids = sorted({r["sample_id"] for r in rows if "duration_seconds" in r and r["duration_seconds"] < .1})
    output = {"expected_models": len(names), "manifest_samples": len(samples),
              "attempts": len(rows), "common_samples": len(common),
              "speakers": speakers, "samples_by_speaker": dict(Counter(s.get("speaker", "unknown") for s in samples)),
              "duration_outlier_ids": long_ids,
              "short_audio_ids": short_ids,
              "models": {}}
    for name in names:
        attempted = [r for r in rows if r["model"] == name]
        successful = [r for r in attempted if r["status"] == "ok"]
        comparable = [r for r in successful if r["sample_id"] in common]
        by_speaker = {speaker: metrics([r for r in comparable if r.get("speaker", "unknown") == speaker])
                      for speaker in speakers}
        represented = [m for m in by_speaker.values() if m["samples"]]
        summary = {
            "attempted": len(attempted), "successful": len(successful),
            "failed": len(attempted) - len(successful),
            "coverage": len(successful) / len(attempted),
            "common": metrics(comparable), "all_successful": metrics(successful),
            "per_speaker": by_speaker,
            "macro_speaker_wer": statistics.mean(m["wer"] for m in represented) if represented else None,
            "macro_speaker_cer": statistics.mean(m["cer"] for m in represented) if represented else None,
            "macro_speaker_exact_match": statistics.mean(m["exact_match_rate"] for m in represented) if represented else None,
            "single_word": metrics([r for r in comparable if len(r["normalized_reference"].split()) == 1]),
            "multi_word": metrics([r for r in comparable if len(r["normalized_reference"].split()) > 1]),
            "without_duration_outliers": metrics([r for r in comparable if r["sample_id"] not in long_ids]),
            "without_duration_flags": metrics([r for r in comparable if r["sample_id"] not in set(long_ids + short_ids)]),
            "per_session": {key: metrics([r for r in comparable if f"{r.get('speaker')}/{r.get('session')}" == key])
                            for key in sorted({f"{r.get('speaker')}/{r.get('session')}" for r in comparable})},
            "errors": [{"sample_id": r["sample_id"], "error": r.get("error", "unknown")}
                       for r in attempted if r["status"] != "ok"],
        }
        output["models"][name] = summary
    output["ranking_macro_speaker_wer"] = sorted(names, key=lambda n: (
        output["models"][n]["macro_speaker_wer"] is None,
        output["models"][n]["macro_speaker_wer"] or 0))
    output["ranking_corpus_wer"] = sorted(names, key=lambda n: (
        output["models"][n]["common"]["wer"] is None,
        output["models"][n]["common"]["wer"] or 0))
    return output
