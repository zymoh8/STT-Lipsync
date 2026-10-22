import unittest

from stt_bench.metrics import score
from stt_bench.reporting import analyze, metrics, percentile


def row(model, sample_id, speaker, reference, hypothesis):
    return {"model": model, "sample_id": sample_id, "speaker": speaker,
            "session": "Session1", "status": "ok", "reference": reference,
            "hypothesis": hypothesis, "pcm_sha256": sample_id, "duration_seconds": 2,
            "transcription_seconds": 1, "real_time_factor": .5, **score(reference, hypothesis)}


class ReportingTests(unittest.TestCase):
    def test_corpus_and_equal_speaker_averages_differ(self):
        rows = [row("a", "short", "M01", "a", "wrong"),
                row("a", "long", "M02", "a b c", "a b c")]
        samples = [{"id": "short", "speaker": "M01"}, {"id": "long", "speaker": "M02"}]
        result = analyze({"models": {"a": {}}, "results": rows}, samples)["models"]["a"]
        self.assertEqual(result["common"]["wer"], .25)
        self.assertEqual(result["macro_speaker_wer"], .5)
        self.assertEqual(result["common"]["exact_match_rate"], .5)
        self.assertEqual(result["common"]["real_time_factor"], .5)

    def test_common_subset_coverage_and_pcm_integrity(self):
        rows = [row("a", "x", "M01", "a", "a"), row("a", "y", "M01", "a", "wrong"),
                row("b", "x", "M01", "a", "a"),
                {"model": "b", "sample_id": "y", "status": "error", "error": "failure"}]
        samples = [{"id": "x", "speaker": "M01"}, {"id": "y", "speaker": "M01"}]
        report = {"models": {"a": {}, "b": {}}, "results": rows}
        result = analyze(report, samples)
        self.assertEqual(result["common_samples"], 1)
        self.assertEqual(result["models"]["a"]["common"]["wer"], 0)
        self.assertEqual(result["models"]["b"]["coverage"], .5)
        rows[2]["pcm_sha256"] = "changed"
        with self.assertRaises(ValueError):
            analyze(report, samples)

    def test_percentiles_and_unbounded_wer(self):
        self.assertEqual(percentile([1, 3], .5), 2)
        self.assertEqual(percentile([1], .95), 1)
        self.assertIsNone(percentile([], .5))
        result = metrics([row("a", "x", "M01", "a", "a b c")])
        self.assertEqual(result["wer"], 2)
        self.assertEqual(result["word"]["insertions"], 2)

    def test_duration_sensitivity_does_not_remove_primary_samples(self):
        rows = [row("a", "normal", "M01", "hello", "hello"),
                row("a", "long", "M01", "hello", "extra words"),
                row("a", "short", "M01", "hello", "")]
        rows[1]["duration_seconds"] = 180
        rows[2]["duration_seconds"] = .009
        samples = [{"id": r["sample_id"], "speaker": "M01"} for r in rows]
        result = analyze({"models": {"a": {}}, "results": rows}, samples)
        self.assertEqual(result["common_samples"], 3)
        self.assertEqual(result["duration_outlier_ids"], ["long"])
        self.assertEqual(result["short_audio_ids"], ["short"])
        self.assertEqual(result["models"]["a"]["without_duration_outliers"]["samples"], 2)
        self.assertEqual(result["models"]["a"]["without_duration_flags"]["samples"], 1)
