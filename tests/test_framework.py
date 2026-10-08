import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

from stt_bench.audio import Audio, load_audio, prepare_audio
from stt_bench.cli import main, make_demo
from stt_bench.metrics import edit_counts, normalize, score
from stt_bench.models import PredictionsModel, VoskModel
from stt_bench.runner import evaluate, load_manifest


def write_wav(path, *, channels=1, rate=16000, width=2):
    pcm = b"\x01\x02" * 16000 * channels
    with wave.open(str(path), "wb") as output:
        output.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm)
    return pcm


class MetricsTests(unittest.TestCase):
    def test_known_edits(self):
        counts = edit_counts("a b c d".split(), "a x c e f".split())
        self.assertEqual(counts.errors, 3)
        self.assertEqual(counts.rate, .75)
        self.assertEqual(edit_counts(["a", "b"], ["a"]).deletions, 1)
        self.assertEqual(edit_counts(["a"], ["b"]).substitutions, 1)
        self.assertEqual(edit_counts(["a"], ["a", "b"]).insertions, 1)

    def test_empty_reference_and_unbounded_error(self):
        self.assertEqual(score("", "")["word"]["rate"], 0)
        self.assertEqual(score("", "false words")["word"]["rate"], 2)
        self.assertEqual(score("a", "a b c")["word"]["rate"], 2)
        self.assertEqual(score("two words", "")["word"]["rate"], 1)

    def test_normalization_and_characters(self):
        self.assertEqual(normalize("  HELLO,\tworld! Don’t. "), "hello world don't")
        self.assertEqual(score("Hello, WORLD!", "hello world")["word"]["rate"], 0)
        self.assertGreater(score("Hello", "hello", False)["word"]["rate"], 0)
        self.assertEqual(score("cat", "cut")["character"]["rate"], 1 / 3)
        self.assertEqual(score("a b", "ab")["character"]["reference_units"], 3)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_pcm_excludes_header_and_chunks_reassemble(self):
        path = self.directory / "test.wav"
        pcm = write_wav(path)
        audio = load_audio(path)
        self.assertEqual(audio.pcm, pcm)
        self.assertEqual(b"".join(audio.chunks(333)), pcm)
        self.assertEqual(audio.duration_seconds, 1)
        with self.assertRaises(ValueError):
            list(audio.chunks(0))

    def test_reject_stereo_wrong_rate_empty_and_truncated(self):
        for options in ({"channels": 2}, {"rate": 8000}, {"width": 1}):
            path = self.directory / "bad.wav"
            write_wav(path, **options)
            with self.assertRaises(ValueError):
                load_audio(path)
        path = self.directory / "bad.wav"
        write_wav(path)
        path.write_bytes(path.read_bytes()[:-10])
        with self.assertRaises(ValueError):
            load_audio(path)
        with wave.open(str(path), "wb") as out:
            out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        with self.assertRaises(ValueError):
            load_audio(path)

    def test_manifest_validation_and_relative_paths(self):
        make_demo(self.directory)
        path = self.directory / "manifest.jsonl"
        self.assertEqual(load_manifest(path)[0]["audio"], str(self.directory / "silence.wav"))
        path.write_text(path.read_text() * 2)
        with self.assertRaises(ValueError):
            load_manifest(path)
        for content in ('[]\n', '{"id":"x"}\n', 'bad\n', ''):
            path.write_text(content)
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_weighted_rates_failures_and_common_samples(self):
        path = self.directory / "audio.wav"
        write_wav(path)
        samples = [{"id": "short", "audio": str(path), "reference": "a"},
                   {"id": "long", "audio": str(path), "reference": "a b c"}]
        a = self.directory / "a.json"
        b = self.directory / "b.json"
        a.write_text(json.dumps({"short": "wrong", "long": "a b c"}))
        b.write_text(json.dumps({"long": "a b c"}))
        report = evaluate(samples, [PredictionsModel("a", a), PredictionsModel("b", b)])
        self.assertEqual(report["summary"]["a"]["successful_word"]["rate"], .25)
        self.assertEqual(report["summary"]["b"]["failed_samples"], 1)
        self.assertEqual(report["common_sample_ids"], ["long"])
        self.assertEqual(report["summary"]["a"]["common_word"]["rate"], 0)
        samples[0]["audio"] = str(self.directory / "missing.wav")
        report = evaluate(samples[:1], [PredictionsModel("a", a)])
        self.assertEqual(report["results"][0]["status"], "error")
        self.assertIsNone(report["summary"]["a"]["successful_word"])

    def test_vosk_collects_segments_and_final_and_resets_recognizer(self):
        class Recognizer:
            instances = 0

            def __init__(self, model, rate):
                Recognizer.instances += 1
                self.chunks = []
                self.rate = rate

            def AcceptWaveform(self, data):
                self.chunks.append(data)
                return len(self.chunks) == 1

            def Result(self):
                return '{"text": "first segment"}'

            def FinalResult(self):
                return '{"text": "last segment"}'

        adapter = VoskModel.__new__(VoskModel)
        adapter._model = object()
        adapter._recognizer = Recognizer
        audio = Audio(b"\x00\x00" * 5000, 16000)
        for _ in range(2):
            self.assertEqual(adapter.transcribe(audio, "x"), "first segment last segment")
        self.assertEqual(Recognizer.instances, 2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_prepare_downmix_and_resample(self):
        source = self.directory / "stereo.wav"
        write_wav(source, channels=2, rate=8000)
        destination = self.directory / "mono.wav"
        prepare_audio(source, destination)
        self.assertEqual(load_audio(destination).duration_seconds, 2)
        with self.assertRaises(ValueError):
            prepare_audio(source, destination)

    def test_cli_demo_report_and_failure_exit(self):
        with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(main(["demo", "--directory", str(self.directory)]), 0)
            arguments = ["run", str(self.directory / "manifest.jsonl"), "--predictions",
                         f"saved={self.directory / 'predictions.json'}", "--output",
                         str(self.directory / "results")]
            self.assertEqual(main(arguments), 0)
            self.assertEqual(main(arguments), 1)  # Don't overwrite previous experiments.
            report = json.loads((self.directory / "results/report.json").read_text())
            self.assertEqual(report["results"][0]["word"]["rate"], 0)
            self.assertTrue((self.directory / "results/samples.csv").is_file())
            (self.directory / "predictions.json").write_text('{}')
            arguments[-1] = str(self.directory / "failed")
            self.assertEqual(main(arguments), 1)
            self.assertEqual(main(["run", str(self.directory / "manifest.jsonl")]), 1)


if __name__ == "__main__":
    unittest.main()
