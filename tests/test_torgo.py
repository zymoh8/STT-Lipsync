import tempfile
import unittest
from pathlib import Path
import wave

from stt_bench.runner import load_manifest
from stt_bench.torgo import import_torgo


class TorgoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def recording(self, speaker, session, number, prompt, microphone="headMic"):
        folder = self.root / "TORGO" / speaker / session
        audio = folder / f"wav_{microphone}" / f"{number}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(bytes(320))
        if prompt is not None:
            prompts = folder / "prompts"
            prompts.mkdir(exist_ok=True)
            (prompts / f"{number}.txt").write_text(prompt)
        return audio

    def test_pairing_selection_unique_ids_and_relative_paths(self):
        for speaker, session in (("M01", "Session1"), ("M02", "Session1"), ("M01", "Session2")):
            self.recording(speaker, session, "0001", "Hello world.")
        self.recording("MC01", "Session1", "0001", "Control")
        self.recording("F01", "Session1", "0001", "Female")
        self.recording("M01", "Session1", "0001", "Hello world.", "arrayMic")
        output = self.root / "manifests" / "torgo.jsonl"
        audit = import_torgo(self.root / "TORGO", output)
        samples = load_manifest(output)
        self.assertEqual(audit["included_samples"], 3)
        self.assertEqual(len({s["id"] for s in samples}), 3)
        self.assertTrue(all(Path(s["audio"]).is_file() for s in samples))
        self.assertTrue(all(s["microphone"] == "headMic" for s in samples))
        self.assertTrue(all(s["reference"] == "Hello world." for s in samples))
        with self.assertRaises(ValueError):
            import_torgo(self.root / "TORGO", output)

    def test_exclusions_are_audited_instead_of_scored_as_silence(self):
        for number, prompt in enumerate(("hello", "xxx", "photo.jpg", "", "[noise]", "/ah/", None)):
            self.recording("M01", "Session1", str(number), prompt)
        bad = self.recording("M01", "Session1", "bad", "word")
        bad.write_bytes(b"not a wav")
        output = self.root / "manifest.jsonl"
        audit = import_torgo(self.root, output)
        self.assertEqual(audit["included_samples"], 1)
        self.assertEqual(len(audit["skipped"]), 7)
        self.assertEqual(load_manifest(output)[0]["reference"], "hello")
        self.assertTrue(output.with_suffix(".audit.json").is_file())

    def test_missing_microphone_is_reported_without_fallback(self):
        self.recording("M01", "Session1", "0001", "hello", "arrayMic")
        audit = import_torgo(self.root, self.root / "head.jsonl")
        self.assertEqual(audit["included_samples"], 0)
        self.assertIn("missing wav_headMic", audit["skipped"][0]["reason"])
        audit = import_torgo(self.root, self.root / "array.jsonl", "arrayMic")
        self.assertEqual(audit["included_samples"], 1)


if __name__ == "__main__":
    unittest.main()
