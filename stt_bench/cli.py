import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import wave

from .audio import load_audio, prepare_audio
from .models import PredictionsModel, VoskModel
from .runner import evaluate, load_manifest


def model_spec(value: str):
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    return name, Path(path)


def make_demo(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    files = [directory / name for name in ("silence.wav", "manifest.jsonl", "predictions.json")]
    if any(path.exists() for path in files):
        raise ValueError("Demo files already exist; choose a new directory")
    with wave.open(str(files[0]), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00" * 16000)
    files[1].write_text(json.dumps({"id": "silence", "audio": "silence.wav", "reference": ""}) + "\n",
                        encoding="utf-8")
    files[2].write_text(json.dumps({"silence": ""}) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate STT transcripts against labeled audio")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate WAV and inspect PCM bytes")
    inspect.add_argument("audio", type=Path)
    inspect.add_argument("--sample-rate", type=int, default=16000)
    prepare = commands.add_parser("prepare", help="Convert audio with FFmpeg to PCM WAV")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("destination", type=Path)
    prepare.add_argument("--sample-rate", type=int, default=16000)
    demo = commands.add_parser("demo", help="Generate one second of silence and saved predictions")
    demo.add_argument("--directory", type=Path, default=Path("demo"))
    run = commands.add_parser("run", help="Run models and write JSON/CSV reports")
    run.add_argument("manifest", type=Path)
    run.add_argument("--vosk", type=model_spec, action="append", default=[], metavar="NAME=MODEL_DIR")
    run.add_argument("--predictions", type=model_spec, action="append", default=[], metavar="NAME=JSON")
    run.add_argument("--sample-rate", type=int, default=16000)
    run.add_argument("--raw-text", action="store_true", help="Disable text normalization")
    run.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        if getattr(args, "sample_rate", 16000) <= 0:
            raise ValueError("Sample rate must be positive")
        if args.command == "demo":
            make_demo(args.directory)
            print(f"Created silence demo in {args.directory} (pipeline check, not an accuracy benchmark)")
        elif args.command == "inspect":
            audio = load_audio(args.audio, args.sample_rate)
            print(json.dumps({"sample_rate": audio.sample_rate, "channels": audio.channels,
                              "sample_width_bytes": audio.sample_width, "pcm_bytes": len(audio.pcm),
                              "duration_seconds": audio.duration_seconds}, indent=2))
        elif args.command == "prepare":
            prepare_audio(args.source, args.destination, args.sample_rate)
            print(f"Prepared {args.destination}")
        else:
            samples = load_manifest(args.manifest)
            specs = args.vosk + args.predictions
            if not specs or len({name for name, _ in specs}) != len(specs):
                raise ValueError("Supply --vosk and/or --predictions with unique model names")
            json_path = args.output / "report.json"
            csv_path = args.output / "samples.csv"
            if json_path.exists() or csv_path.exists():
                raise ValueError("Report already exists; choose a new --output directory")
            models = [VoskModel(name, path) for name, path in args.vosk]
            models += [PredictionsModel(name, path) for name, path in args.predictions]
            report = evaluate(samples, models, sample_rate=args.sample_rate,
                              normalize_text=not args.raw_text)
            report["manifest"] = str(args.manifest.resolve())
            args.output.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            fields = ["sample_id", "model", "status", "reference", "hypothesis", "wer", "cer",
                      "duration_seconds", "transcription_seconds", "real_time_factor", "error"]
            with csv_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for row in report["results"]:
                    flat = {key: row.get(key, "") for key in fields}
                    flat.update(wer=row.get("word", {}).get("rate", ""),
                                cer=row.get("character", {}).get("rate", ""))
                    writer.writerow(flat)
            for name, summary in report["summary"].items():
                word = summary["common_word"]
                wer = f"{word['rate']:.2%}" if word else "n/a"
                print(f"{name}: common-sample WER {wer}; "
                      f"{summary['successful_samples']} successful, {summary['failed_samples']} failed")
            print(f"Reports: {json_path} and {csv_path}")
            if any(s["failed_samples"] for s in report["summary"].values()):
                return 1
    except (OSError, ValueError, RuntimeError, wave.Error, EOFError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
