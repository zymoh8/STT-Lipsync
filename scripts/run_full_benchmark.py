"""Run the downloaded six-model suite sequentially with resumable JSONL checkpoints."""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_bench.models import VoskModel, WhisperModel, Wav2Vec2Model, ParakeetModel
from stt_bench.runner import evaluate, load_manifest


SPECS = [
    ("vosk-small", VoskModel, "vosk-model-small-en-us-0.15"),
    ("vosk-lgraph", VoskModel, "vosk-model-en-us-0.22-lgraph"),
    ("vosk-large", VoskModel, "vosk-model-en-us-0.22"),
    ("wav2vec2", Wav2Vec2Model, "wav2vec2-base-960h"),
    ("parakeet", ParakeetModel, "parakeet-tdt-0.6b-v3"),
    ("whisper", WhisperModel, "whisper-small"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--only", choices=[spec[0] for spec in SPECS],
                        help="Run one worker; finalize the suite separately once all workers finish")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--inference-threads", type=int)
    parser.add_argument("--sample-workers", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    import torch
    import vosk
    if args.parts < 1 or not 0 <= args.part < args.parts:
        parser.error("Require 0 <= part < parts")
    torch.set_num_threads(args.inference_threads or args.threads)
    torch.set_num_interop_threads(1)
    vosk.SetLogLevel(-1)
    samples = load_manifest(args.manifest)
    sample_ids = {s["id"] for s in samples}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "run-config.json"
    config = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "models": [{"name": n, "adapter": c.__name__, "path": str(Path("models", p).resolve())}
                   for n, c, p in SPECS],
        "torch_threads": args.threads, "torch_interop_threads": 1,
        "sample_rate": 16000, "device": "cpu", "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(), "normalization": "nfkc_casefold_punctuation_whitespace",
    }
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Run configuration changed; choose a new output directory")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    all_rows = []
    metadata = {}
    for name, factory, folder in SPECS:
        if args.only and name != args.only:
            continue
        base_checkpoint = args.output / f"{name}.jsonl"
        suffix = f".part{args.part}" if args.parts > 1 else ""
        checkpoint = args.output / f"{name}{suffix}.jsonl"
        model_metadata = args.output / f"{name}{suffix}.metadata.json"
        previous = []
        sources = [checkpoint]
        if args.parts > 1:
            sources = [base_checkpoint, checkpoint]
        else:
            sources += sorted(args.output.glob(f"{name}.part*.jsonl"))
        for source in sources:
            if not source.exists():
                continue
            data = source.read_bytes()
            complete = data.rfind(b"\n") + 1
            if complete != len(data):
                with source.open("r+b") as stream:
                    stream.truncate(complete)
            previous += [json.loads(line) for line in data[:complete].splitlines()]
        seen = {r["sample_id"] for r in previous}
        if len(seen) != len(previous) or not seen <= sample_ids or any(r["model"] != name for r in previous):
            raise ValueError(f"Invalid checkpoint for {name}")
        pending = [sample for sample in samples if sample["id"] not in seen]
        if args.parts > 1:
            partition_ids = {sample["id"] for index, sample in enumerate(samples) if index % args.parts == args.part}
            pending = [sample for sample in pending if sample["id"] in partition_ids]
        print(f"{name}: {len(previous)}/{len(samples)} already saved; {len(pending)} pending", flush=True)
        if pending:
            if args.finalize_only:
                raise ValueError(f"{name} still has {len(pending)} pending samples")
            started = time.perf_counter()
            model = factory(name, Path("models") / folder)
            init_seconds = time.perf_counter() - started
            info = {**model.metadata, "initialization_seconds": init_seconds,
                    "torch_threads": torch.get_num_threads(), "sample_workers": args.sample_workers,
                    "started_at_utc": datetime.now(timezone.utc).isoformat()}
            if model_metadata.exists():
                old = json.loads(model_metadata.read_text())
                for key in model.metadata:
                    if old.get(key) != model.metadata[key]:
                        raise ValueError(f"Model metadata changed for {name}")
            else:
                model_metadata.write_text(json.dumps(info, indent=2) + "\n")
            completed = len(previous)
            with checkpoint.open("a", encoding="utf-8") as stream:
                def save(row):
                    nonlocal completed
                    row["torch_threads"] = torch.get_num_threads()
                    row["sample_workers"] = args.sample_workers
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed += 1
                    if completed % 25 == 0 or completed == len(samples):
                        print(f"{name}: {completed}/{len(samples)}; latest={row['sample_id']}; "
                              f"status={row['status']}", flush=True)
                if args.sample_workers > 1:
                    if factory is not VoskModel:
                        raise ValueError("Shared-model sample workers are supported only for Vosk recognizers")
                    added = []
                    with ThreadPoolExecutor(max_workers=args.sample_workers) as pool:
                        futures = [pool.submit(evaluate, [sample], [model]) for sample in pending]
                        for future in as_completed(futures):
                            row = future.result()["results"][0]
                            save(row)
                            added.append(row)
                else:
                    added = evaluate(pending, [model], on_result=save)["results"]
            previous += added
            del added, model
            gc.collect()
        metadata_source = model_metadata
        if not metadata_source.exists():
            metadata_source = next(iter(sorted(args.output.glob(f"{name}.part*.metadata.json"))))
        metadata[name] = json.loads(metadata_source.read_text())
        if args.parts == 1:
            metadata[name]["partition_metadata"] = [json.loads(path.read_text()) for path in
                                                    sorted(args.output.glob(f"{name}.part*.metadata.json"))]
        all_rows += previous
    if args.only:
        print(f"WORKER COMPLETE: {args.only}, {len(all_rows)} attempts", flush=True)
        return
    successful_sets = [{r["sample_id"] for r in all_rows if r["model"] == name and r["status"] == "ok"}
                       for name, _, _ in SPECS]
    common = set.intersection(*successful_sets)
    effective_path = args.output / "whisper-effective-generation-config.json"
    if effective_path.exists():
        effective = json.loads(effective_path.read_text())
        metadata["whisper"]["effective_decoding"] = {
            key: effective.get(key) for key in ("num_beams", "max_length", "max_new_tokens", "language", "task", "return_timestamps")
        }
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "python_version": platform.python_version(), "sample_rate": 16000,
              "normalization": config["normalization"], "manifest": config["manifest"],
              "run_config": config, "models": metadata, "common_sample_ids": sorted(common),
              "results": all_rows}
    from stt_bench.reporting import analyze
    analysis = analyze(report, samples)
    report["summary"] = {}
    for name, item in analysis["models"].items():
        summary = {"successful_samples": item["successful"], "failed_samples": item["failed"],
                   "macro_speaker_wer": item["macro_speaker_wer"],
                   "transcription_seconds": item["all_successful"].get("transcription_seconds", 0),
                   "real_time_factor": item["all_successful"].get("real_time_factor")}
        for prefix, key in (("successful", "all_successful"), ("common", "common")):
            for unit, metric in (("word", "wer"), ("character", "cer")):
                subset = item[key]
                summary[f"{prefix}_{unit}"] = ({**subset[unit], "rate": subset[metric]}
                                               if subset["samples"] else None)
        report["summary"][name] = summary
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    # An atomic final file distinguishes a complete six-model run from checkpoints.
    temp = args.output / "report.json.tmp"
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temp.replace(args.output / "report.json")
    fields = ["sample_id", "speaker", "session", "microphone", "model", "status", "reference",
              "hypothesis", "wer", "cer", "duration_seconds", "transcription_seconds", "real_time_factor", "error"]
    with (args.output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            flat = {key: row.get(key, "") for key in fields}
            flat.update(wer=row.get("word", {}).get("rate", ""), cer=row.get("character", {}).get("rate", ""))
            writer.writerow(flat)
    errors = sum(r["status"] != "ok" for r in all_rows)
    print(f"COMPLETE: {len(all_rows)} attempts, {len(common)} common samples, {errors} failures", flush=True)


if __name__ == "__main__":
    main()
