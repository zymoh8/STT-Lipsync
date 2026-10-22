# Speech-to-text testing framework

Read audio → run model adapters → compare against human transcripts → JSON/CSV reports.
Python 3.10+; the core and tests use only the standard library. Run commands from this directory.

## Try the pipeline without a model

```bash
python3 -m unittest discover -s tests -v
python3 -m stt_bench demo
python3 -m stt_bench inspect demo/silence.wav
python3 -m stt_bench run demo/manifest.jsonl --predictions saved=demo/predictions.json --output results/demo
```

The demo generates silence and scores an explicitly saved empty prediction. It verifies the
pipeline; it does **not** measure speech recognition accuracy or run an AI model.

## Prepare real data

Each recording needs a manually checked reference transcript. Start with short recordings
covering the accents, noise, microphones, and vocabulary relevant to your project. Keep the
same evaluation recordings for all models; keep tuning/training data separate.

Use 16,000 Hz, 16-bit, mono PCM WAV files. The `prepare` command needs FFmpeg on your PATH:

```bash
python3 -m stt_bench prepare recordings/example.mp3 data/audio/example.wav
python3 -m stt_bench inspect data/audio/example.wav
```

Conversion decodes audio, downmixes channels, and resamples; it does not improve the original
recording quality. Existing destinations are refused. To preserve another sample rate, use
`--sample-rate` consistently on prepare, inspect, and run, with a compatible model.

Create `data/manifest.jsonl`, one JSON object per line:

```json
{"id":"example","audio":"audio/example.wav","reference":"The actual words spoken."}
```

Audio paths resolve relative to the manifest file. IDs must be unique. An empty reference
is permitted for silence/non-speech tests. `data/manifest.example.jsonl` illustrates the
schema; its placeholder audio files are not included.

## Run Vosk

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[vosk]"
```

Download and extract a suitable language model from the [official Vosk model catalog](https://alphacephei.com/vosk/models)
into `models/`. Point to the extracted directory, not the ZIP. Models are loaded locally;
this framework does not automatically download weights or send audio to a service.

```bash
python3 -m stt_bench run data/manifest.jsonl \
  --vosk small=models/vosk-model-small-en-us-0.15 \
  --output results/vosk-small
```

Compare multiple downloaded models by repeating `--vosk` with unique names:

```bash
python3 -m stt_bench run data/manifest.jsonl \
  --vosk small=models/vosk-model-small-en-us-0.15 \
  --vosk large=models/vosk-model-en-us-0.22 \
  --vosk lgraph=models/vosk-model-en-us-0.22-lgraph \
  --output results/comparison
```

All three use the same `VoskModel` adapter; downloaded model folders are weights,
not Python modules to import in `models.py`. Vosk handles the lgraph model's
dynamic graph files when loading its directory. This benchmark uses its default
vocabulary, without a custom grammar.

This adapter follows [Vosk's Python example](https://github.com/alphacep/vosk-api/blob/master/python/example/test_simple.py):
it passes PCM chunks to a fresh recognizer for each recording and joins completed segments
with the final result. The model weights are loaded once per run. WAV container headers
are excluded from the bytes passed to the recognizer.

## Run Whisper, Wav2Vec2, and Parakeet

Activate `conda activate STT`, or recreate the environment with
`conda env create -f environment.yml`. For an existing Python environment,
install `python -m pip install -e ".[vosk,transformers]"`.

Download the official Transformers weights (skip duplicate backend formats):

```bash
hf download openai/whisper-small --local-dir models/whisper-small --include '*.json' '*.safetensors' '*.txt' '*.model' 'README.md' 'LICENSE*'
hf download facebook/wav2vec2-base-960h --local-dir models/wav2vec2-base-960h --include '*.json' '*.safetensors' '*.txt' '*.model' 'README.md' 'LICENSE*'
hf download nvidia/parakeet-tdt-0.6b-v3 --local-dir models/parakeet-tdt-0.6b-v3 --include '*.json' '*.safetensors' '*.txt' '*.model' 'README.md' 'LICENSE*'
```

Run all three, optionally adding the existing `--vosk` options to the same command:

```bash
python -m stt_bench run data/manifest.jsonl \
  --whisper whisper=models/whisper-small \
  --wav2vec2 wav2vec2=models/wav2vec2-base-960h \
  --parakeet parakeet=models/parakeet-tdt-0.6b-v3 \
  --output results/non-vosk
```

These adapters load only local files, run float32 CPU inference, and require
16 kHz, 16-bit mono PCM WAV. All requested models remain loaded for the run;
run models separately if memory is limited. Whisper transcribes in the detected
language (rather than translating). Wav2Vec2 uses greedy CTC decoding without
an external language model, with 30-second chunks and 5-second overlap.
Parakeet processes each complete recording, so start with short recordings.
These are file transcription adapters, not streaming latency measurements.

Official model cards: [Whisper Small](https://huggingface.co/openai/whisper-small),
[Wav2Vec2 Base 960h](https://huggingface.co/facebook/wav2vec2-base-960h),
and [Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
Parakeet weights use CC BY 4.0; retain the downloaded attribution/license files.

## Evaluate TORGO male dysarthric speech

Extract `M.tar.bz2` into `data/torgo/` so it contains `M01/` through `M05/`.
Keep the archive and extracted data out of Git. Only the `prompts/` and
`wav_headMic/` or `wav_arrayMic/` files are needed for this acoustic benchmark.

```bash
conda activate STT
python -m stt_bench import-torgo data/torgo \
  --microphone headMic --output data/torgo-manifest.jsonl

python -m stt_bench run data/torgo-manifest.jsonl \
  --vosk vosk-small=models/vosk-model-small-en-us-0.15 \
  --vosk vosk-lgraph=models/vosk-model-en-us-0.22-lgraph \
  --vosk vosk-large=models/vosk-model-en-us-0.22 \
  --whisper whisper=models/whisper-small \
  --wav2vec2 wav2vec2=models/wav2vec2-base-960h \
  --parakeet parakeet=models/parakeet-tdt-0.6b-v3 \
  --output results/torgo-headmic
```

The importer selects M01–M05, excludes controls, pairs each WAV with its session's
same-numbered `.txt` prompt, validates audio, and creates unique IDs across
speakers and sessions. Choose one microphone for the comparison; using both
would count two recordings of the same utterance. Missing microphone sessions
are logged without switching microphones. Use `--microphone arrayMic` and a new
output manifest to evaluate that microphone separately.

Review `data/torgo-manifest.audit.json` for missing, malformed, empty, `xxx`,
image-description, and annotated/phonetic prompts. Such files are excluded,
not labeled as silence. The filters are conservative and cannot identify every
non-word task. Supplied prompts describe intended speech and must be checked
against the audio before interpreting WER as verbatim transcription accuracy.
Image-description recordings need separately transcribed references.

JSON/CSV results retain speaker, session, microphone, and reference type so you
can compare speakers separately as well as viewing corpus scores. Start with
a small manifest to estimate runtime; all six models are loaded together and
CPU inference over the full corpus can take substantial time. This cohort
measures recognition on male dysarthric speech, not general-population accuracy.

See the [official TORGO documentation](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
for recording notes, task definitions, and academic-use/citation requirements.

The current extracted male archive yields 1,920 head-microphone recordings after
filtering, with 118 exclusions/session issues recorded in the audit. The generated
`data/torgo-pilot.jsonl` selects one word and one sentence per speaker (10 clips).
Use that manifest in the same six-model command for a quick integration run.
Its initial results are in `results/torgo-pilot/`: all 60 predictions completed.
Whisper used automatic language detection and produced incorrect-language and
repetitive outputs on some clips. These pilot scores are diagnostic, not a
validated ranking of the models on the full cohort.

## Understand the reports

The completed six-model TORGO benchmark is available in the
[PDF report](output/pdf/torgo-six-model-benchmark.pdf), including model specifications
and Raspberry Pi feasibility guidance. Model weights, downloaded TORGO audio,
generated manifests/audits, and raw result files are kept out of Git. Generate a
local manifest with `import-torgo` before running the benchmark below.

For a complete six-model run with per-recording checkpoints and progress logs:

```bash
python scripts/run_full_benchmark.py data/torgo-manifest.jsonl \
  --output results/torgo-full --threads 4
python scripts/validate_benchmark.py results/torgo-full/report.json \
  --manifest data/torgo-manifest.jsonl
python scripts/make_benchmark_report.py results/torgo-full/report.json \
  --manifest data/torgo-manifest.jsonl \
  --output output/pdf/torgo-six-model-benchmark.pdf
```

The first command resumes completed IDs from its JSONL checkpoints and verifies
the saved manifest/configuration before continuing. `--only MODEL` runs a single
model worker; `--finalize-only` merges completed workers without inference.
Whisper can be partitioned into separate processes with `--parts N --part I`;
each partition writes its own checkpoint. Keep partition counts fixed during a
run. `--sample-workers N` is available only for Vosk's independent recognizers.
Workers must finish before finalizing. Concurrent-worker timing is observational
under shared load and should not be treated as an isolated speed comparison.

The PDF generator needs `reportlab`, `matplotlib`, and Arial font files from
macOS. It writes `analysis.json`, `leaderboard.csv`, and `per-speaker.csv` beside
the raw report. The primary ranking averages corpus WER equally over speakers;
it also reports pooled WER/CER, per-recording averages, exact matches, timing,
word/multi-word subsets, failures, and a duration-outlier sensitivity check.

`report.json` contains raw and normalized transcripts, edit counts, sample durations,
transcription times, errors, PCM hashes, model metadata, and aggregated scores.
`samples.csv` provides per-recording results for spreadsheets. Choose a new output
directory for each run; previous reports are not overwritten.

- **WER** = (word substitutions + deletions + insertions) / reference word count.
- **CER** applies the same formula to characters, including spaces.
- Lower scores are better; 0 means identical after the chosen normalization.
  WER can exceed 100% due to inserted words, so it is not a bounded accuracy percentage.
- Corpus scores sum edit counts and reference lengths rather than averaging sample rates.
- With no reference units, the rate is 0 for an empty prediction and the insertion count
  otherwise. This follows the [JiWER empty-reference convention](https://github.com/jitsi/jiwer).
- Default normalization applies Unicode NFKC, case folding, whitespace collapse, and
  punctuation replacement with spaces, preserving apostrophes. It does not rewrite numbers
  or expand contractions. Use `--raw-text` to disable normalization. WER still splits on
  whitespace; for languages without space-separated words, prioritize CER or add a tokenizer.
- Failed samples are recorded separately, never scored as successful silence. The command
  exits with status 1 if any sample fails; it still writes the completed report.
- `successful_word`/`successful_character` cover each model's successful samples.
  `common_word`/`common_character` cover only recordings completed by **every** model;
  those are the comparable corpus scores. Check coverage before drawing conclusions.
  A score is null when no recordings qualify.
- Real-time factor = transcription seconds / audio seconds; below 1 is faster than playback.
  It excludes file decoding and initial model loading, includes per-recording recognizer
  setup, and is a single-run measurement. Warm-up and hardware affect it. Saved predictions
  only time a lookup and must not be used to compare inference speed.

The edit-distance implementation uses linear working memory, but quadratic time; use short
utterances for this initial suite. It measures transcript agreement, not semantic correctness
or speaker attribution.

## Add another model

Implement `STTModel` from `stt_bench/models.py`: expose `name`, JSON-serializable `metadata`,
and `transcribe(audio: Audio, sample_id: str) -> str`. `Audio.pcm` holds decoded signed
16-bit little-endian samples; `sample_rate` describes them. Adapters can turn these samples
into tensors or a format required by their model. Add a CLI factory/option in `cli.py`, or
call `evaluate(load_manifest(path), [your_model])` directly from Python.

You can score outputs from any external model immediately with `--predictions NAME=FILE`.
The file is a JSON object mapping manifest IDs to transcript strings:

```json
{"example":"The model's recognized words."}
```

Missing predictions are errors; they are never silently substituted with empty strings.
