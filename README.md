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
  --vosk large=models/your-other-extracted-model \
  --output results/comparison
```

This adapter follows [Vosk's Python example](https://github.com/alphacep/vosk-api/blob/master/python/example/test_simple.py):
it passes PCM chunks to a fresh recognizer for each recording and joins completed segments
with the final result. The model weights are loaded once per run. WAV container headers
are excluded from the bytes passed to the recognizer.

## Understand the reports

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
