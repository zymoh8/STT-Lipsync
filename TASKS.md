# Work order — speech recognition benchmark

Forward this. It's written for whoever is doing the software.

## What we're settling

Our LipSync dictation add-on currently runs
`vosk-model-small-en-us-0.15` — a 40 MB model, the smallest in its
family, chosen for phones. We have no idea whether it's the right choice
because we have never measured anything.

Specifically unknown:

- How bad it actually is, in a number
- Whether a larger model fixes it, and by how much
- Whether any of this holds for **dysarthric speech**, which is the only
  population that matters for this project
- What fits in 4 GB on a Pi 4 while staying fast enough to use

Nothing else in the project can be decided until this is answered. The
enclosure design is blocked on it — the board choice depends on the
model, and the model isn't picked.

## Deliverable

A results table, reproducible with one command, covering every candidate
model, broken down by dysarthria severity, with WER, real-time factor and
peak memory.

Plus the harness that produced it, because the next team will need to
re-run this with models that don't exist yet.

## What needs the Pi, and what doesn't

Almost none of this needs the Raspberry Pi. Only two numbers do:
**real-time factor** and **peak memory** — and they come last.

Word error rate is a property of the model and the audio, not the
machine. Run the whole accuracy sweep on laptops, where it is 10-50x
faster, then confirm on the Pi at the end.

Pi NOT required: corpus loaders, manifests, normalisation, scoring,
reporting, recording your own test audio, installing and learning the
model APIs, the entire accuracy comparison.

Pi required: timing and memory measurement; building whisper.cpp for
ARM; confirming a model actually loads in 4 GB.

Two cautions. Timing runs need the Pi **otherwise idle** — no second SSH
session doing anything, or the RTF numbers are noise. And quantised
builds can differ slightly between ARM and x86, so re-run the baseline
and the winner on the Pi and check the WER matches your laptop numbers
before reporting.

## Week 1

1. Clone the harness. Run `python tests/smoke_test.py`. Must print
   `ALL PASSED`.
2. Read `eval/normalize.py` and `README.md`, specifically the section
   "Four things that will silently ruin your results". These are not
   style notes — each one produces a plausible-looking table that means
   nothing.
3. On the Pi: download the candidate Vosk models, build whisper.cpp,
   install Moonshine. Fill in `config.yaml`.
4. Sanity run against your own voices: record ~20 sentences each with
   the project mic, `tools/make_manifest.py folder`, run the harness.
   You are not measuring anything real yet — you are proving the
   pipeline works end to end on real audio and real models.

Expected output of week 1: a WER table for three or four models on your
own speech. It will be optimistic and unrepresentative. That's fine —
it's a pipeline test, and it's also the baseline we currently don't have.

## Week 2+ (TORGO needs no access request — download it today)

5. Download TORGO from https://www.cs.toronto.edu/~complingweb/data/TORGO/
   (M, F, MC, FC tarballs; 18 GB uncompressed; laptop disk, not the Pi's
   SD card). Convert with `tools/convert_audio.sh`, then
   `tools/make_manifest.py torgo …` then **`--verify`**. The loaders were
   written from published descriptions, not against real data. Expect to
   fix `build_torgo` in `eval/datasets.py`. That's the only place
   directory layout lives.
6. Fill `TORGO_SEVERITY` from the corpus documentation. Without it the
   severity table — the whole point — is one column of `unknown`.
7. Run on the **dev** split. Iterate there.
8. `--worst` on each model. Read twenty failures per model with your own
   eyes before believing any aggregate. This catches normalisation bugs,
   broken paths, and references that aren't transcripts. It is the single
   highest-value hour in this whole task.
9. Run the **test** split once, at the end. Report those numbers.

## Rules

- Never tune on the test split. Split is by speaker, seeded, don't change
  the seed.
- Timing and memory come from the Pi. WER can be computed anywhere.
- Don't normalise inside an adapter. Adapters return raw engine output;
  `normalize()` handles the rest, identically for every engine.
- If you change `eval/normalize.py`, every previous result is void.
  Re-run everything.

## If something looks wrong

Run the dummy engine at `error_rate: 0.0`. It must report exactly 0.00 %
WER. If it doesn't, the bug is in normalisation or the manifest, not the
engine you were testing.

## Things worth trying if there's time

- Vosk restricted grammar for a constrained vocabulary (`grammar:` in
  the adapter config). For an AAC device, a domain vocabulary — *nurse,
  water, pain, help* — may beat a bigger general model outright.
- Character error rate as well as word error rate. WER saturates near
  1.0 on severe dysarthria and stops discriminating between models; CER
  keeps resolving.
- Per-speaker WER, not just per-severity. Severity labels are coarse and
  individual variation inside a band is large.
