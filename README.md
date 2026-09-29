# LipSync STT — evaluation harness

Benchmarks speech-to-text engines for the LipSync assistive dictation
project, under the constraint that the winner has to run on a 4 GB
Raspberry Pi 4.

**The question this exists to answer:** which model gives the best word
error rate *on dysarthric speech*, within the compute budget of the
hardware. Not general-purpose accuracy — that problem is solved and the
phone in the user's hand already does it better than we can. Our users
are the ones mainstream ASR fails.

The harness is the deliverable, not scaffolding. Once it works, "test
another model" costs an hour instead of a week, which is what makes it
useful to whoever picks this project up next.

---

## Quick start

```bash
git clone <this repo> && cd lipsync-stt-eval
python tests/smoke_test.py          # no models or data needed
```

That must print `ALL PASSED` before you trust any number this machine
produces. It generates fake audio, runs fake engines, and checks the
whole chain — manifest → adapter → scoring → report.

Then:

```bash
pip install -r requirements.txt     # optional but recommended
cp config.example.yaml config.yaml  # edit the model paths
```

## Workflow

```bash
# 1. Build a manifest from a corpus (one row per utterance)
python tools/make_manifest.py torgo /data/TORGO data/torgo.csv

# 2. VERIFY IT BY HAND. Play five files, read five references.
python tools/make_manifest.py --verify data/torgo.csv

# 3. Smoke-run on 10 utterances before committing to hours of Pi time
python run_eval.py --manifest data/torgo.csv --out results/tmp.csv --limit 10

# 4. Full run on the dev split
python run_eval.py --manifest data/torgo.csv --holdout dev \
                   --out results/dev.csv

# 5. Read the tables
python run_eval.py --report results/dev.csv

# 6. Eyeball the failures — this catches bugs no assertion will
python run_eval.py --report results/dev.csv --worst vosk-0.22

# 7. Once. At the end. On the held-out speakers.
python run_eval.py --manifest data/torgo.csv --holdout test \
                   --out results/test.csv
```

## Layout

```
run_eval.py               runner + reporter CLI
config.example.yaml       which models to test
eval/normalize.py         text normalisation      <- read this one
eval/metrics.py           WER / CER
eval/datasets.py          corpus -> manifest; severity maps live here
eval/report.py            aggregation and tables
eval/adapters/            one file per engine
tools/make_manifest.py    build + verify manifests
tests/smoke_test.py       end-to-end check, no deps
```

## Adding an engine

Write one file in `eval/adapters/`:

```python
from .base import Adapter, register

@register
class MyAdapter(Adapter):
    name = "myengine"

    def __init__(self, model_path: str, **kw):
        super().__init__(**kw)
        self.model_path = model_path

    def prepare(self):
        self._m = load(self.model_path)      # once, untimed

    def _transcribe(self, wav_path: str) -> str:
        return self._m.run(wav_path)         # raw text, don't normalise
```

Add it to `config.yaml`. Nothing else changes.

---

## Four things that will silently ruin your results

**1. Normalisation.** Whisper writes `"I have 25 dollars."`; Vosk writes
`"i have twenty five dollars"`. Scored raw, Whisper looks catastrophically
worse for reasons that have nothing to do with hearing. `normalize()` is
applied to both reference and hypothesis with identical settings. If you
change that file, every previous number becomes incomparable and you
re-run everything. The smoke test asserts a vosk-style and a
whisper-style output of the same content score identically — keep that
test passing.

**2. Splitting by utterance instead of by speaker.** `split_holdout()`
splits by speaker on purpose. Utterance-level splits put the same
speaker in dev and test, which flatters anything speaker-adaptive. Every
real deployment meets a speaker it has never heard.

**3. Reporting one averaged WER.** Report **by severity band**. "Holds up
through moderate dysarthria, collapses past it" is a finding. A single
pooled number across controls and severe speakers is a statistic that
answers nobody's question. Also note the aggregate here pools errors over
reference words rather than averaging per-utterance WERs — on corpora
full of single-word prompts those differ a lot, and averaging flatters
engines that do well on isolated words.

**4. An unverified manifest.** If audio and references are mismatched, or
severity is `unknown` everywhere, the output table still looks completely
reasonable. `--verify` exists for this. Use it every time you regenerate.

---

## What to report

Three tables, all produced by `--report`:

| | |
|---|---|
| **Overall** | WER, median RTF, peak RSS per model |
| **By severity** | the one that matters |
| **By prompt type** | isolated words vs sentences behave differently |

Report RTF and peak memory alongside accuracy. The constraint is a 4 GB
Pi — a model that is 2 % better and does not fit is worthless to this
project. Accuracy-per-resource *is* the contribution.

Timing numbers only mean anything measured on the target hardware. WER
can be computed anywhere; RTF and RSS must come from the Pi.

## Candidate models

| model | size | LibriSpeech WER | note |
|---|---|---|---|
| vosk-model-small-en-us-0.15 | 40 MB | 9.85 % | current baseline |
| vosk-model-en-us-0.22-lgraph | 128 MB | 7.82 % | small-board option |
| vosk-model-en-us-0.22 | 1.8 GB | 5.69 % | the free win |
| vosk-model-en-us-0.42-gigaspeech | 2.3 GB | 5.64 % | most speaker-diverse data |
| whisper.cpp base.en | ~142 MB | — | better on accented speech |
| whisper.cpp small.en | ~466 MB | — | viable: we are not streaming |
| Moonshine base | small | — | built for edge + short utterances |

Those WERs are LibriSpeech test-clean — read audiobook speech, the
easiest benchmark there is. The same vosk-0.22 model scores 29.78 % on
call-centre audio. Expect dysarthric numbers far worse. What transfers is
the ordering, not the values — and even the ordering may not hold, which
is the entire reason this harness exists.

## Corpora

- **TORGO** — CP and ALS speakers, acoustic + articulatory.
  **Free direct download**, no licence or agreement needed:
  https://www.cs.toronto.edu/~complingweb/data/TORGO/
  Four tarballs — `M` and `F` (dysarthric), `MC` and `FC` (control).
  18 GB uncompressed. Free for academic non-profit use *provided you
  cite the TORGO paper* in anything you publish or present. Put that
  citation in the symposium slides.
  You do not need the `.pos` articulatory files — those need the TAPADM
  toolbox to read and are irrelevant to ASR benchmarking. Audio only.
- **UA-Speech** — UIUC, isolated words from CP speakers. Request-gated;
  lower priority now that TORGO is free and immediate.
- **Speech Accessibility Project** — 1,500 hours from ~999 participants.
  The better dataset, and slower to get: a Data Use Agreement needing
  two signatures (you plus an authorised university representative) and
  a one-page proposal to speechaccessibility@beckman.illinois.edu.
  Start it in parallel; work on TORGO while it clears.

Severity labels are **not** in the directory structures. Fill
`TORGO_SEVERITY` and `UASPEECH_SEVERITY` in `eval/datasets.py` from the
documentation shipped with the licensed corpus. Until you do, the
severity table is one column of `unknown`.

Using these existing de-identified corpora is secondary analysis and
does not require IRB review. Recruiting people and recording them does.

## Status

The corpus loaders were written from published descriptions of TORGO and
UA-Speech, not against the real data — nobody had a licensed copy when
this was written. Expect the first `make_manifest` run to need fixes.
Directory layout assumptions live only in `build_torgo` /
`build_uaspeech`; fix those and nothing else changes.

Everything else is tested and passing.
