"""
Word and character error rate.

WER = (substitutions + deletions + insertions) / words_in_reference

Uses jiwer when available (faster, well tested); otherwise falls back to
a pure-python Levenshtein so the harness runs with zero dependencies.
Both paths are checked against each other in tests/test_metrics.py.

Note WER can exceed 1.0 - an engine that hallucinates 30 words against a
5 word reference scores >1. That is correct, not a bug. Do not clamp it;
clamping hides exactly the failure mode you most want to see on
dysarthric audio.
"""
from __future__ import annotations

from dataclasses import dataclass

from .normalize import tokens

try:  # optional
    import jiwer  # type: ignore
    _HAS_JIWER = True
except Exception:  # pragma: no cover
    _HAS_JIWER = False


@dataclass
class ErrorCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_len: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.ref_len == 0:
            # Empty reference: 0.0 if hypothesis also empty, else 1.0.
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.ref_len


def _levenshtein_counts(ref: list[str], hyp: list[str]) -> ErrorCounts:
    """Edit distance with S/D/I breakdown via backtrace."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1],  # substitute
                                  d[i - 1][j],      # delete
                                  d[i][j - 1])      # insert
    i, j = n, m
    c = ErrorCounts(ref_len=n)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            c.substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            c.deletions += 1
            i -= 1
        else:
            c.insertions += 1
            j -= 1
    return c


def word_errors(reference: str, hypothesis: str, **norm_kw) -> ErrorCounts:
    ref = tokens(reference, **norm_kw)
    hyp = tokens(hypothesis, **norm_kw)
    if _HAS_JIWER:
        try:
            out = jiwer.process_words([" ".join(ref)], [" ".join(hyp)])
            return ErrorCounts(out.substitutions, out.deletions,
                               out.insertions, len(ref))
        except Exception:
            pass
    return _levenshtein_counts(ref, hyp)


def wer(reference: str, hypothesis: str, **norm_kw) -> float:
    return word_errors(reference, hypothesis, **norm_kw).wer


def cer(reference: str, hypothesis: str, **norm_kw) -> float:
    """Character error rate - more forgiving, useful when WER saturates
    near 1.0 on severe dysarthria and stops discriminating."""
    from .normalize import normalize
    r = list(normalize(reference, **norm_kw).replace(" ", ""))
    h = list(normalize(hypothesis, **norm_kw).replace(" ", ""))
    c = _levenshtein_counts(r, h)
    return c.wer


def corpus_wer(rows: list[ErrorCounts]) -> float:
    """Aggregate WER over many utterances.

    This pools errors and reference words across the corpus rather than
    averaging per-utterance WERs. The two differ, and pooling is the
    standard. Averaging per-utterance over-weights short utterances,
    which in dysarthric corpora are often isolated words - so the
    averaged figure flatters engines that do well on single words.
    """
    e = sum(r.errors for r in rows)
    n = sum(r.ref_len for r in rows)
    return e / n if n else 0.0
