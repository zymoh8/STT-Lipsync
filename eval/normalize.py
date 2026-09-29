"""
Text normalisation.

THE MOST IMPORTANT FILE IN THIS REPO. Read this before changing anything.

Different engines emit wildly different surface text for identical
recognition. Whisper writes "I have 25 dollars." with capitals,
punctuation and digits. Vosk writes "i have twenty five dollars".
Scored against each other raw, Whisper looks catastrophically worse for
reasons that have nothing to do with how well it heard the audio.

So: every hypothesis AND every reference goes through normalize() before
scoring, using the SAME settings. If you change this file, every number
in every past result becomes incomparable. Re-run everything.
"""
from __future__ import annotations

import re
import unicodedata

# Numbers we expect in read-speech corpora. Extend as needed - but if you
# extend it, re-run the whole benchmark.
_UNITS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen",
}
_TENS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}

# Contractions -> expanded. Engines disagree constantly on these.
_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "let's": "let us",
}

# Filler / hesitation tokens. Dysarthric corpora and engines both produce
# these inconsistently; dropping them avoids penalising an engine for
# transcribing a real disfluency the reference ignored.
_FILLERS = {"uh", "um", "erm", "hmm", "mm", "eh", "ah"}

# Corpus annotation markers, e.g. TORGO's [noise] or UA-Speech brackets.
_BRACKETED = re.compile(r"[\[\(<][^\]\)>]*[\]\)>]")


def _int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + ("" if r == 0 else " " + _UNITS[r])
    if n < 1000:
        h, r = divmod(n, 100)
        out = _UNITS[h] + " hundred"
        return out if r == 0 else out + " " + _int_to_words(r)
    if n < 1_000_000:
        th, r = divmod(n, 1000)
        out = _int_to_words(th) + " thousand"
        return out if r == 0 else out + " " + _int_to_words(r)
    m, r = divmod(n, 1_000_000)
    out = _int_to_words(m) + " million"
    return out if r == 0 else out + " " + _int_to_words(r)


def _expand_numbers(text: str) -> str:
    def sub(m: re.Match) -> str:
        raw = m.group(0)
        try:
            return " " + _int_to_words(int(raw)) + " "
        except (ValueError, KeyError):
            return " " + raw + " "
    return re.sub(r"\d+", sub, text)


def normalize(text: str, *, drop_fillers: bool = True,
              expand_numbers: bool = True) -> str:
    """Canonical form used for BOTH references and hypotheses."""
    if text is None:
        return ""
    # Unicode fold: curly quotes, accented chars, non-breaking spaces
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-")
    t = t.lower()

    t = _BRACKETED.sub(" ", t)

    for a, b in _CONTRACTIONS.items():
        t = t.replace(a, b)

    if expand_numbers:
        t = _expand_numbers(t)

    # Hyphens and slashes become word breaks, not deletions:
    # "twenty-five" -> "twenty five", not "twentyfive"
    t = re.sub(r"[-/]", " ", t)

    # Strip everything that isn't a letter, digit or space. Apostrophes go
    # too, so "dont" == "don't" after expansion leftovers.
    t = re.sub(r"[^a-z0-9 ]+", " ", t)

    words = t.split()
    if drop_fillers:
        words = [w for w in words if w not in _FILLERS]
    return " ".join(words)


def tokens(text: str, **kw) -> list[str]:
    n = normalize(text, **kw)
    return n.split() if n else []
