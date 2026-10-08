"""Deterministic Levenshtein scoring with linear working memory."""

from dataclasses import asdict, dataclass
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    # Preserve apostrophes in contractions; turn other punctuation into boundaries.
    text = "".join(" " if unicodedata.category(c).startswith("P") and c != "'" else c
                   for c in text)
    return " ".join(text.split())


@dataclass(frozen=True)
class Counts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_units: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        # Empty references: zero for silence, insertion count for hallucinations.
        return self.errors / self.reference_units if self.reference_units else float(self.errors)

    def __add__(self, other):
        return Counts(*(a + b for a, b in zip(asdict(self).values(), asdict(other).values())))

    def to_dict(self):
        return {**asdict(self), "errors": self.errors, "rate": self.rate}


def edit_counts(reference, hypothesis) -> Counts:
    # Cells store (total, substitutions, deletions, insertions). Ties prefer
    # substitution, then deletion, then insertion. Counts can vary across optimal
    # alignments, but the total edit distance remains identical.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, token in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, predicted in enumerate(hypothesis, 1):
            if token == predicted:
                current.append(previous[j - 1])
                continue
            t, s, d, ins = previous[j - 1]
            substitution = (t + 1, s + 1, d, ins)
            t, s, d, ins = previous[j]
            deletion = (t + 1, s, d + 1, ins)
            t, s, d, ins = current[j - 1]
            insertion = (t + 1, s, d, ins + 1)
            current.append(min((substitution, deletion, insertion), key=lambda cell: cell[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return Counts(substitutions, deletions, insertions, len(reference))


def score(reference: str, hypothesis: str, normalize_text: bool = True):
    ref = normalize(reference) if normalize_text else reference
    hyp = normalize(hypothesis) if normalize_text else hypothesis
    return {
        "normalized_reference": ref,
        "normalized_hypothesis": hyp,
        "word": edit_counts(ref.split(), hyp.split()).to_dict(),
        "character": edit_counts(ref, hyp).to_dict(),
    }
