"""Aggregate a results CSV into the tables you actually present."""
from __future__ import annotations

import csv
from collections import defaultdict

from .datasets import SEVERITIES

SEV_ORDER = {s: i for i, s in enumerate(SEVERITIES)}


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pool(rows: list[dict]) -> dict:
    """Pooled WER - total errors over total reference words.

    NOT the mean of per-utterance WERs. See metrics.corpus_wer for why
    that distinction matters on corpora full of single-word prompts.
    """
    errs = sum(int(r["errors"]) for r in rows)
    refs = sum(int(r["ref_len"]) for r in rows)
    rtfs = [float(r["rtf"]) for r in rows if r.get("rtf")]
    return {
        "n": len(rows),
        "wer": errs / refs if refs else 0.0,
        "rtf_median": sorted(rtfs)[len(rtfs) // 2] if rtfs else 0.0,
        "peak_rss_mb": max((float(r.get("peak_rss_mb") or 0) for r in rows),
                           default=0.0),
    }


def by_model(rows: list[dict]) -> dict[str, dict]:
    g = defaultdict(list)
    for r in rows:
        g[r["model"]].append(r)
    return {k: _pool(v) for k, v in sorted(g.items())}


def by_model_severity(rows: list[dict]) -> dict[tuple[str, str], dict]:
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r.get("severity") or "unknown")].append(r)
    return {k: _pool(v) for k, v in sorted(
        g.items(), key=lambda kv: (kv[0][0], SEV_ORDER.get(kv[0][1], 99)))}


def by_model_prompt(rows: list[dict]) -> dict[tuple[str, str], dict]:
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r.get("prompt_type") or "unknown")].append(r)
    return {k: _pool(v) for k, v in sorted(g.items())}


def markdown(rows: list[dict]) -> str:
    out = []
    out.append("## Overall\n")
    out.append("| model | utts | WER | median RTF | peak RSS (MB) |")
    out.append("|---|---:|---:|---:|---:|")
    for m, s in by_model(rows).items():
        out.append(f"| {m} | {s['n']} | {s['wer']*100:.2f}% | "
                   f"{s['rtf_median']:.2f} | {s['peak_rss_mb']:.0f} |")

    out.append("\n## By dysarthria severity\n")
    out.append("_The table that matters. A single averaged WER hides the "
               "only question you are asking._\n")
    sevs = [s for s in SEVERITIES
            if any((r.get("severity") or "unknown") == s for r in rows)]
    out.append("| model | " + " | ".join(sevs) + " |")
    out.append("|---" * (len(sevs) + 1) + "|")
    ms = by_model_severity(rows)
    for m in sorted({k[0] for k in ms}):
        cells = []
        for s in sevs:
            v = ms.get((m, s))
            cells.append(f"{v['wer']*100:.2f}% (n={v['n']})" if v else "-")
        out.append(f"| {m} | " + " | ".join(cells) + " |")

    pts = sorted({(r.get("prompt_type") or "unknown") for r in rows})
    if len(pts) > 1:
        out.append("\n## By prompt type\n")
        out.append("| model | " + " | ".join(pts) + " |")
        out.append("|---" * (len(pts) + 1) + "|")
        mp = by_model_prompt(rows)
        for m in sorted({k[0] for k in mp}):
            cells = []
            for p in pts:
                v = mp.get((m, p))
                cells.append(f"{v['wer']*100:.2f}%" if v else "-")
            out.append(f"| {m} | " + " | ".join(cells) + " |")

    return "\n".join(out) + "\n"


def worst_utterances(rows: list[dict], model: str, n: int = 20) -> list[dict]:
    """Eyeball these before believing any number. Catches normalisation
    bugs, broken audio paths, and references that are not transcripts."""
    sel = [r for r in rows if r["model"] == model and int(r["ref_len"]) > 0]
    sel.sort(key=lambda r: float(r["wer"]), reverse=True)
    return sel[:n]
