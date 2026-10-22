"""Analyze a complete six-model TORGO run and build a visually reviewed PDF."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stt-mpl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_bench.runner import load_manifest
from stt_bench.reporting import analyze


LABELS = {"vosk-small": "Vosk Small", "vosk-lgraph": "Vosk Lgraph", "vosk-large": "Vosk Large",
          "whisper": "Whisper Small", "wav2vec2": "Wav2Vec2 Base", "parakeet": "Parakeet TDT v3"}


def pct(value):
    return "n/a" if value is None else f"{value * 100:,.1f}%"


def number(value, places=2):
    return "n/a" if value is None else f"{value:,.{places}f}"


def safe(text, limit=160):
    if any(c.isalpha() and ord(c) > 591 for c in text):
        text = "[Non-Latin text; complete prediction is in samples.csv]"
    text = text.replace("\n", " ")
    return escape(text[:limit] + ("..." if len(text) > limit else ""))


def create_report(report_path, manifest_path, output_path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from stt_bench.audio import load_audio

    report = json.loads(report_path.read_text())
    samples = load_manifest(manifest_path)
    analysis = analyze(report, samples)
    if len(report["models"]) != 6:
        raise ValueError("Expected all six model configurations")
    if report["run_config"]["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("Manifest changed since inference")
    run_dir = report_path.parent
    validation_path = run_dir / "validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    if validation and (validation["status"] != "passed" or validation["report_sha256"] != hashlib.sha256(report_path.read_bytes()).hexdigest()):
        raise ValueError("Validation is stale or failed; rerun validate_benchmark.py")
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    ranks = analysis["ranking_macro_speaker_wer"]
    models = analysis["models"]
    common = set(report["common_sample_ids"])
    if len(common) != analysis["common_samples"]:
        raise ValueError("Common-sample IDs differ from independently computed coverage")
    with (run_dir / "leaderboard.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "macro_speaker_wer", "corpus_wer", "corpus_cer", "mean_utterance_wer",
                         "exact_match_rate", "successful", "failed", "common_samples"])
        for name in ranks:
            m = models[name]
            c = m["common"]
            writer.writerow([name, m["macro_speaker_wer"], c["wer"], c["cer"], c.get("mean_utterance_wer"),
                             c["exact_match_rate"], m["successful"], m["failed"], c["samples"]])
    with (run_dir / "per-speaker.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["speaker", "model", "samples", "wer", "cer", "exact_match_rate"])
        for speaker in analysis["speakers"]:
            for name in ranks:
                m = models[name]["per_speaker"][speaker]
                writer.writerow([speaker, name, m["samples"], m["wer"], m["cer"], m["exact_match_rate"]])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets = Path("tmp/pdfs")
    assets.mkdir(parents=True, exist_ok=True)
    for name, filename in (("Report", "Arial.ttf"), ("ReportBold", "Arial Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(Path("/System/Library/Fonts/Supplemental") / filename)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold", italic="Report", boldItalic="ReportBold")
    navy = colors.HexColor("#142D4E")
    teal = colors.HexColor("#087F8C")
    muted = colors.HexColor("#536273")
    pale = colors.HexColor("#EAF3F5")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="ReportBold", fontSize=27, leading=31, textColor=navy, spaceAfter=14))
    styles.add(ParagraphStyle("ReportH1", fontName="ReportBold", fontSize=19, leading=23, textColor=navy, spaceAfter=12))
    styles.add(ParagraphStyle("ReportH2", fontName="ReportBold", fontSize=11, leading=15, textColor=teal, spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle("ReportBody", fontName="Report", fontSize=9.3, leading=14, textColor=navy, spaceAfter=9))
    styles.add(ParagraphStyle("ReportSmall", fontName="Report", fontSize=7.7, leading=10.7, textColor=muted, spaceAfter=7))
    styles.add(ParagraphStyle("ReportCell", fontName="Report", fontSize=8, leading=11, textColor=navy))
    styles.add(ParagraphStyle("ReportHeader", fontName="ReportBold", fontSize=7.5, leading=10, textColor=colors.white))
    story = []
    width = A4[0] - 92
    def p(text, style="ReportBody"):
        return Paragraph(text, styles[style])
    def add(text, style="ReportBody"):
        story.append(p(text, style))
    def heading(title, subtitle=None):
        add(title, "ReportH1")
        if subtitle:
            add(subtitle)
    def table(headers, rows, widths=None, font=8):
        cells = [[p(escape(str(c)), "ReportHeader") for c in headers]]
        cells += [[p(str(c), "ReportCell") for c in row] for row in rows]
        t = Table(cells, colWidths=widths or [width / len(headers)] * len(headers), repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, 0), .5, navy),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))
    def page():
        story.append(PageBreak())
    now = datetime.now(ZoneInfo("America/New_York"))
    winner = ranks[0]
    word_winner = max(ranks, key=lambda n: models[n]["single_word"]["exact_match_rate"])
    exact_winner = max(ranks, key=lambda n: models[n]["common"]["exact_match_rate"])
    total_failures = sum(m["failed"] for m in models.values())
    representative = {r["sample_id"]: r for r in report["results"] if r["status"] == "ok"}
    duration_by_id = {sample["id"]: (representative[sample["id"]]["duration_seconds"]
                       if sample["id"] in representative else load_audio(sample["audio"]).duration_seconds)
                      for sample in samples}
    total_audio = sum(duration_by_id.values())
    reference_words = sum(len(representative[sample["id"]]["normalized_reference"].split())
                          for sample in samples if sample["id"] in representative)
    # Page 1: decision-oriented overview.
    add("FULL-CORPUS BENCHMARK / TORGO", "ReportH2")
    add("Recognizing male<br/>dysarthric speech", "ReportTitle")
    add(f"Six downloaded models | {len(samples):,} recordings | {len(analysis['speakers'])} speakers<br/>"
        f"Prepared {now.strftime('%B %d, %Y at %I:%M %p %Z')}", "ReportSmall")
    add(f"<b>{LABELS[winner]}</b> has the lowest equal-weight speaker WER in this run: "
        f"<b>{pct(models[winner]['macro_speaker_wer'])}</b>. Its corpus WER is "
        f"<b>{pct(models[winner]['common']['wer'])}</b>. Lower WER/CER is better; higher exact-match rate is better.")
    table(["Rank / model", "Speaker mean WER", "Corpus WER", "Corpus CER", "Exact match"], [
        [f"{index}. {LABELS[name]}", pct(models[name]["macro_speaker_wer"]),
         pct(models[name]["common"]["wer"]), pct(models[name]["common"]["cer"]),
         pct(models[name]["common"]["exact_match_rate"])]
        for index, name in enumerate(ranks, 1)], [150, 88, 84, 84, width - 406])
    add(f"<b>Coverage:</b> {analysis['attempts']:,} model-recording attempts; "
        f"{total_failures:,} runtime failures; {analysis['common_samples']:,} recordings completed by every model.")
    add("<b>Primary ranking:</b> calculate corpus WER within each speaker, then average the five speaker WERs "
        "equally. This prevents speakers with more recordings from dominating the primary score.")
    add(f"<b>Task trade-off:</b> {LABELS[word_winner]} has the highest single-word exact-match rate "
        f"({pct(models[word_winner]['single_word']['exact_match_rate'])}). {LABELS[exact_winner]} has the highest "
        f"whole-clip exact-match rate ({pct(models[exact_winner]['common']['exact_match_rate'])}), but its "
        "extra and repetitive output on other clips drives substantially higher WER. The primary ranking "
        "measures error burden, not just the number of perfectly matched clips.")
    add("<b>Interpretation:</b> this is agreement with TORGO's supplied prompts on the selected head-microphone "
        "cohort. It is not clinical validation, a guarantee for a new speaker, or a measure of meaning understood. "
        "No model was tuned or fine-tuned for this run.")
    add("WER can exceed 100% when extra words are predicted. It must not be converted into a bounded accuracy "
        "percentage by subtracting it from 100%.", "ReportSmall")
    page()

    # Page 2: corpus and design.
    heading("What was evaluated", "All accepted recordings in the existing TORGO head-microphone manifest were evaluated once per model.")
    table(["Speaker", "Recordings", "Audio (min)", "Reference words"], [
        [speaker, str(analysis["samples_by_speaker"][speaker]),
         number(sum(duration_by_id[s["id"]] for s in samples if s.get("speaker") == speaker) / 60, 1),
         str(sum(len(representative[s["id"]]["normalized_reference"].split()) for s in samples
                 if s.get("speaker") == speaker and s["id"] in representative))]
        for speaker in analysis["speakers"]], [100, 115, 130, width - 345])
    add(f"Total distinct audio: <b>{total_audio / 3600:.2f} hours</b>; approximately "
        f"<b>{reference_words:,} reference words</b>. Speaker IDs M01-M05 are the male dysarthric cohort; controls are excluded.")
    add("Import exclusions", "ReportH2")
    audit_path = manifest_path.with_suffix(".audit.json")
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        reasons = Counter(row["reason"] for row in audit["skipped"])
        table(["Exclusion / session issue", "Count"], [[escape(reason), str(count)] for reason, count in reasons.items()], [width - 60, 60])
    add("Only head-microphone audio was selected. M04/Session1 has no head-microphone folder and was not "
        "substituted with array-microphone audio. Two microphone recordings of the same utterance were not double-counted.", "ReportSmall")
    add("Procedure and fairness", "ReportH2")
    add("Every model received the same decoded 16 kHz, mono, signed 16-bit PCM per sample. PCM SHA-256 values "
        "were checked across successful model outputs. All models attempted every manifest ID. Rankings use only "
        "recordings successfully completed by all six models, with failures and coverage reported separately.")
    if validation:
        add(f"Independent QA: RapidFuzz {validation['rapidfuzz_version']} verified "
            f"{validation['independent_edit_distance_checks']:,} word/character edit totals with zero mismatches. "
            f"All {validation['original_pcm_files_verified']:,} original PCM files were rehashed and matched the saved inputs.", "ReportSmall")
    add("This corpus includes single-word and multi-word stimuli. It contains no dedicated silence benchmark in this manifest. "
        "Prompt matching and basic exclusions were automated; references have not been exhaustively verified against the speech.", "ReportSmall")
    page()

    # Page 3: between-speaker variation.
    heading("Performance by speaker", "Corpus WER within each speaker. The same common-success recordings are used for every model.")
    matrix = np.array([[models[name]["per_speaker"][speaker]["wer"] for speaker in analysis["speakers"]] for name in ranks], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.imshow(np.clip(matrix, 0, 1.5), cmap="YlOrRd", vmin=0, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(analysis["speakers"])), analysis["speakers"])
    ax.set_yticks(range(len(ranks)), [LABELS[n] for n in ranks])
    ax.tick_params(length=0, labelsize=10)
    for i in range(len(ranks)):
        for j in range(len(analysis["speakers"])):
            ax.text(j, i, pct(matrix[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if matrix[i, j] > 1 else "#142D4E", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=1.1)
    chart = assets / "speaker-wer.png"
    fig.savefig(chart, dpi=190, facecolor="white")
    plt.close(fig)
    story.append(Image(str(chart), width=width, height=width * 4.1 / 8.2))
    add("Color intensity is capped at 150% WER; cell labels show the full, unbounded scores.", "ReportSmall")
    winners = []
    for speaker in analysis["speakers"]:
        ordered = sorted(ranks, key=lambda n: models[n]["per_speaker"][speaker]["wer"])
        best = ordered[0]
        tied = [n for n in ordered if abs(models[n]["per_speaker"][speaker]["wer"] - models[best]["per_speaker"][speaker]["wer"]) < 1e-12]
        winners.append([speaker, " / ".join(LABELS[n] for n in tied), pct(models[best]["per_speaker"][speaker]["wer"]),
                        str(models[best]["per_speaker"][speaker]["samples"])])
    table(["Speaker", "Lowest WER model", "WER", "Common clips"], winners, [75, 210, 100, width - 385])
    add("Use these rows when choosing a model for a particular individual. An overall winner need not win for "
        "every speaker. Do not infer impairment severity from recognition errors: severity labels were not independently assessed here.")
    page()

    # Page 4: task lengths and averaging choices.
    heading("Words, longer utterances, and averages")
    table(["Model", "Word clips", "Word exact match", "Multi-word clips", "Multi-word WER", "Multi-word exact"], [
        [LABELS[n], str(models[n]["single_word"]["samples"]), pct(models[n]["single_word"]["exact_match_rate"]),
         str(models[n]["multi_word"]["samples"]), pct(models[n]["multi_word"]["wer"]),
         pct(models[n]["multi_word"]["exact_match_rate"])] for n in ranks],
        [124, 57, 83, 66, 85, width - 415])
    add("A single-word clip has one whitespace-delimited token after normalization; multi-word means two or more. "
        "Multi-word clips include sentences and passage fragments. These groups are derived from reference text, not manually validated task labels.")
    add("Three ways to average errors", "ReportH2")
    table(["Model", "Corpus WER", "Mean speaker WER", "Mean clip WER", "Median clip WER"], [
        [LABELS[n], pct(models[n]["common"]["wer"]), pct(models[n]["macro_speaker_wer"]),
         pct(models[n]["common"].get("mean_utterance_wer")), pct(models[n]["common"].get("median_utterance_wer"))]
        for n in ranks], [134, 85, 100, 90, width - 409])
    add("<b>Corpus WER:</b> total word edits divided by total reference words. Longer references receive more weight.<br/>"
        "<b>Mean speaker WER:</b> the mean of speaker-level corpus WERs. Each speaker receives equal weight.<br/>"
        "<b>Mean clip WER:</b> the mean of per-recording WERs. A one-word recording and a long sentence receive equal weight.<br/>"
        "<b>Exact match:</b> normalized prediction equals normalized reference in full. This is a bounded percentage of complete matches.")
    add("The corpus and speaker-level averages answer different questions. Rank by the speaker mean for broad "
        "cohort balance, then inspect the individual and task-length results before choosing a model.")
    page()

    # Page 5: errors, including expansion diagnostics.
    heading("How the models fail", "Word edit components below are divided by the common subset's reference word count. S + D + I equals corpus WER.")
    error_rows = []
    for n in ranks:
        c = models[n]["common"]
        w = c["word"]
        denom = w["reference_units"] or 1
        error_rows.append([LABELS[n], pct(w["substitutions"] / denom), pct(w["deletions"] / denom),
                           pct(w["insertions"] / denom), pct(c["wer"])])
    table(["Model", "Substitutions", "Deletions", "Insertions", "Total WER"], error_rows,
          [144, 94, 88, 88, width - 414])
    add("A substitution replaces a reference word; a deletion omits it; an insertion adds a word. A model that "
        "produces fluent extra text can have substantially worse WER than a model that returns only a few words.")
    table(["Model", "Empty outputs", "Expansion flags", "Non-Latin flags"], [
        [LABELS[n], str(models[n]["common"]["empty_predictions"]),
         str(models[n]["common"]["expansion_flags"]), str(models[n]["common"]["non_latin_flags"])]
        for n in ranks], [180, 100, 110, width - 390])
    add("<b>Empty output:</b> the normalized prediction has no text.<br/>"
        "<b>Expansion flag:</b> predicted word count is at least three times the reference count and at least ten words.<br/>"
        "<b>Non-Latin flag:</b> at least one alphabetic character has a non-Latin Unicode name.")
    add("These are review flags, not proven hallucinations or automatic failure labels. They may overlap and do not "
        "measure factual correctness. Raw predictions are preserved in samples.csv, including non-Latin and repeated output.", "ReportSmall")
    whisper_flags = models["whisper"]["common"]["non_latin_flags"]
    parakeet_flags = models["parakeet"]["common"]["non_latin_flags"]
    add(f"The multilingual models produced non-Latin flags on {parakeet_flags:,} Parakeet outputs and "
        f"{whisper_flags:,} Whisper outputs. Whisper used automatic language detection, and Parakeet also "
        "selects output language automatically. A separate language-controlled experiment could help distinguish "
        "language-selection failures from recognition failures; it was not substituted into this baseline.")
    page()

    # Page 6: latency and reliability.
    heading("Runtime and reliability", "Timing is descriptive: model workers ran concurrently on this computer, not under isolated laboratory conditions.")
    table(["Model", "Inference min", "Median sec", "P95 sec", "Overall RTF", "P95 RTF"], [
        [LABELS[n], number(models[n]["all_successful"]["transcription_seconds"] / 60, 1),
         number(models[n]["all_successful"]["median_transcription_seconds"]),
         number(models[n]["all_successful"]["p95_transcription_seconds"]),
         number(models[n]["all_successful"]["real_time_factor"]),
         number(models[n]["all_successful"]["p95_real_time_factor"])]
        for n in ranks], [134, 84, 73, 73, 76, width - 440])
    add("RTF = inference seconds / recording seconds. An overall RTF below 1 means aggregate processing was "
        "faster than audio playback in this run; it does not establish streaming response latency. P95 is the "
        "95th percentile across successful recordings. Inference totals across concurrent workers must not be "
        "summed and described as elapsed wall-clock time. Timing statistics use successful attempts only; "
        "unfinished calls interrupted during checkpoint rescheduling are excluded.")
    table(["Model", "Completed / attempted", "Coverage", "First load sec"], [
        [LABELS[n], f"{models[n]['successful']:,} / {models[n]['attempted']:,}", pct(models[n]["coverage"]),
         number(report["models"][n].get("initialization_seconds"))]
        for n in ranks], [160, 160, 90, width - 410])
    if total_failures:
        affected = "; ".join(f"{LABELS[n]}: {models[n]['failed']}" for n in ranks if models[n]["failed"])
        messages = Counter(item["error"] for n in ranks for item in models[n]["errors"])
        message, count = messages.most_common(1)[0]
        add(f"<b>Recorded runtime failures:</b> {escape(affected)}. Most common error ({count} cases): "
            f"{safe(message, 200)}. Exact sample IDs and all error messages are retained in analysis.json.", "ReportSmall")
    add("What timing includes", "ReportH2")
    add("The timed region is adapter.transcribe(): waveform preparation inside the adapter, feature extraction, "
        "decoding, and transcript production. WAV decoding, score calculation, checkpoint writes, and initial weight loading "
        "are excluded. Vosk recognizer creation is included per recording. Load time is reported separately.")
    cfg = report["run_config"]
    hardware_path = run_dir / "hardware.json"
    hardware = json.loads(hardware_path.read_text()) if hardware_path.exists() else {}
    cpu_name = escape(hardware.get("machdep.cpu.brand_string", cfg["machine"]))
    add(f"Environment: {escape(cfg['platform'])}; {cfg['logical_cpus']} logical CPUs; CPU inference. "
        f"Default PyTorch intra-op threads: {cfg['torch_threads']}; inter-op threads: {cfg['torch_interop_threads']}. "
        "Whisper and Parakeet partition workers used one intra-op thread each; Vosk Lgraph used four independent recognizers. "
        f"Processor: {cpu_name}; 48 GiB RAM. Transformer adapters use float32. "
        "No explicit warm-up or repeat timing trials were performed.", "ReportSmall")
    add(f"Python {escape(cfg['python'])}; Vosk {escape(report['models']['vosk-small'].get('vosk_version', 'unknown'))}; "
        f"Transformers {escape(report['models']['whisper'].get('transformers_version', 'unknown'))}; "
        f"PyTorch {escape(report['models']['whisper'].get('torch_version', 'unknown'))}.", "ReportSmall")
    add("Workers, brief diagnostic runs, and report preparation shared system resources. The first Vosk Small checkpoint rows were "
        "measured before worker concurrency began. Timing differences are not a controlled speed ranking. "
        "Load times show the first saved process initialization, not the sum across partition workers. "
        "Use a repeated, isolated deployment benchmark for a speed-critical decision.", "ReportSmall")
    page()

    # Page 7: transparent sample-level evidence.
    heading("Example predictions", "Two deterministically selected examples illustrate variation. They are not a representative accuracy sample.")
    by_id = {}
    for row in report["results"]:
        if row["status"] == "ok" and row["sample_id"] in common:
            by_id.setdefault(row["sample_id"], {})[row["model"]] = row
    choices = []
    for predicate in (lambda row: len(row["normalized_reference"].split()) == 1,
                      lambda row: 3 <= len(row["normalized_reference"].split()) <= 10):
        for sample_id in sorted(by_id):
            first = next(iter(by_id[sample_id].values()))
            if predicate(first):
                choices.append(sample_id)
                break
    for sample_id in choices:
        first = next(iter(by_id[sample_id].values()))
        add(escape(sample_id), "ReportH2")
        add(f"<b>Supplied reference:</b> {safe(first['reference'], 200)}")
        table(["Model", "Predicted text", "WER"], [
            [LABELS[n], safe(by_id[sample_id][n]["hypothesis"], 125) or "[empty output]",
             pct(by_id[sample_id][n]["word"]["rate"])] for n in ranks], [122, width - 185, 63])
    add("Examples are the first eligible single-word and 3-10-word common-success clips by sorted sample ID. "
        "Long predictions are shortened for display; predictions containing non-Latin scripts are replaced with a label "
        "in this PDF only. Complete text and scores remain in the CSV and JSON outputs.", "ReportSmall")
    page()

    # Page 8: duration/reference anomalies are retained in the primary benchmark.
    heading("Data-quality sensitivity check")
    long_ids = analysis["duration_outlier_ids"]
    short_ids = analysis["short_audio_ids"]
    add(f"The benchmark contains <b>{len(long_ids)} recording(s) longer than 30 seconds</b>. "
        "All are retained in the primary scores. This secondary diagnostic removes only those IDs from the "
        "common-success subset; no predictions or reference words are edited. "
        f"A further <b>{len(short_ids)} recordings are shorter than 100 milliseconds</b>; several are paired with "
        "full-sentence prompts. These are flagged for audio/reference review, not silently removed.")
    if long_ids:
        table(["Recording", "Seconds", "Supplied reference"], [
            [escape(sample_id), number(duration_by_id[sample_id]), safe(representative[sample_id]["reference"], 90)]
            for sample_id in long_ids], [255, 65, width - 320])
        table(["Model", "Full corpus WER", "Remove >30s", "Remove all duration flags"], [
            [LABELS[n], pct(models[n]["common"]["wer"]), pct(models[n]["without_duration_outliers"]["wer"]),
             pct(models[n]["without_duration_flags"]["wer"])]
            for n in ranks], [145, 112, 135, width - 392])
        add(f"Comparable clips: full {models[winner]['common']['samples']:,}; without long clips "
            f"{models[winner]['without_duration_outliers']['samples']:,}; without either duration flag "
            f"{models[winner]['without_duration_flags']['samples']:,}.", "ReportSmall")
        add("The 179.94-second M01 recording is paired with the one-word prompt 'hill', while the next-longest "
            "accepted recording is about 23 seconds. Duration alone does not prove the recording is invalid. "
            "Listen to this file and verify prompt alignment before using a cleaned dataset for a separate evaluation.")
    if short_ids:
        source_by_id = {s["id"]: s for s in samples}
        table(["Short-file example", "Millisec", "Supplied reference"], [
            [escape(sample_id), number(1000 * duration_by_id[sample_id], 1), safe(source_by_id[sample_id]["reference"], 80)]
            for sample_id in short_ids[:3]], [240, 64, width - 304])
        add("All short-file IDs are listed in analysis.json. Clips that cause any model to fail are already excluded "
            "from every model's common-success accuracy scores; they remain in coverage and failure counts.", "ReportSmall")
    add("Why this matters", "ReportH2")
    add("A long recording paired with a very short reference can accumulate many insertion errors. Word error "
        "rates are unbounded, so a single problematic clip can noticeably affect pooled and speaker-level scores. "
        "The full-corpus ranking remains the primary result; the diagnostic is clearly separated to avoid silently "
        "discarding a difficult example after observing model behavior.")
    add("To improve the next benchmark, review reference/audio alignment and recording notes, correct references "
        "independently of model predictions, and freeze a new manifest before rerunning every model. Keep original "
        "and cleaned results as separate experiments.")
    page()

    # Page 9: implications, limitations, and provenance.
    heading("How to use these results")
    add(f"<b>First candidate for further validation:</b> {LABELS[winner]}, based on the lowest equal-weight speaker "
        "WER in this fixed cohort. Inspect speaker-level results rather than assuming the same model will be best for "
        "a new individual. Test the chosen configuration on fresh recordings from the intended user before selecting it for deployment.")
    add(f"<b>For isolated words:</b> start by comparing {LABELS[word_winner]}, the single-word exact-match leader "
        f"at {pct(models[word_winner]['single_word']['exact_match_rate'])}, with the speaker-specific alternatives. "
        "The overall WER leader and the word-match leader answer different use cases. Substantial errors remain "
        "in every evaluated configuration.")
    add("Limits that affect the conclusion", "ReportH2")
    add("1. <b>References:</b> TORGO supplies intended prompts. Omissions, restarts, interviewer speech, or deviations "
        "from the prompt can affect scores. The references were not exhaustively manually corrected.<br/>"
        "2. <b>Cohort:</b> five male speakers, one microphone type, and the imported sessions. No female, control, "
        "new-speaker, background-noise, or silence generalization claim follows from this run.<br/>"
        "3. <b>Normalization:</b> Unicode NFKC, case folding, punctuation-to-space replacement, and whitespace collapse; "
        "apostrophes are preserved. Written numbers are not converted to spoken-number forms; contractions are not expanded. "
        "CER includes spaces. Thus correct-looking alternate text formatting can still incur errors.<br/>"
        "4. <b>Task:</b> final transcripts were compared, not semantic meaning, speaker attribution, streaming stability, "
        "word-timestamp accuracy, or lipsync alignment.<br/>"
        "5. <b>Inference:</b> one fixed decoding configuration per model; no language-model tuning, fine-tuning, or "
        "hyperparameter search. Public-corpus overlap with pretraining data is unknown. No statistical significance "
        "or population-level confidence interval is claimed from five speakers.")
    add("Decoding configurations", "ReportH2")
    add("Vosk Small: en-us-0.15; Vosk Lgraph: en-us-0.22-lgraph; Vosk Large: en-us-0.22. "
        "All use default unrestricted recognizers and a fresh recognizer per clip. Wav2Vec2: facebook/wav2vec2-base-960h, "
        "greedy CTC without an external language model, 30-second chunks with 5-second stride. Parakeet: "
        "nvidia/parakeet-tdt-0.6b-v3, default generation over each recording. Whisper: openai/whisper-small, "
        "transcription task with automatic language detection, timestamp-enabled generation, five beams, "
        "and max_length=448 from the loaded model/pipeline configuration.", "ReportSmall")
    add("Reproducibility and supporting files", "ReportH2")
    manifest_hash = report["run_config"]["manifest_sha256"]
    report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    add(f"Manifest SHA-256: {manifest_hash}<br/>Raw report SHA-256: {report_hash}", "ReportSmall")
    add(f"Run directory: {escape(str(run_dir))}<br/>"
        "report.json and samples.csv preserve all predictions; analysis.json, leaderboard.csv, and per-speaker.csv "
        "contain calculated summaries. Per-model JSONL checkpoints, metadata JSON, logs, run-config.json, and "
        "execution.json document completion and environment details. Rebuild this report with "
        "scripts/make_benchmark_report.py and the same raw report and manifest.", "ReportSmall")
    add("Dataset documentation and required publication citation: "
        "<link href='https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html' color='#087F8C'>"
        "University of Toronto TORGO database</link>. "
        "Rudzicz, Namasivayam and Wolff (2012), The TORGO database of acoustic and articulatory speech from speakers "
        "with dysarthria, Language Resources and Evaluation 46(4), 523-541. Academic-use terms apply.", "ReportSmall")
    page()

    # Page 10: published specifications, local storage, and deployment judgment.
    heading("Model specifications and Raspberry Pi fit")
    add("Reference device: <b>Raspberry Pi 5, 8 GB RAM</b>, 64-bit OS, active cooling, CPU inference, "
        "one model at a time. Pi assessments below are engineering judgments, not measured Pi results.", "ReportSmall")
    table(["Downloaded model", "Parameters", "Local files", "Memory guidance", "Pi 5 / 8 GB assessment"], [
        ["<b>Vosk Small</b><br/>en-us-0.15 [1]", "Not<br/>published", "68 MiB",
         "About <b>300 MB runtime</b> is Vosk's typical small-model guidance; not a measurement of this run.",
         "<b>Strongest starting point.</b> Officially targets Raspberry Pi. Validate latency with the lipsync app running."],
        ["<b>Vosk Lgraph</b><br/>en-us-0.22-lgraph [1]", "Not<br/>published", "204 MiB",
         "Model-specific runtime RAM is <b>not published</b>. Dynamic decoding graph; storage size does not predict peak RAM.",
         "<b>Candidate to test.</b> Compact files make it worth a Pi trial; measure peak RAM and latency before committing."],
        ["<b>Vosk Large</b><br/>en-us-0.22 [1]", "Not<br/>published", "2.67 GiB",
         "Vosk's big-model category can use <b>up to 16 GB RAM</b>. This is not a measured requirement for en-us-0.22.",
         "<b>Poor default choice.</b> 8 GB may be insufficient. Even a 16 GB Pi needs a memory/latency trial and OS headroom."],
        ["<b>Whisper Small</b><br/>openai/whisper-small [2]", "244 M", "926 MiB",
         "FP32 weights alone: <b>~0.98 GB</b>. Total runtime RAM is higher. Alternate whisper.cpp reports ~852 MB for small [5].",
         "<b>Conditional.</b> Try whisper.cpp, which supports Pi and quantization. Current FP32, five-beam Python decoding needs testing."],
        ["<b>Wav2Vec2 Base</b><br/>wav2vec2-base-960h [3]", "94.4 M", "360 MiB",
         "FP32 weights alone: <b>~0.38 GB</b>. Framework and audio activations add RAM; no total-RAM figure is published on the card.",
         "<b>Plausible candidate.</b> Smallest neural model here; keep chunking. ARM dependency support and speed must be checked."],
        ["<b>Parakeet TDT v3</b><br/>0.6b-v3 [4]", "~600 M", "2.34 GiB",
         "FP32 weights alone: <b>~2.4 GB</b>. Card says at least 2 GB to load (configuration dependent); total FP32 RAM is higher.",
         "<b>Experimental.</b> Loading short clips may fit, but highest neural weight cost here. Optimized for NVIDIA GPUs; Pi has no CUDA."],
    ], [104, 64, 52, 136, width - 356])
    add("How to read the memory figures", "ReportH2")
    add("<b>Peak RAM was not measured in this benchmark.</b> FP32 figures are calculated as published parameter count "
        "x 4 bytes, rounded; they exclude Python/PyTorch, activations, decoder caches, audio buffers, loading copies, "
        "and the OS. Longer audio, beams, or concurrent models can raise memory use. They are lower bounds, "
        "not safe total-RAM budgets. The whisper.cpp figure uses a different runtime/representation from this benchmark.", "ReportSmall")
    add("Local files are the rounded sum of file bytes in each downloaded directory, inspected October 20, 2026 "
        "(MiB/GiB are binary; MB/GB are decimal). They are neither ZIP sizes nor runtime RAM. "
        "Vosk's cards do not give parameter counts; graph and language-model files also consume storage and memory.", "ReportSmall")
    add("Deployment decision", "ReportH2")
    add("Start with Vosk Small for a Pi prototype, then compare Lgraph and Wav2Vec2 on the target speaker. "
        "Parakeet leads this cohort's WER, but needs a separate Pi feasibility trial. Loading successfully does not establish "
        "real-time performance. Measure peak resident memory, sustained latency, and thermal behavior on the actual board. "
        "Recheck WER after changing runtime, quantization, chunking, or decoding.", "ReportSmall")
    add("Sources (accessed October 20, 2026)", "ReportH2")
    add("[1] <link href='https://alphacephei.com/vosk/models' color='#087F8C'>Vosk model catalog / runtime guidance</link>; "
        "[2] <link href='https://huggingface.co/openai/whisper-small' color='#087F8C'>OpenAI Whisper Small</link>; "
        "[3] <link href='https://huggingface.co/facebook/wav2vec2-base-960h' color='#087F8C'>Meta Wav2Vec2 Base 960h</link>; "
        "[4] <link href='https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3' color='#087F8C'>NVIDIA Parakeet TDT 0.6b v3</link>; "
        "[5] <link href='https://github.com/ggml-org/whisper.cpp' color='#087F8C'>whisper.cpp memory / Pi support</link>; "
        "[6] <link href='https://www.raspberrypi.com/products/raspberry-pi-5/' color='#087F8C'>Raspberry Pi 5 specifications</link> "
        "(quad-core 2.4 GHz Arm Cortex-A76, RAM options up to 16 GB).", "ReportSmall")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D5DEE7"))
        canvas.line(46, 41, A4[0] - 46, 41)
        canvas.setFont("Report", 7)
        canvas.setFillColor(muted)
        canvas.drawString(46, 28, "STT-Lipsync | TORGO six-model baseline | Lower WER/CER is better")
        canvas.drawRightString(A4[0] - 46, 28, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=46, leftMargin=46,
                            topMargin=43, bottomMargin=55, title="TORGO Six-Model Speech Recognition Benchmark",
                            author="STT-Lipsync benchmark", allowSplitting=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"pdf": str(output_path), "models": len(models), "attempts": analysis["attempts"],
                      "common_samples": analysis["common_samples"], "winner": winner,
                      "macro_speaker_wer": models[winner]["macro_speaker_wer"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/torgo-manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/torgo-six-model-benchmark.pdf"))
    args = parser.parse_args()
    create_report(args.report, args.manifest, args.output)
