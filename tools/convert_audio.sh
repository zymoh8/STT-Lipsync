#!/usr/bin/env bash
# Convert a corpus to 16 kHz mono 16-bit PCM, once, up front.
#
#   bash tools/convert_audio.sh /data/TORGO_raw /data/TORGO_16k
#
# Why this exists: the Vosk adapter rejects anything that isn't 16-bit
# mono, and resampling inside the benchmark loop would land in the timing
# numbers. Convert once, benchmark the converted tree, and RTF measures
# the model instead of ffmpeg.
#
# Requires ffmpeg. Originals are never modified.
set -euo pipefail

SRC="${1:?usage: convert_audio.sh SRC_DIR DEST_DIR}"
DST="${2:?usage: convert_audio.sh SRC_DIR DEST_DIR}"
RATE="${3:-16000}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }

count=0
skipped=0
while IFS= read -r -d '' f; do
  rel="${f#"$SRC"/}"
  out="$DST/$rel"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  ffmpeg -nostdin -loglevel error -y -i "$f" \
         -ac 1 -ar "$RATE" -sample_fmt s16 -c:a pcm_s16le "$out"
  count=$((count + 1))
  (( count % 200 == 0 )) && echo "  converted $count..."
done < <(find "$SRC" -type f \( -iname '*.wav' -o -iname '*.flac' \
                                -o -iname '*.mp3' -o -iname '*.m4a' \) -print0)

# Sidecar transcripts / prompt files travel with the audio.
while IFS= read -r -d '' f; do
  rel="${f#"$SRC"/}"
  out="$DST/$rel"
  mkdir -p "$(dirname "$out")"
  cp -n "$f" "$out" 2>/dev/null || true
done < <(find "$SRC" -type f -iname '*.txt' -print0)

echo "converted $count files, skipped $skipped already present -> $DST"
echo
echo "Verify one before you trust the tree:"
echo "  ffprobe -v error -show_entries stream=sample_rate,channels,sample_fmt \\"
echo "          -of default=nw=1 \"\$(find '$DST' -name '*.wav' | head -1)\""
echo "Expect: sample_rate=$RATE  channels=1  sample_fmt=s16"
