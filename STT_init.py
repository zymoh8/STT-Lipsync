"""Transcribe the MacBook microphone locally with Vosk. Stop with Ctrl+C."""

import argparse
import json
from pathlib import Path
import queue
import re
import sys

import sounddevice as sd
from vosk import KaldiRecognizer, Model


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "vosk-model-small-en-us-0.15"
VOICE_COMMANDS = re.compile(
    r"\b(begin\s+transcription|transcription\s+start|terminate\s+transcription)\b",
    re.IGNORECASE,
)
START_PROMPT = 'To begin transcription say "transcription start".'


def device_id(value):
    """Accept either a sounddevice index or a device name."""
    try:
        return int(value)
    except ValueError:
        return value


def clean_transcript(text):
    """Clean finalized text before display. Extend sentence cleanup here."""
    return " ".join(text.split())


def ranked_candidates(result):
    """Rank by mean word confidence when available, otherwise Vosk's score."""
    alternatives = result.get("alternatives", [result])
    alternatives = [item for item in alternatives if item.get("text", "").strip()]
    if not alternatives:
        return []
    have_word_confidence = all(
        item.get("result") and all("conf" in word for word in item["result"])
        for item in alternatives
    )

    def score(item):
        if have_word_confidence:
            words = item["result"]
            return sum(word["conf"] for word in words) / len(words)
        return item.get("confidence", float("-inf"))

    candidates = []
    seen = set()
    for item in sorted(alternatives, key=score, reverse=True):
        text = clean_transcript(item["text"])
        if text.casefold() not in seen:
            candidates.append(text)
            seen.add(text.casefold())
        if len(candidates) == 3:
            break
    return candidates


class TranscriptionSession:
    """Keep dictation separate from confirmation and selection replies."""

    def __init__(self):
        self.state = "paused"
        self.candidates = []
        self.speech_detected = False

    def detect_speech(self):
        if self.state == "listening" and not self.speech_detected:
            print("Speech detected. Transcribing...", flush=True)
            self.speech_detected = True

    def show_choices(self):
        print("Which interpretation did you mean?", flush=True)
        for number, text in enumerate(self.candidates, start=1):
            print(f'{number}. "{text}"', flush=True)
        labels = [f'"{word}"' for word in ("one", "two", "three")[:len(self.candidates)]]
        labels.append('"four"')
        print('4. Repeat your sentence\nSay ' + ", ".join(labels) + ".", flush=True)

    def accept(self, index):
        print(f'You said: "{clean_transcript(self.candidates[index])}"', flush=True)
        self.candidates = []
        self.state = "listening"
        print("Ready for your next sentence.", flush=True)

    def handle_text(self, candidates):
        if not candidates or self.state == "paused":
            return
        reply = re.sub(r"[^\w\s]", "", candidates[0]).lower().strip()
        if self.state == "listening":
            self.detect_speech()
            self.candidates = candidates
            self.state = "confirming"
            print(f'I heard: "{candidates[0]}"', flush=True)
            print('Is this what you said? Say "yes" or "no".', flush=True)
        elif self.state == "confirming":
            if reply in ("yes", "yes please", "correct", "that is correct"):
                self.accept(0)
            elif reply in ("no", "nope", "incorrect", "that is incorrect"):
                self.state = "choosing"
                self.show_choices()
            else:
                print('Please say "yes" or "no" to confirm the sentence.', flush=True)
        elif self.state == "choosing":
            selection = re.sub(r"^(?:option|number|choose|select)\s+", "", reply)
            choices = {"one": 0, "1": 0, "two": 1, "2": 1,
                       "three": 2, "3": 2, "four": 3, "4": 3,
                       "repeat": 3, "repeat sentence": 3, "repeat my sentence": 3}
            index = choices.get(selection)
            if index == 3:
                self.candidates = []
                self.state = "listening"
                print("Please repeat your sentence.", flush=True)
            elif index is not None and index < len(self.candidates):
                self.accept(index)
            else:
                print("That choice was not recognized or is unavailable.", flush=True)
                self.show_choices()

    def handle_result(self, result):
        candidates = ranked_candidates(result)
        if not candidates:
            self.speech_detected = False
            return
        parts = [VOICE_COMMANDS.split(text) for text in candidates]
        primary = parts[0]
        # Match command placement before borrowing alternatives for text segments.
        commands = [clean_transcript(part).lower() for part in primary[1::2]]
        compatible = [pieces for pieces in parts if
                      [clean_transcript(part).lower() for part in pieces[1::2]] == commands]
        for index, part in enumerate(primary):
            if index % 2:
                command = clean_transcript(part).lower()
                if command == "terminate transcription":
                    self.state = "paused"
                    self.candidates = []
                    print(f"Transcription paused. {START_PROMPT}", flush=True)
                elif self.state == "paused":
                    self.state = "listening"
                    print('Transcription enabled. Say "terminate transcription" to pause.',
                          flush=True)
                self.speech_detected = False
            else:
                texts = list(dict.fromkeys(clean_transcript(pieces[index])
                                           for pieces in compatible if pieces[index].strip()))
                self.handle_text(texts)
        self.speech_detected = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="path to an extracted Vosk model")
    parser.add_argument("--device", type=device_id,
                        help="input device name or index (default: MacBook microphone)")
    parser.add_argument("--list-devices", action="store_true",
                        help="list audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.model.is_dir():
        parser.error(f"Model directory not found: {args.model}")

    device = args.device
    if device is None:
        for index, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] > 0 and "macbook" in info["name"].lower():
                device = index
                break
        # On other computers, None selects the system's default input.
    info = sd.query_devices(device, "input")
    samplerate = int(info["default_samplerate"])

    print(f"Loading model: {args.model}", flush=True)
    model = Model(str(args.model))
    # Vosk handles resampling when told the actual incoming sample rate.
    recognizer = KaldiRecognizer(model, samplerate)
    recognizer.SetWords(True)
    recognizer.SetMaxAlternatives(3)
    session = TranscriptionSession()
    # Ten 100 ms chunks buffer up to one second of audio.
    audio_queue = queue.Queue(maxsize=10)
    dropped_chunks = 0
    reported_drops = 0

    def callback(indata, frames, time, status):
        nonlocal dropped_chunks
        if status:
            print(f"\nAudio warning: {status}", file=sys.stderr)
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            # Never block microphone capture while recognition catches up.
            dropped_chunks += 1

    def process(data):
        nonlocal reported_drops
        if dropped_chunks != reported_drops:
            reported_drops = dropped_chunks
            print("\nWarning: audio chunks are being dropped because transcription "
                  "is falling behind. Some of your speech may be missing. "
                  "Please slow down or pause and restart your sentence. "
                  f"({reported_drops} audio chunks dropped in total.)",
                  file=sys.stderr, flush=True)
        if recognizer.AcceptWaveform(data):
            session.handle_result(json.loads(recognizer.Result()))
        else:
            current = json.loads(recognizer.PartialResult()).get("partial", "")
            if current.strip():
                session.detect_speech()

    try:
        print(f"Opening {info['name']} ({samplerate} Hz). "
              "Allow microphone access if macOS asks.", flush=True)
        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=max(1, int(samplerate * 0.1)),
            dtype="int16",
            channels=1,
            device=device,
            callback=callback,
        ):
            print(f"Listening: {info['name']} ({samplerate} Hz). Speak; Ctrl+C to stop.",
                  flush=True)
            print(START_PROMPT, flush=True)
            while True:
                process(audio_queue.get())
    except KeyboardInterrupt:
        # The stream is closed before draining the remaining captured audio.
        while True:
            try:
                process(audio_queue.get_nowait())
            except queue.Empty:
                break
        session.handle_result(json.loads(recognizer.FinalResult()))
        if session.candidates:
            print("Stopped before confirmation. The pending sentence was not accepted.",
                  flush=True)
        print("Stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except (sd.PortAudioError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Check --list-devices and macOS microphone permissions for your terminal/app.",
              file=sys.stderr)
        sys.exit(1)
