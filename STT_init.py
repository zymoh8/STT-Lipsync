import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
import numpy as np

q = queue.Queue(maxsize=5)
def callback(indata, frames, time, status):
    if status:
        if "overflow" not in str(status):
            print(status)

    audio = np.frombuffer(indata, dtype=np.int16)
    audio = audio[::3] 

    if not q.full():
        q.put(audio.tobytes())

print("Loading model...")
model = Model(lang="en-us")

# Vosk expects 16000
rec = KaldiRecognizer(model, 16000)

with sd.RawInputStream(
    samplerate=48000,
    blocksize=2000,
    dtype='int16',
    channels=1,
    device=3,
    callback=callback
):
    print("🎤 Speak...")

    last_partial = ""

    while True:
        data = q.get()

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")

            if text:
                print("\n✅ You said:", text)

            last_partial = ""

        else:
            partial = json.loads(rec.PartialResult())
            current = partial.get("partial", "")

            if current != last_partial:
                print("\r🎤 " + current, end="", flush=True)
                last_partial = current
