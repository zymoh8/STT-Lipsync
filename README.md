# Voice-controlled speech transcription

`STT_init.py` captures microphone audio and transcribes English speech locally with
Vosk. Voice commands enable and pause transcription output. After each utterance,
the program asks you to confirm its interpretation before accepting the text.

The program is being developed for accessibility use, including speakers with
dysarthria or unclear speech. Recognition accuracy and voice-command reliability
depend on the speaker, microphone, and environment. The intended deployment
target is a Raspberry Pi with 2 GB of RAM. Performance still needs to be evaluated
on that device.

## Setup

You need a microphone, a Python environment with `vosk` and `sounddevice`, and an
extracted English Vosk model. Development has used Python 3.11 and Vosk 0.3.44.

Run the following commands from the repository folder.

### Use the existing Conda environment

On the development MacBook, the `STT` environment already contains the required
packages.

```bash
conda activate STT
python STT_init.py
```

### Set up a new environment

Using an installed Python 3.11 interpreter, create an environment and install the
dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install vosk sounddevice
```

The default model directory is located beside the script.

```text
STT-Lipsync/
├── STT_init.py
└── models/
    └── vosk-model-small-en-us-0.15/
        ├── am/
        ├── conf/
        ├── graph/
        └── ivector/
```

The development workspace already has this model. For a fresh installation,
download and extract `vosk-model-small-en-us-0.15` from the
[Vosk model catalog](https://alphacephei.com/vosk/models).

Then launch the program.

```bash
python STT_init.py
```

On macOS, allow microphone access for the terminal or application running Python.
Permissions are managed under **System Settings → Privacy & Security → Microphone**.

## Use the program

Speak each command or response, pause briefly, and wait for the next prompt.
Vosk finalizes utterances after a pause in speech, which also determines when
commands and confirmation responses take effect.

1. **Enable transcription.** At startup, the program prints
   `To begin transcription say "transcription start".` Say **“transcription start”**
   or **“begin transcription.”**
2. **Speak an utterance.** The program prints `Speech detected. Transcribing...`
   after Vosk recognizes the first words. Pause when finished. It displays the
   highest-ranked interpretation and asks `Is this what you said? Say "yes" or "no".`
3. **Confirm or review.** Say **“yes”** to accept the displayed text. Say **“no”** to
   see up to three distinct interpretations, including the original suggestion.
4. **Choose or repeat.** Say **“one,” “two,”** or **“three”** to
   accept a listed interpretation. Say **“four”** or **“repeat”** to discard
   the pending interpretations and speak the sentence again. Vosk may return fewer
   than three interpretations. Choose from the displayed options.
5. **Continue.** After acceptance, the program prints `You said: "..."` and
   `Ready for your next sentence.` It then waits for your next utterance.

While a confirmation or selection is pending, speech is interpreted as a response
to that prompt. Wait for `Ready for your next sentence.` before dictating again.

### Voice commands

| Say | Action |
| --- | --- |
| “Transcription start” or “begin transcription” | Enable transcription output. |
| “Terminate transcription” | Pause output and discard any pending interpretation. |
| “Yes” | Accept the first interpretation when asked for confirmation. |
| “No” | Display the available interpretations when asked for confirmation. |
| “One,” “two,” or “three” | Accept the corresponding displayed choice. |
| “Four” or “repeat” | Request another attempt from the choice menu. |

Start and pause phrases are reserved commands, including when recognized inside
a longer utterance. The microphone and Vosk keep processing audio while output
is paused so the program can hear the next start command. Status messages and
audio warnings can still appear while paused.

Press **Ctrl+C** to exit. The program closes the audio stream and processes the
remaining buffered audio. Any interpretation still awaiting confirmation is
reported as unaccepted.

### Example interaction

This example assumes Vosk returns three distinct interpretations. Lines beginning
with `Speaker` represent spoken input.

```text
To begin transcription say "transcription start".
Speaker: transcription start
Transcription enabled. Say "terminate transcription" to pause.
Speaker: I need water
Speech detected. Transcribing...
I heard: "i feed water"
Is this what you said? Say "yes" or "no".
Speaker: no
Which interpretation did you mean?
1. "i feed water"
2. "i need water"
3. "i need more water"
4. Repeat your sentence
Say "one", "two", "three", "four".
Speaker: two
You said: "i need water"
Ready for your next sentence.
Speaker: terminate transcription
Transcription paused. To begin transcription say "transcription start".
```

## Microphone and model options

The script automatically selects an input device with “MacBook” in its name.
On other computers, it uses the system's default input device. List the available
devices to choose another microphone.

```bash
python STT_init.py --list-devices
python STT_init.py --device "MacBook Pro Microphone"
```

You can also pass a numeric device index from the list. Device indices can change
when audio devices are connected or disconnected.

```bash
python STT_init.py --device 1
```

Use `--model` to select another extracted model directory. The default model path
is resolved relative to the script, so it also works when launched from another
directory. Explicit relative paths passed to `--model` use your current directory.

```bash
python STT_init.py --model /path/to/extracted-vosk-model
python STT_init.py --help
```

## Recognition and output behavior

- Audio is captured as mono, 16-bit samples at the selected microphone's default
  sample rate. Vosk receives that sample rate directly.
- Each audio chunk represents approximately 100 milliseconds. A queue holds up to
  10 chunks, or approximately one second of buffered audio.
- If processing falls behind and the queue fills, incoming chunks are dropped.
  A warning explains that speech may be missing, advises slowing down or pausing
  and restarting the sentence, and reports the cumulative dropped-chunk count.
- Vosk is configured with word output enabled and at most three alternatives.
  The script ranks candidates by average word confidence when every candidate
  supplies word confidence. The tested Vosk 0.3.44 installation omits those values
  in alternative results, so the script uses Vosk's sentence-level decoding
  scores. These scores are used for ranking and should not be read as percentages.
- `clean_transcript()` currently normalizes whitespace. It is the extension point
  for future text cleanup. The displayed alternatives come from Vosk.
- Recognition runs locally with the extracted model. Accepted text appears in the
  terminal. Recording audio and exporting transcripts would require additional
  functionality.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError` for `vosk` or `sounddevice` | Activate the intended environment and run `python -m pip install vosk sounddevice`. |
| Model directory error | Extract the model under `models/` as shown above, or provide its directory with `--model`. |
| Program stalls while opening the microphone on macOS | Check microphone permission for your terminal or IDE, then restart the program. |
| Microphone or PortAudio error | Check `--list-devices`, select an input device with `--device`, and verify system microphone access. Linux installations also require the PortAudio system library. |
| Speech produces no transcript | Say the start command, pause, and wait for `Transcription enabled.` Also check whether a confirmation or selection is pending. |
| Command or selection is misunderstood | Repeat the prompted response and pause. Say the number aloud, such as “two”. |
| Fewer than three interpretations appear | Choose an available interpretation or four to repeat. Vosk returns at most three candidates. |
| Audio-drop warnings appear repeatedly | Pause to let the buffer drain and repeat the affected sentence. Persistent drops indicate recognition is struggling to keep up with incoming audio. |
