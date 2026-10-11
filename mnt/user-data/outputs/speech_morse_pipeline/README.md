# Speak -> Morse -> Audio Pipeline — starter code

The most ambitious of the eight: chains speech-to-text onto Project
01's Morse link, so a spoken sentence ends up transmitted as Morse
tones and decoded back to text by a second script.

- **morse_link_common.py** — copied straight from Project 01 (text↔
  Morse↔audio, plus the decoder). No changes needed — reuse, don't
  rewrite.
- **speech_morse_pipeline_sender.py** — listens on the mic, transcribes
  speech to text (Google's free Speech Recognition API via the
  `speech_recognition` library, needs internet), converts to Morse,
  plays it through the speaker.
- **speech_morse_pipeline_receiver.py** — same as Project 01's
  receiver: records a window, decodes the Morse tones back to text.

## Setup
```
pip install SpeechRecognition sounddevice numpy scipy pyaudio
python speech_morse_pipeline_receiver.py   # start listening first, on laptop B
python speech_morse_pipeline_sender.py     # then speak, on laptop A
```
Note: `pyaudio` can be fiddly to install on some systems (it's a
dependency of `speech_recognition`'s microphone support) — if `pip
install pyaudio` fails, look up the platform-specific fix early, don't
leave it for the last few hours.

## Build order (this is the "stretch" project — build 01 first)
1. Get Project 01 (Acoustic Morse Link) fully working first — this
   project depends on it entirely for the transmit/receive half.
2. Test speech-to-text alone (print the transcribed text, no Morse
   yet) to confirm mic + internet + API access all work in your venue.
3. Wire the two halves together last.

## Known weak point
Free speech-to-text accuracy drops in loud rooms — have a typed-text
fallback path ready (just call `morse.text_to_audio(text)` directly)
in case live transcription is unreliable during judging.
