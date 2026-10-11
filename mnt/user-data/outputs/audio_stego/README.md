# Audio Steganography — starter code

- **audio_stego_common.py** — the core: message framing (length byte +
  payload bits), tone-burst embedding (FSK-style, two high frequencies
  standing in for bit 0/1), and a Goertzel-based decoder. Run
  `python audio_stego_common.py` for a self-contained round-trip test
  against a synthetic "music-like" carrier (no files needed).

- **audio_stego_encode.py** — hides a message inside a real `.wav`
  file: `python audio_stego_encode.py input.wav "secret" output.wav`

- **audio_stego_decode.py** — pulls the message back out:
  `python audio_stego_decode.py output.wav`

## Setup
```
pip install numpy soundfile
python audio_stego_encode.py song.wav "hi judges" hidden.wav
python audio_stego_decode.py hidden.wav
```

## Notes
- Embed frequencies (`F0`/`F1` = 6000/7000 Hz) sit high on purpose —
  most music and speech energy lives well below that, so the added
  tones stay close to inaudible while still being easy to detect with
  Goertzel.
- `EMBED_AMPLITUDE` trades inaudibility against reliability — raise it
  if decoding fails on louder carriers, lower it if the tones are
  audible in the demo.
- Good bonus for the demo: plot a spectrogram of the stego file next
  to the original so the hidden band is visible even though it's not
  really audible — nice "aha" moment for judges.
