# Acoustic Morse Link — starter code

- **morse_link_common.py** — text↔Morse tables, Morse→audio renderer,
  and an envelope-based audio→Morse decoder. Run
  `python morse_link_common.py` for a self-contained round-trip test
  (no mic/speaker needed — sends "SOS HELLO" through the encode/decode
  pipeline in memory and checks it matches).

- **morse_link_sender.py** — type a message, it plays as tone bursts
  through the speaker.

- **morse_link_receiver.py** — records a fixed window from the mic and
  decodes it back to text.

## Setup
```
pip install numpy scipy sounddevice
python morse_link_receiver.py   # on the listening laptop, start first
python morse_link_sender.py     # then send from the other laptop
```

## Tuning notes
- `UNIT` in `morse_link_common.py` controls speed — increase it if the
  receiver is missing dots/dashes in a noisy room.
- `threshold_ratio` in `audio_to_morse()` sets how loud a tone must be
  relative to the recording's peak to count as "on" — lower it if
  quiet dots are being missed, raise it if background noise is being
  read as tones.
