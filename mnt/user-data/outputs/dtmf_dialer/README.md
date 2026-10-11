# DTMF Dialer & Decoder — starter code

- **dtmf_dialer_common.py** — frequency table, tone generator, and a
  Goertzel-algorithm decoder. Run `python dtmf_dialer_common.py` for a
  self-contained round-trip test (dials "911#42*0" in memory, decodes
  it back, checks it matches — no mic/speaker needed).

- **dtmf_dialer_keypad.py** — clickable on-screen keypad that plays
  real DTMF tones through the speaker as you dial.

- **dtmf_dialer_decoder.py** — records from the mic and decodes
  dialed digits live using the same Goertzel decoder.

## Setup
```
pip install numpy matplotlib sounddevice
python dtmf_dialer_decoder.py   # start listening first
python dtmf_dialer_keypad.py    # then dial on the keypad
```

## Notes
- The Goertzel algorithm checks energy at 7 known frequencies (4 row +
  3 column) instead of a full FFT — this is the same trick real
  telephone-exchange hardware used, and it's a nice line for the demo.
- `decode_chunk`'s noise-floor threshold (`1e3`) may need tuning
  against your laptop's mic gain — print `powers` while testing if
  digits are being missed or false-detected.
