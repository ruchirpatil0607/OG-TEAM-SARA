# FSK Walkie-Talkie — starter code

Three files:

- **fsk_walkie_talkie_dsp.py** — the core: text↔bits, frame format
  (preamble + length + payload + checksum), FSK modulator, Goertzel
  tone detector, and a preamble-search decoder that locates the frame
  inside a recording. Run `python fsk_walkie_talkie_dsp.py` for an
  offline round-trip sanity check (no mic/speaker needed).

- **fsk_walkie_talkie_sender.py** — run on the transmitting laptop.
  Type a message, it plays as tones out the speaker.

- **fsk_walkie_talkie_receiver.py** — run on the receiving laptop
  *first*. It counts down, records 8 seconds, then decodes and
  prints whatever it heard.

## Setup
```
pip install numpy sounddevice
```

## Demo flow
1. On laptop B: `python fsk_walkie_talkie_receiver.py` — wait for the countdown.
2. On laptop A, once B says "listening": `python fsk_walkie_talkie_sender.py`, type your message.
3. Watch B print the decoded text (and whether the checksum passed).

## Where to extend next (per the project plan)
1. **Tune for the room** — if `confidence < 0.8` fails often, try
   widening the tone gap (e.g. 1000/2000 Hz) or slowing `BAUD` down
   further; both make the receiver's job easier at the cost of speed.
2. **Continuous listening** — right now the receiver records a fixed
   window. A rolling buffer + energy-based trigger would let it listen
   indefinitely instead of needing a countdown.
3. **Retry on checksum fail** — have the sender repeat the frame 2–3×
   and have the receiver accept the first one that checksums clean.
4. **Backup plan** — keep a pre-recorded .wav of a working exchange in
   case the venue is too loud for a live mic demo.
