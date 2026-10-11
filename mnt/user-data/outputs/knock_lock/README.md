# Knock-Code Lock — starter code

- **knock_lock_common.py** — onset detection (envelope + peak
  finding), tempo-independent rhythm normalization, and a
  tolerance-based matcher. Run `python knock_lock_common.py` for a
  synthetic test: enrolls a rhythm, verifies the *same* rhythm at a
  different tempo matches, and confirms a *different* rhythm is
  rejected — no mic needed.

- **knock_lock_enroll.py** — records your secret knock rhythm and
  saves it to `secret_pattern.json`.

- **knock_lock_verify.py** — records an attempt and checks it against
  the saved pattern; prints UNLOCKED or denied.

## Setup
```
pip install numpy scipy sounddevice
python knock_lock_enroll.py   # do this once
python knock_lock_verify.py   # then try to unlock
```

## Notes
- The pattern is normalized (each gap as a fraction of the total
  duration), so knocking the right rhythm faster or slower still
  unlocks — only the relative timing matters, which is a nice detail
  to call out in the demo.
- `TOLERANCE` in knock_lock_verify.py controls strictness — lower it
  for a tighter/safer match, raise it if genuine attempts keep getting
  rejected.
- Bonus for the demo: tie a successful unlock to something visible —
  print a big banner, launch an app, or (safely) trigger a script.
