# Webcam Heart Rate (rPPG) — starter code

- **heart_rate_webcam_dsp.py** — the actual signal processing: bandpass
  filter + FFT peak picking to estimate BPM from a green-channel
  intensity time series. Camera-independent — run
  `python heart_rate_webcam_dsp.py` for a synthetic test (injects a
  known 75 bpm pulse into fake noisy data and checks the estimate
  matches, no webcam needed).

- **heart_rate_webcam_app.py** — the live app: OpenCV face detection,
  forehead-region green-channel tracking, rolling-window BPM estimate
  overlaid on the video feed.

## Setup
```
pip install opencv-python numpy scipy
python heart_rate_webcam_app.py
```
Press `q` to quit. Give it ~8 seconds to fill its analysis window
before the BPM reading stabilizes.

## Tuning / demo notes
- Works best with steady, even lighting on the face and minimal
  movement — say so during the demo; it's an honest limitation of
  rPPG generally, not a bug.
- `WINDOW_SECONDS` trades responsiveness for stability — shorter
  reacts faster but is noisier.
- Nice bonus for the demo: plot the filtered green-channel waveform
  next to the video feed so judges can see the "pulse wave" directly
  (matplotlib in a second window, or overlay a small sparkline).
