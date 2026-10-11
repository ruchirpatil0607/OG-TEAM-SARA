# 💓 PulseVision — Contactless Heart Rate Monitoring & Analytics System

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)]()

> **PulseVision** is a complete, interactive, contactless heart-rate estimation and analytics application built with Python, OpenCV, MediaPipe, SciPy, and Streamlit. It uses remote photoplethysmography (rPPG) to measure micro-variations in diffuse skin reflectance from a standard laptop webcam—with **zero dedicated sensors, zero external hardware, and zero cloud APIs required**.

---

## 📑 Table of Contents

- [1. Executive Summary & Capabilities](#1-executive-summary--capabilities)
- [2. System Architecture](#2-system-architecture)
- [3. Scientific Principle: How rPPG Works](#3-scientific-principle-how-rppg-works)
- [4. Digital Signal Processing Pipeline](#4-digital-signal-processing-pipeline)
  - [Stage A: Video Sampling & Timestamps](#stage-a-video-sampling--timestamps)
  - [Stage B: Facial Skin ROI Extraction](#stage-b-facial-skin-roi-extraction)
  - [Stage C: Temporal Detrending & Butterworth Filtering](#stage-c-temporal-detrending--butterworth-filtering)
  - [Stage D: rPPG Color Decompositions (Green, CHROM, POS)](#stage-d-rppg-color-decompositions-green-chrom-pos)
  - [Stage E: Frequency Analysis (Welch PSD & FFT)](#stage-e-frequency-analysis-welch-psd--fft)
  - [Stage F: Multi-Factor Signal Quality Index (SQI)](#stage-f-multi-factor-signal-quality-index-sqi)
- [5. Application Pages & Features](#5-application-pages--features)
- [6. Installation & Windows Setup Guide](#6-installation--windows-setup-guide)
- [7. How to Launch](#7-how-to-launch)
- [8. Automated Testing & Verification](#8-automated-testing--verification)
- [9. Privacy & Data Safety Policy](#9-privacy--data-safety-policy)
- [10. Known Limitations & Medical Disclaimer](#10-known-limitations--medical-disclaimer)
- [11. 3-Minute Hackathon Presentation Guide](#11-3-minute-hackathon-presentation-guide)

---

## 1. Executive Summary & Capabilities

PulseVision transforms standard optical hardware into a physiological estimation platform:

* **Real-Time Video Capture**: Thread-safe OpenCV video ingestion with microsecond timestamping and variable FPS handling.
* **MediaPipe Face Mesh Tracking**: Real-time 478-point facial landmark tracking to locate forehead and cheek ROIs while suppressing eyes, hair, and non-skin regions.
* **Personalized Health Profile & Target Cardiac Zones**: Captures Age, Biological Gender, Activity Context, and Participant Name to calculate age-predicted Maximum Heart Rate ($HR_{max} = 220 - \text{Age}$) and visualize real-time cardiovascular training intensity across 5 distinct metabolic zones.
* **Continuous & Responsive BPM Engine**: Fast initialization with parabolic sub-bin FFT peak interpolation and adaptive temporal EMA smoothing—providing instant feedback without hanging on `--`.
* **Four rPPG Extraction Algorithms**:
  * **ADAPTIVE (Ensemble)**: Default fusion combining high-sensitivity Green reflectance with POS motion-vector projection.
  * **Green Channel**: Fast absorption baseline.
  * **CHROM (Chrominance-based)**: Illumination-invariant method (de Haan & Jeanne, 2013).
  * **POS (Plane-Orthogonal-to-Skin)**: Motion-resistant skin-vector projection (Wang et al., 2017).
* **Multi-Factor Signal Quality Index (SQI)**: Combines face tracking stability, motion velocity, skin contrast, and spectral SNR.
* **Interactive Modern Glassmorphism Dashboard**: Dark Navy & Cyan design with glowing borders, pulsating heart animations, real-time waveform oscilloscopes, Welch PSD spectra, and interactive Plotly analytics.
* **Session Lifecycle**: Start, Pause, Resume, Stop, Restart, and Clear controls with pause-aware acquisition timers.
* **Local Persistence**: SQLite database storing participant demographics, summary statistics, and time-series JSON.
* **Downloadable Exports**: CSV tables and publication-grade ReportLab PDF reports with user demographics, training zones, charts, and regulatory notices.

---

## 2. System Architecture

```
PulseVision/
│
├── app.py                      # Main Streamlit Multi-Page Controller & Dashboard
├── requirements.txt            # Python Dependencies
├── README.md                   # Full Documentation & Scientific Guide
├── .gitignore                  # Git Ignore Rules
├── launch_pulsevision.bat      # One-Click Windows Batch Launcher
├── run_tests.py                # Standalone Test Suite Runner
├── face_landmarker.task        # MediaPipe FaceLandmarker Model Asset
│
├── assets/
│   └── styles.css              # Dark Navy, Teal, and Cyan Theme Stylesheet
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Physiological Constants & DSP Thresholds
│   ├── camera.py               # Thread-Safe Camera Streaming & Frame Buffer
│   ├── face_tracking.py        # MediaPipe FaceLandmarker & Motion Tracker
│   ├── roi_extraction.py       # Forehead & Cheek ROI Isolator & Skin Filter
│   ├── pulse_processing.py     # Detrending, Butterworth Filter, GREEN/CHROM/POS
│   ├── heart_rate.py           # Welch PSD, Zero-Padded FFT, SNR, BPM Smoothing
│   ├── signal_quality.py       # Multi-Factor SQI & Real-Time User Coaching
│   ├── session_manager.py      # State Machine, Live Buffers, Summary Stats
│   ├── database.py             # SQLite Persistence Engine & Query Layer
│   ├── analytics.py            # Plotly Chart Generators (Waveform, Trend, PSD)
│   └── report_generator.py     # CSV Exporter & ReportLab PDF Report Engine
│
├── tests/
│   ├── __init__.py
│   ├── test_heart_rate.py      # BPM Accuracy & Spectral Tests
│   ├── test_signal_processing.py # DSP Filter & Algorithm Tests
│   ├── test_session_manager.py # State Machine & Lifecycle Tests
│   ├── test_database.py        # SQLite Storage & Query Tests
│   └── test_reports.py         # CSV & PDF Export Validation Tests
│
└── data/
    ├── .gitkeep
    └── pulsevision.db          # Auto-Created SQLite Database
```

---

## 3. Scientific Principle: How rPPG Works

Remote Photoplethysmography (rPPG) exploits the optical properties of human blood and skin:
1. During each cardiac ventricular contraction, oxygenated blood surges into peripheral capillary networks.
2. Hemoglobin strongly absorbs light in the green spectrum (~520–580 nm) while reflecting more light in red and blue.
3. This creates minute, cyclical shifts in reflected skin luminance ($\approx 0.1\% - 1.0\%$ of overall intensity).
4. By spatially averaging pixel values across large skin regions (e.g., forehead and cheeks) and tracking the signal across time, digital filters can isolate this blood volume pulse (BVP).

---

## 4. Digital Signal Processing Pipeline

### Stage A: Video Sampling & Timestamps
Frames are captured with exact wall-clock timestamps $t_i = \text{time.time()}$ to compute instantaneous sampling intervals $\Delta t = t_i - t_{i-1}$, handling dropped frames and variable webcam rates.

### Stage B: Facial Skin ROI Extraction
Using MediaPipe FaceMesh:
* **Forehead**: Bounded between eyebrows (landmarks 107, 336) and top forehead (landmark 10).
* **Cheeks**: Sub-ocular zones below landmarks 117, 346, avoiding lips, nostrils, and eyes.
* **Skin Masking**: Discards extreme darks (hair/shadows, gray < 35) and specular reflections (glare, gray > 245). Computes spatial means: $\bar{R}(t), \bar{G}(t), \bar{B}(t)$.

### Stage C: Temporal Detrending & Butterworth Filtering
* **Normalization**: $c_n(t) = \frac{c(t)}{\mu_c} - 1.0$, eliminating constant baseline ambient lighting.
* **Linear Detrending**: Removes low-frequency lighting drifts using `scipy.signal.detrend`.
* **Zero-Phase Filter**: 3rd-order Butterworth bandpass filter ($0.70\text{ Hz} \le f \le 3.50\text{ Hz}$, corresponding to $42 - 210\text{ BPM}$) using `filtfilt` to prevent phase distortion.

### Stage D: rPPG Color Decompositions (Green, CHROM, POS)
1. **Green Baseline**: $S_{\text{GREEN}}(t) = G_n(t)$
2. **Chrominance-based (CHROM)** (de Haan & Jeanne, 2013):
   $$X_s = 3 R_n - 2 G_n, \quad Y_s = 1.5 R_n + G_n - 1.5 B_n$$
   $$S_{\text{CHROM}} = X_s - \left(\frac{\sigma(X_s)}{\sigma(Y_s)}\right) Y_s$$
3. **Plane-Orthogonal-to-Skin (POS)** (Wang et al., 2017):
   $$P_1 = G_n - B_n, \quad P_2 = G_n + B_n - 2 R_n$$
   $$S_{\text{POS}} = P_1 + \left(\frac{\sigma(P_1)}{\sigma(P_2)}\right) P_2$$

### Stage E: Frequency Analysis (Welch PSD & FFT)
* Applied with Hanning window and zero-padded to $N = 2048$ bins for $\approx 0.02\text{ Hz}$ frequency resolution.
* Dominant cardiac peak $f_{\text{peak}} \in [0.70, 3.50]\text{ Hz}$ yields heart rate:
  $$\text{BPM} = f_{\text{peak}} \times 60$$
* **Peak SNR**:
  $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left( \frac{P(f_{\text{peak}} \pm 0.15\text{Hz})}{P_{\text{total}} - P(f_{\text{peak}} \pm 0.15\text{Hz}) + \epsilon} \right)$$
* **Temporal Smoothing**: Exponential moving average ($\alpha = 0.25$) with maximum step rate limiting.

### Stage F: Multi-Factor Signal Quality Index (SQI)
Composite metric ($0\% - 100\%$):
$$\text{SQI} = \text{Score}_{\text{motion}} (30\text{pts}) + \text{Score}_{\text{skin}} (25\text{pts}) + \text{Score}_{\text{SNR}} (45\text{pts})$$
* **Reliability Gate**: If $\text{SQI} < 40\%$ or motion velocity $> 18\text{ px/frame}$, the application **withholds numerical BPM** and shows a clear prompt (e.g. *"Stay still"*, *"Position face"*). **No fake numbers are ever fabricated.**

---

## 5. Application Pages & Features

| Page | Description |
|---|---|
| **🏠 Dashboard** | Introduction to rPPG, best practice positioning guide, quick start configuration, privacy overview. |
| **💓 Live Monitor** | Live webcam feed with ROI overlays, instantaneous BPM card with pulsing heart, session stats (Avg, Min, Max, Usability %), Start/Pause/Resume/Stop/Restart buttons, live pulse wave, and Welch PSD spectrum. |
| **📊 Analytics** | Deep dive into session data: raw RGB channel time series, filtered optical waveforms, BPM trends (with gaps preserved), Welch PSD, SQI trends, and side-by-side algorithm comparison (Green vs CHROM vs POS). |
| **📜 Session History** | Searchable SQLite database of past sessions, detailed review, multi-session comparative overlays, and deletion tools. |
| **📄 Reports** | Downloadable CSV datasets (time series & summary) and publication-grade ReportLab PDF reports with embedded charts and non-diagnostic disclaimers. |
| **⚙️ Settings & Privacy** | Webcam device selection, duration presets, algorithm selection, auto-save preferences, and database management. |

---

## 6. Installation & Windows Setup Guide

### Prerequisites
* Windows 10 or 11
* Python 3.10, 3.11, or 3.12 (Tested on Python 3.11.5)
* Standard built-in or USB webcam

### Step-by-Step Setup
1. Open PowerShell or Command Prompt.
2. Navigate to the project directory:
   ```powershell
   cd C:\Users\vnile\Downloads\01-03-06-08-prijects\PulseVision
   ```
3. (Optional but recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

---

## 7. How to Launch

### Option 1: One-Click Batch Script
Double-click `launch_pulsevision.bat` in the `PulseVision` folder.

### Option 2: Command Line
```powershell
python -m streamlit run app.py
```
Streamlit will automatically open your default browser at:
```
http://localhost:8501
```

---

## 8. Automated Testing & Verification

Run the comprehensive unit and integration test suite:
```powershell
python run_tests.py
```
Or via pytest:
```powershell
python -m pytest tests -v
```

### Test Coverage Summary:
* `test_bpm_estimation_known_frequency`: Verifies synthetic 72 BPM sine wave estimates to within $\pm 2.5\text{ BPM}$.
* `test_bpm_estimation_different_rates`: Validates rates across 54, 80, 110, and 140 BPM.
* `test_insufficient_samples`: Ensures buffers $< 30$ samples return invalid status without crashing.
* `test_nan_and_inf_handling`: Verifies corrupted signals are sanitized.
* `test_detrend_signal`: Confirms linear illumination drift is eliminated.
* `test_butter_bandpass_filter`: Confirms out-of-band attenuation ($0.1\text{ Hz}$ and $8.0\text{ Hz}$ rejected, $1.2\text{ Hz}$ passed).
* `test_rppg_algorithms_output_shape`: Validates Adaptive, Green, CHROM, and POS array outputs.
* `test_session_state_transitions`: Verifies IDLE $\to$ MONITORING $\to$ PAUSED $\to$ COMPLETED lifecycle.
* `test_session_demographics_and_zones`: Verifies Age, Gender, Activity State, Max HR, and Cardiac Zones calculation.
* `test_session_statistics_calculation`: Confirms mathematical accuracy of mean, min, and max.
* `test_database_crud`: Verifies SQLite table creation, demographic insertion, querying, and deletion.
* `test_csv_generation`: Verifies detailed points and summary CSV exports.
* `test_pdf_generation`: Validates ReportLab PDF generation, demographic tables, embedded charts, and disclaimers.

---

## 9. Privacy & Data Safety Policy

* **100% On-Device Ingestion**: All image frames are ingested directly into volatile RAM and discarded immediately after extracting spatial RGB averages.
* **No Video Retention**: No video clips, frames, or biometric photos are ever saved to disk or transmitted across the internet.
* **Zero Cloud APIs**: All computer vision and digital signal processing run locally on CPU.
* **User Control**: Historical session records stored in the local SQLite database can be searched, reviewed, or deleted by the user at any time.

---

## 10. Known Limitations & Medical Disclaimer

> ### ⚠️ IMPORTANT MEDICAL & REGULATORY DISCLAIMER
> **PulseVision is an experimental computer-vision research and wellness demonstration application. It is NOT a medical diagnostic device, has not been certified by the FDA, CE, or any health regulatory body, and must NOT be used as a substitute for professional medical advice, clinical electrocardiograms (ECG), or clinical pulse oximeters.**
>
> **Known Limitations:**
> * Ambient lighting shifts (e.g., flickering fluorescent lamps, rapid changes in sunlight) can introduce spectral noise.
> * Gross subject movement, speaking, or deep head nods degrade optical SNR.
> * Video compression artifacts or low-framerate webcams ($< 15\text{ FPS}$) limit precision.

---

## 11. 3-Minute Hackathon Presentation Guide

1. **Problem Statement (0:00 - 0:30)**:
   * "Traditional pulse monitors require wearable sensors, chest straps, or fingertip clips. What if your everyday laptop camera could track your cardiovascular rhythm completely contactlessly?"
2. **The Working Principle (0:30 - 1:00)**:
   * "With every heartbeat, blood surges into your facial micro-capillaries, causing sub-perceptual changes in light absorption. Using MediaPipe, we isolate the forehead and cheeks, extract raw RGB channels, filter them with a Butterworth bandpass, and isolate blood volume pulse waveforms."
3. **Live Demonstration (1:00 - 2:00)**:
   * Open `Live Monitor`.
   * Point camera at presenter: Show real-time forehead and cheek tracking badges.
   * Highlight the live optical pulse waveform oscillating with the heartbeat and the dominant Welch PSD peak matching the estimated BPM.
   * Demonstrate the **Pause / Resume** functionality and **Signal Quality Index (SQI)**.
4. **Analytics & Reports (2:00 - 2:40)**:
   * Switch to `Analytics`: Show the multi-algorithm comparison (Green vs CHROM vs POS).
   * Switch to `Reports`: Generate and download the publication-grade PDF report with embedded trend charts.
5. **Impact & Safety (2:40 - 3:00)**:
   * Highlight local execution, privacy-first design, and non-diagnostic honesty.
