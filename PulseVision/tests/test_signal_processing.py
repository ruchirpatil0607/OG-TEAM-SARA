"""
test_signal_processing.py — Unit tests for digital filtering and rPPG extraction algorithms.
"""

import numpy as np
from src.pulse_processing import (
    butter_bandpass_filter,
    detrend_signal,
    extract_pulse_chrom,
    extract_pulse_green,
    extract_pulse_pos,
    normalize_channels,
)


def test_detrend_signal():
    """Verify that slow linear baseline drift is eliminated."""
    t = np.linspace(0, 10, 300)
    drift = 5.0 * t  # Linear ramp
    sine = np.sin(2.0 * np.pi * 1.2 * t)
    signal = sine + drift

    detrended = detrend_signal(signal, window_size=30)
    # The slope should be substantially reduced
    assert abs(np.mean(detrended[:50]) - np.mean(detrended[-50:])) < 2.0


def test_butter_bandpass_filter():
    """Verify bandpass filter passes in-band frequencies and rejects out-of-band frequencies."""
    fs = 30.0
    t = np.arange(0, 10.0, 1.0 / fs)
    
    # 0.1 Hz is out of band (low drift)
    low_freq = np.sin(2.0 * np.pi * 0.1 * t)
    # 1.2 Hz is in-band (72 BPM)
    in_band = np.sin(2.0 * np.pi * 1.2 * t)
    # 8.0 Hz is out of band (high frequency noise)
    high_freq = np.sin(2.0 * np.pi * 8.0 * t)

    composite = low_freq + in_band + high_freq
    filtered = butter_bandpass_filter(composite, fs, low=0.7, high=3.5)

    # In-band component power should dominate
    fft_vals = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    
    peak_freq = freqs[np.argmax(fft_vals)]
    assert abs(peak_freq - 1.2) < 0.15, f"Expected 1.2 Hz, got {peak_freq}"


def test_rppg_algorithms_output_shape():
    """Verify GREEN, CHROM, and POS algorithms return correctly shaped arrays."""
    fs = 30.0
    n = 300
    r = np.full(n, 120.0) + np.random.normal(0, 1.0, n)
    g = np.full(n, 100.0) + np.random.normal(0, 1.0, n)
    b = np.full(n, 80.0) + np.random.normal(0, 1.0, n)

    pulse_g = extract_pulse_green(r, g, b, fs)
    pulse_chrom = extract_pulse_chrom(r, g, b, fs)
    pulse_pos = extract_pulse_pos(r, g, b, fs)

    assert len(pulse_g) == n
    assert len(pulse_chrom) == n
    assert len(pulse_pos) == n
    assert np.all(np.isfinite(pulse_chrom))
    assert np.all(np.isfinite(pulse_pos))
