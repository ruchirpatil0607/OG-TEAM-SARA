"""
test_heart_rate.py — Unit tests for frequency domain analysis and BPM estimation.
"""

import numpy as np
from src.heart_rate import HeartRateEstimator
from src.config import BPM_MAX, BPM_MIN


def test_bpm_estimation_known_frequency():
    """Verify that a synthetic sine wave at a known heart rate produces accurate BPM."""
    fs = 30.0  # 30 FPS
    duration = 10.0  # 10 seconds
    t = np.arange(0, duration, 1.0 / fs)
    
    # Test 72 BPM = 1.2 Hz
    target_bpm = 72.0
    freq_hz = target_bpm / 60.0
    synthetic_pulse = np.sin(2.0 * np.pi * freq_hz * t)

    estimator = HeartRateEstimator(ema_alpha=1.0)
    res = estimator.estimate(synthetic_pulse, fs, apply_smoothing=False)

    assert res.is_valid is True
    assert res.bpm is not None
    # Tolerance within +/- 2.5 BPM
    assert abs(res.bpm - target_bpm) <= 2.5, f"Expected ~{target_bpm} BPM, got {res.bpm}"


def test_bpm_estimation_different_rates():
    """Test estimation across normal, low, and elevated physiological rates."""
    fs = 30.0
    t = np.arange(0, 12.0, 1.0 / fs)
    estimator = HeartRateEstimator(ema_alpha=1.0)

    for target_bpm in [54.0, 80.0, 110.0, 140.0]:
        pulse = np.sin(2.0 * np.pi * (target_bpm / 60.0) * t)
        res = estimator.estimate(pulse, fs, apply_smoothing=False)
        assert res.is_valid is True
        assert abs(res.bpm - target_bpm) <= 3.0, f"Expected {target_bpm}, got {res.bpm}"


def test_insufficient_samples():
    """Signals shorter than required minimum window should return invalid status without crashing."""
    fs = 30.0
    short_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    estimator = HeartRateEstimator()
    res = estimator.estimate(short_signal, fs)
    assert res.is_valid is False
    assert res.bpm is None


def test_nan_and_inf_handling():
    """Signals containing NaN or Inf must not crash."""
    fs = 30.0
    t = np.arange(0, 8.0, 1.0 / fs)
    corrupted = np.sin(2.0 * np.pi * 1.2 * t)
    corrupted[10] = np.nan
    corrupted[20] = np.inf

    estimator = HeartRateEstimator()
    res = estimator.estimate(corrupted, fs)
    # Must run cleanly without throwing exceptions
    assert isinstance(res.is_valid, bool)
