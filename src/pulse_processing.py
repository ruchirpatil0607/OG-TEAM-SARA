"""
pulse_processing.py — rPPG Digital Signal Processing Pipeline for PulseVision.
Implements temporal normalization, baseline detrending, Butterworth bandpass filtering,
and rPPG extraction algorithms: Green Channel, CHROM, POS, and Adaptive Ensemble.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.signal import butter, detrend, filtfilt

from src.config import (
    ALG_ADAPTIVE,
    ALG_CHROM,
    ALG_GREEN,
    ALG_POS,
    BUTTERWORTH_ORDER,
    HR_HIGH_HZ,
    HR_LOW_HZ,
)


def butter_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = HR_LOW_HZ,
    high: float = HR_HIGH_HZ,
    order: int = BUTTERWORTH_ORDER
) -> np.ndarray:
    """
    Applies zero-phase Butterworth bandpass filter to 1D signal.
    Handles short signal lengths, invalid sampling rates, and NaN/inf values.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 18 or fs <= 0.0:
        return signal

    # Replace any NaNs or infinities with median
    if not np.all(np.isfinite(signal)):
        med = np.nanmedian(signal)
        if not np.isfinite(med):
            med = 0.0
        signal = np.nan_to_num(signal, nan=med, posinf=med, neginf=med)

    nyq = 0.5 * fs
    # Ensure cutoff frequencies satisfy 0 < low < high < nyq
    low_norm = max(0.01, min(low / nyq, 0.90))
    high_norm = max(low_norm + 0.05, min(high / nyq, 0.99))

    try:
        b, a = butter(order, [low_norm, high_norm], btype="bandpass")
        padlen = min(len(signal) - 1, 3 * max(len(a), len(b)))
        if padlen > 0:
            filtered = filtfilt(b, a, signal, padlen=padlen)
        else:
            filtered = signal - np.mean(signal)
        return filtered
    except Exception as e:
        return signal - np.mean(signal)


def detrend_signal(signal: np.ndarray, window_size: int = 15) -> np.ndarray:
    """Removes low-frequency baseline illumination drift using linear detrending."""
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 3:
        return signal - np.mean(signal)
    return detrend(signal, type="linear")


def normalize_channels(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizes color channels by dividing by their temporal mean:
    c_norm = (c / mean(c)) - 1.0.
    Eliminates constant ambient lighting intensity.
    """
    def _norm(c):
        mean_val = np.mean(c)
        if abs(mean_val) < 1e-6:
            return np.zeros_like(c)
        return (c / mean_val) - 1.0

    return _norm(r), _norm(g), _norm(b)


def extract_pulse_green(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> np.ndarray:
    """
    Baseline Green channel rPPG extraction.
    Green light has highest hemoglobin absorption contrast in skin.
    """
    _, g_norm, _ = normalize_channels(r, g, b)
    detrended = detrend_signal(g_norm)
    return butter_bandpass_filter(detrended, fs)


def extract_pulse_chrom(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> np.ndarray:
    """
    Chrominance-based rPPG method (de Haan & Jeanne, 2013).
    Projects normalized RGB signals onto two orthogonal chrominance vectors.
    """
    r_n, g_n, b_n = normalize_channels(r, g, b)

    xs = 3.0 * r_n - 2.0 * g_n
    ys = 1.5 * r_n + g_n - 1.5 * b_n

    std_xs = np.std(xs)
    std_ys = np.std(ys)

    if std_ys < 1e-6:
        alpha = 1.0
    else:
        alpha = std_xs / std_ys

    s = xs - alpha * ys
    detrended = detrend_signal(s)
    return butter_bandpass_filter(detrended, fs)


def extract_pulse_pos(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> np.ndarray:
    """
    Plane-Orthogonal-to-Skin (POS) rPPG method (Wang et al., 2017).
    Projects normalized signals onto plane orthogonal to skin-tone vector.
    """
    r_n, g_n, b_n = normalize_channels(r, g, b)

    p1 = g_n - b_n
    p2 = g_n + b_n - 2.0 * r_n

    std_p1 = np.std(p1)
    std_p2 = np.std(p2)

    if std_p2 < 1e-6:
        alpha = 1.0
    else:
        alpha = std_p1 / std_p2

    h = p1 + alpha * p2
    detrended = detrend_signal(h)
    return butter_bandpass_filter(detrended, fs)


def extract_pulse_adaptive(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> np.ndarray:
    """
    Adaptive Ensemble rPPG: Fuses Green channel baseline and POS projection.
    Provides superior noise suppression and maximal peak clarity under webcam illumination.
    """
    wave_green = extract_pulse_green(r, g, b, fs)
    wave_pos = extract_pulse_pos(r, g, b, fs)

    # Standardize variances
    std_g = np.std(wave_green) + 1e-6
    std_p = np.std(wave_pos) + 1e-6
    norm_g = wave_green / std_g
    norm_p = wave_pos / std_p

    # Combine: 55% Green + 45% POS
    combined = 0.55 * norm_g + 0.45 * norm_p
    return butter_bandpass_filter(combined, fs)


def extract_pulse_signal(
    r_series: np.ndarray,
    g_series: np.ndarray,
    b_series: np.ndarray,
    fs: float,
    algorithm: str = ALG_ADAPTIVE
) -> np.ndarray:
    """Unified dispatcher for rPPG pulse signal extraction."""
    r = np.asarray(r_series, dtype=np.float64)
    g = np.asarray(g_series, dtype=np.float64)
    b = np.asarray(b_series, dtype=np.float64)

    if algorithm == ALG_GREEN:
        return extract_pulse_green(r, g, b, fs)
    elif algorithm == ALG_POS:
        return extract_pulse_pos(r, g, b, fs)
    elif algorithm == ALG_CHROM:
        return extract_pulse_chrom(r, g, b, fs)
    else:
        return extract_pulse_adaptive(r, g, b, fs)


def compare_all_algorithms(
    r_series: np.ndarray,
    g_series: np.ndarray,
    b_series: np.ndarray,
    fs: float
) -> Dict[str, np.ndarray]:
    """Runs all algorithms on the same RGB data buffer for direct comparison."""
    return {
        ALG_ADAPTIVE: extract_pulse_adaptive(r_series, g_series, b_series, fs),
        ALG_GREEN: extract_pulse_green(r_series, g_series, b_series, fs),
        ALG_POS: extract_pulse_pos(r_series, g_series, b_series, fs),
        ALG_CHROM: extract_pulse_chrom(r_series, g_series, b_series, fs),
    }
