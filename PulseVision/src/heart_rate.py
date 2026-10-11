"""
heart_rate.py — Frequency Domain Analysis and Heart Rate Estimation for PulseVision.
Performs spectral density estimation (zero-padded FFT), parabolic peak refinement,
in-band spectral SNR calculation, physiological validation, and adaptive smoothing.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.signal import find_peaks

from src.config import (
    BPM_MAX,
    BPM_MIN,
    FFT_ZERO_PADDING,
    HR_HIGH_HZ,
    HR_LOW_HZ,
    MIN_SNR_DB,
)


class HeartRateEstimate:
    """Container for estimated heart rate and spectral parameters."""
    def __init__(
        self,
        bpm: Optional[float] = None,
        peak_freq_hz: float = 0.0,
        snr_db: float = 0.0,
        frequencies: Optional[np.ndarray] = None,
        power_spectrum: Optional[np.ndarray] = None,
        is_valid: bool = False,
        quality_score: float = 0.0,
        message: str = "Collecting signal"
    ):
        self.bpm = bpm
        self.peak_freq_hz = peak_freq_hz
        self.snr_db = snr_db
        self.frequencies = frequencies if frequencies is not None else np.array([])
        self.power_spectrum = power_spectrum if power_spectrum is not None else np.array([])
        self.is_valid = is_valid
        self.quality_score = quality_score
        self.message = message


class HeartRateEstimator:
    """Estimates BPM from filtered pulse waveform using spectral peak detection and adaptive smoothing."""

    def __init__(self, ema_alpha: float = 0.22):
        self.ema_alpha = ema_alpha
        self.smoothed_bpm: Optional[float] = None
        self.recent_peaks = []

    def reset(self):
        """Reset historical smoothing state for new session."""
        self.smoothed_bpm = None
        self.recent_peaks.clear()

    def estimate(
        self,
        pulse_signal: np.ndarray,
        fs: float,
        apply_smoothing: bool = True
    ) -> HeartRateEstimate:
        """
        Estimate heart rate in BPM from filtered pulse time-series.
        pulse_signal: 1D array of filtered optical blood volume waveform.
        fs: sampling rate in Hz.
        """
        sig = np.asarray(pulse_signal, dtype=np.float64)
        if not np.all(np.isfinite(sig)):
            med = np.nanmedian(sig)
            if not np.isfinite(med):
                med = 0.0
            sig = np.nan_to_num(sig, nan=med, posinf=med, neginf=med)

        n = len(sig)
        if n < 30 or fs < 10.0:
            return HeartRateEstimate(
                bpm=self.smoothed_bpm if self.smoothed_bpm is not None else None,
                is_valid=False,
                quality_score=0.0,
                message="Buffering initial pulse frames..."
            )

        # 1. Windowing and Zero-Padded FFT
        window = np.hanning(n)
        norm_sig = sig - np.mean(sig)
        std_val = np.std(norm_sig)
        if std_val > 1e-6:
            norm_sig = norm_sig / std_val

        windowed_sig = norm_sig * window

        # Zero-padding for ~0.015 Hz frequency resolution
        n_fft = max(FFT_ZERO_PADDING, 2 ** int(np.ceil(np.log2(n) + 2)))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        fft_vals = np.fft.rfft(windowed_sig, n=n_fft)
        mags = np.abs(fft_vals) ** 2  # Power Spectrum

        # 2. Cardiac Band [0.70 Hz, 3.50 Hz] (42 - 210 BPM)
        in_band = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)
        band_freqs = freqs[in_band]
        band_mags = mags[in_band]

        if len(band_freqs) == 0 or np.max(band_mags) < 1e-9:
            return HeartRateEstimate(
                bpm=self.smoothed_bpm,
                frequencies=freqs[freqs <= 4.0],
                power_spectrum=mags[freqs <= 4.0],
                is_valid=False,
                quality_score=0.0,
                message="Searching for optical pulse..."
            )

        # 3. Peak Detection with prominence & edge rejection
        # Identify top spectral peaks
        peaks, properties = find_peaks(band_mags, prominence=np.max(band_mags) * 0.08, distance=8)
        
        if len(peaks) > 0:
            # Pick highest peak among prominent candidates
            peak_idx = peaks[np.argmax(band_mags[peaks])]
        else:
            peak_idx = int(np.argmax(band_mags))

        peak_freq = float(band_freqs[peak_idx])

        # Parabolic interpolation around peak for sub-bin resolution
        if 0 < peak_idx < len(band_mags) - 1:
            y1 = np.log(max(1e-12, band_mags[peak_idx - 1]))
            y2 = np.log(max(1e-12, band_mags[peak_idx]))
            y3 = np.log(max(1e-12, band_mags[peak_idx + 1]))
            denom = 2.0 * (2.0 * y2 - y1 - y3)
            if abs(denom) > 1e-6:
                delta = (y1 - y3) / denom
                delta_f = band_freqs[1] - band_freqs[0]
                peak_freq = float(peak_freq + delta * delta_f)

        raw_bpm = round(peak_freq * 60.0, 1)

        # 4. Realistic In-Band Spectral SNR Calculation
        # Signal power: within +/- 0.18 Hz (~11 BPM) of cardiac peak
        delta_f = 0.18
        peak_mask = (band_freqs >= peak_freq - delta_f) & (band_freqs <= peak_freq + delta_f)
        p_signal = float(np.sum(band_mags[peak_mask]))
        p_noise = float(np.sum(band_mags[~peak_mask]))

        if p_noise < 1e-9:
            snr_db = 12.0
        else:
            snr_db = round(10.0 * np.log10(max(1e-6, p_signal / p_noise)), 2)

        # Peak Prominence Ratio
        max_power = float(band_mags[peak_idx])
        median_power = float(np.median(band_mags)) + 1e-9
        prominence_ratio = round(max_power / median_power, 2)

        # 5. Quality Score [0, 100]
        # Realistic mapping: SNR >= -2 dB -> 65+, SNR >= +2 dB -> 85+
        snr_factor = max(0.0, min(1.0, (snr_db + 6.0) / 12.0))
        prom_factor = max(0.0, min(1.0, (prominence_ratio - 1.2) / 4.0))
        quality_score = float(np.clip((0.65 * snr_factor + 0.35 * prom_factor) * 100.0, 15.0, 99.0))

        # 6. Validity Determination
        # Valid if within physiological human bounds and distinct from pure noise
        is_valid = bool((BPM_MIN <= raw_bpm <= BPM_MAX) and (snr_db >= MIN_SNR_DB or prominence_ratio >= 1.30))

        # 7. Adaptive Temporal Smoothing
        final_bpm = raw_bpm
        if is_valid and apply_smoothing:
            if self.smoothed_bpm is None:
                self.smoothed_bpm = raw_bpm
            else:
                diff = raw_bpm - self.smoothed_bpm
                # Clamp sudden jumps > 20 BPM to reject momentary twitches
                if abs(diff) > 20.0:
                    step = np.sign(diff) * 8.0
                    target = self.smoothed_bpm + step
                else:
                    target = raw_bpm
                self.smoothed_bpm = round(self.ema_alpha * target + (1.0 - self.ema_alpha) * self.smoothed_bpm, 1)

            final_bpm = self.smoothed_bpm
        elif self.smoothed_bpm is not None:
            # Hold last reliable estimate during brief noise
            final_bpm = self.smoothed_bpm

        status_msg = "Measurement Active" if is_valid else "Refining pulse signal..."

        return HeartRateEstimate(
            bpm=final_bpm,
            peak_freq_hz=round(peak_freq, 3),
            snr_db=snr_db,
            frequencies=band_freqs,
            power_spectrum=band_mags,
            is_valid=is_valid,
            quality_score=round(quality_score, 1),
            message=status_msg
        )
