"""
signal_quality.py — Multi-Factor Signal Quality Index (SQI) Evaluator for PulseVision.
Assesses face continuity, motion stability, skin ROI viability, and spectral SNR
to guide the user with actionable instructions and prevent false readings.
"""

from src.config import (
    MAX_MOTION_VELOCITY,
    MIN_BUFFER_SEC,
    MIN_SKIN_PIXEL_RATIO,
    MIN_SNR_DB,
    SQI_EXCELLENT,
    SQI_FAIR,
    SQI_GOOD,
)


class SignalQualityAssessment:
    """Detailed signal quality index and user coaching diagnostics."""
    def __init__(
        self,
        sqi: float = 0.0,
        status_category: str = "UNRELIABLE",
        is_reliable: bool = False,
        guidance: str = "Position face in view",
        color_hex: str = "#EF4444"
    ):
        self.sqi = sqi
        self.status_category = status_category
        self.is_reliable = is_reliable
        self.guidance = guidance
        self.color_hex = color_hex


class SignalQualityEvaluator:
    """Evaluates multi-factor signal quality index (SQI)."""

    @staticmethod
    def evaluate(
        face_detected: bool,
        motion_velocity: float,
        skin_pixel_ratio: float,
        spectral_snr_db: float,
        buffer_duration_sec: float,
        samples_count: int
    ) -> SignalQualityAssessment:
        """
        Computes composite SQI score from 0 to 100% and determines reliability.
        """
        # 1. Immediate failure checks
        if not face_detected:
            return SignalQualityAssessment(
                sqi=0.0,
                status_category="NO_FACE",
                is_reliable=False,
                guidance="Position your face inside the camera frame",
                color_hex="#EF4444"
            )

        if buffer_duration_sec < MIN_BUFFER_SEC or samples_count < 35:
            progress_pct = min(100.0, (buffer_duration_sec / max(0.1, MIN_BUFFER_SEC)) * 100.0)
            return SignalQualityAssessment(
                sqi=round(progress_pct * 0.45, 1),
                status_category="ACQUIRING",
                is_reliable=True,  # Allow displaying initial calculated BPM while locking on
                guidance=f"Acquiring optical pulse buffer ({buffer_duration_sec:.1f}s / {MIN_BUFFER_SEC:.1f}s)...",
                color_hex="#48CAE4"
            )

        if motion_velocity > MAX_MOTION_VELOCITY:
            return SignalQualityAssessment(
                sqi=30.0,
                status_category="MOTION_ARTIFACT",
                is_reliable=False,
                guidance="Excessive head motion detected — please remain still",
                color_hex="#F59E0B"
            )

        if skin_pixel_ratio < MIN_SKIN_PIXEL_RATIO:
            return SignalQualityAssessment(
                sqi=38.0,
                status_category="POOR_LIGHTING",
                is_reliable=True,
                guidance="Low skin contrast — please face a light source",
                color_hex="#F59E0B"
            )

        # 2. Weighted Sub-Scores
        # Motion score (0 to 30 pts)
        motion_score = max(0.0, min(30.0, (1.0 - (motion_velocity / MAX_MOTION_VELOCITY)) * 30.0))

        # Skin viability score (0 to 25 pts)
        skin_score = max(5.0, min(25.0, (skin_pixel_ratio / 0.85) * 25.0))

        # Spectral SNR score (0 to 45 pts): mapping -6 dB to +6 dB
        snr_clamped = max(-6.0, min(8.0, spectral_snr_db))
        snr_score = max(10.0, min(45.0, ((snr_clamped + 6.0) / 14.0) * 45.0))

        total_sqi = round(motion_score + skin_score + snr_score, 1)

        # 3. Categorization & Reliability Determination
        if total_sqi >= SQI_EXCELLENT:
            cat = "EXCELLENT"
            color = "#10B981" # Emerald
            guidance = "Optimal optical pulse tracking"
            reliable = True
        elif total_sqi >= SQI_GOOD:
            cat = "GOOD"
            color = "#00F2FE" # Teal
            guidance = "Stable pulse measurement"
            reliable = True
        elif total_sqi >= SQI_FAIR:
            cat = "FAIR"
            color = "#F59E0B" # Amber
            guidance = "Moderate signal — keep head steady"
            reliable = True
        else:
            cat = "LOW_QUALITY"
            color = "#EF4444" # Red
            guidance = "Signal noisy — ensure steady lighting and no movement"
            reliable = False

        return SignalQualityAssessment(
            sqi=total_sqi,
            status_category=cat,
            is_reliable=reliable,
            guidance=guidance,
            color_hex=color
        )
