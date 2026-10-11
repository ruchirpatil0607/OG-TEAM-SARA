"""
session_manager.py — Real-time Session State Machine and Data Manager for PulseVision.
Manages session lifecycle (IDLE, MONITORING, PAUSED, COMPLETED), tracks active acquisition
timers excluding paused periods, buffers rolling frames, incorporates user profile (age, gender, activity),
and computes session statistics and target heart rate zones.
"""

from collections import deque
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import (
    ALG_ADAPTIVE,
    BPM_MAX,
    BPM_MIN,
    BUFFER_WINDOW_SEC,
    DEFAULT_ACTIVITY_STATE,
    DEFAULT_ALGORITHM,
    DEFAULT_USER_AGE,
    DEFAULT_USER_GENDER,
    DEFAULT_USER_NAME,
    MIN_BUFFER_SEC,
    get_heart_rate_zone,
    get_max_heart_rate,
)
from src.heart_rate import HeartRateEstimate, HeartRateEstimator
from src.pulse_processing import extract_pulse_signal
from src.signal_quality import SignalQualityAssessment, SignalQualityEvaluator


class SessionState:
    IDLE = "IDLE"
    MONITORING = "MONITORING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"


class SessionManager:
    """Manages active live monitoring sessions, streaming buffers, and statistics."""

    def __init__(self):
        self.state = SessionState.IDLE
        self.session_id: str = ""
        
        # User Demographics & Health Profile
        self.user_label: str = DEFAULT_USER_NAME
        self.user_age: int = DEFAULT_USER_AGE
        self.user_gender: str = DEFAULT_USER_GENDER
        self.activity_state: str = DEFAULT_ACTIVITY_STATE
        
        self.target_duration_sec: float = 60.0
        self.algorithm: str = DEFAULT_ALGORITHM

        # Timing
        self.start_wall_time: Optional[datetime] = None
        self.end_wall_time: Optional[datetime] = None
        self._active_elapsed_sec: float = 0.0
        self._last_active_timestamp: Optional[float] = None
        self._pause_start_timestamp: Optional[float] = None

        # Rolling Buffer for live rPPG extraction (bounded to BUFFER_WINDOW_SEC)
        self.buf_timestamps = deque()
        self.buf_r = deque()
        self.buf_g = deque()
        self.buf_b = deque()

        # Cumulative Session History for Analytics and Export
        self.history_elapsed_sec: List[float] = []
        self.history_raw_r: List[float] = []
        self.history_raw_g: List[float] = []
        self.history_raw_b: List[float] = []
        self.history_bpm_records: List[Tuple[float, float, float]] = []  # (elapsed, bpm, sqi)
        self.history_sqi_records: List[Tuple[float, float, str]] = []    # (elapsed, sqi, status)

        # Latest Instantaneous Readings
        self.latest_estimate: HeartRateEstimate = HeartRateEstimate()
        self.latest_quality: SignalQualityAssessment = SignalQualityAssessment()
        self.latest_pulse_waveform: np.ndarray = np.array([])

        # Subsystems
        self.hr_estimator = HeartRateEstimator()

    def start_session(
        self,
        user_label: str = DEFAULT_USER_NAME,
        user_age: int = DEFAULT_USER_AGE,
        user_gender: str = DEFAULT_USER_GENDER,
        activity_state: str = DEFAULT_ACTIVITY_STATE,
        duration_sec: float = 60.0,
        algorithm: str = DEFAULT_ALGORITHM
    ):
        """Initializes and begins a new monitoring session with user profile."""
        self.session_id = f"PV-{int(time.time())}"
        self.user_label = user_label or DEFAULT_USER_NAME
        self.user_age = int(user_age or DEFAULT_USER_AGE)
        self.user_gender = user_gender or DEFAULT_USER_GENDER
        self.activity_state = activity_state or DEFAULT_ACTIVITY_STATE
        
        self.target_duration_sec = duration_sec
        self.algorithm = algorithm
        self.start_wall_time = datetime.now()
        self.end_wall_time = None
        
        self._active_elapsed_sec = 0.0
        self._last_active_timestamp = time.time()
        self._pause_start_timestamp = None
        
        self.buf_timestamps.clear()
        self.buf_r.clear()
        self.buf_g.clear()
        self.buf_b.clear()

        self.history_elapsed_sec.clear()
        self.history_raw_r.clear()
        self.history_raw_g.clear()
        self.history_raw_b.clear()
        self.history_bpm_records.clear()
        self.history_sqi_records.clear()

        self.latest_estimate = HeartRateEstimate()
        self.latest_quality = SignalQualityAssessment()
        self.latest_pulse_waveform = np.array([])
        self.hr_estimator.reset()

        self.state = SessionState.MONITORING

    def pause_session(self):
        """Pauses acquisition timer and sample intake."""
        if self.state == SessionState.MONITORING:
            self.state = SessionState.PAUSED
            self._pause_start_timestamp = time.time()

    def resume_session(self):
        """Resumes acquisition timer."""
        if self.state == SessionState.PAUSED:
            self._last_active_timestamp = time.time()
            self._pause_start_timestamp = None
            self.state = SessionState.MONITORING

    def stop_session(self):
        """Terminates session and locks final elapsed time."""
        if self.state in [SessionState.MONITORING, SessionState.PAUSED]:
            self.state = SessionState.COMPLETED
            self.end_wall_time = datetime.now()

    def restart_session(self):
        """Restarts the session with the same parameters."""
        self.start_session(
            user_label=self.user_label,
            user_age=self.user_age,
            user_gender=self.user_gender,
            activity_state=self.activity_state,
            duration_sec=self.target_duration_sec,
            algorithm=self.algorithm
        )

    def clear_session(self):
        """Resets session to idle state."""
        self.state = SessionState.IDLE
        self.buf_timestamps.clear()
        self.buf_r.clear()
        self.buf_g.clear()
        self.buf_b.clear()
        self.history_bpm_records.clear()
        self.history_sqi_records.clear()
        self.latest_estimate = HeartRateEstimate()
        self.latest_quality = SignalQualityAssessment()
        self.latest_pulse_waveform = np.array([])

    def add_sample(
        self,
        current_time: float,
        raw_rgb: Optional[Tuple[float, float, float]],
        face_detected: bool,
        motion_velocity: float,
        skin_ratio: float
    ):
        """
        Processes a single video frame's color sample into the active session.
        Calculates time accumulation, sliding window rPPG, and updates estimates.
        """
        if self.state != SessionState.MONITORING:
            return

        # Update active elapsed time (excluding paused duration)
        now = time.time()
        if self._last_active_timestamp is not None:
            dt = now - self._last_active_timestamp
            self._active_elapsed_sec += max(0.0, dt)
        self._last_active_timestamp = now

        # Check target duration completion
        if self.target_duration_sec > 0 and self._active_elapsed_sec >= self.target_duration_sec:
            self.stop_session()
            return

        elapsed = round(self._active_elapsed_sec, 2)

        # Store sample if valid color available
        if raw_rgb is not None:
            r, g, b = raw_rgb
            self.buf_timestamps.append(current_time)
            self.buf_r.append(r)
            self.buf_g.append(g)
            self.buf_b.append(b)

            # Cumulative history
            self.history_elapsed_sec.append(elapsed)
            self.history_raw_r.append(r)
            self.history_raw_g.append(g)
            self.history_raw_b.append(b)

            # Drop samples outside rolling BUFFER_WINDOW_SEC
            while self.buf_timestamps and (current_time - self.buf_timestamps[0] > BUFFER_WINDOW_SEC):
                self.buf_timestamps.popleft()
                self.buf_r.popleft()
                self.buf_g.popleft()
                self.buf_b.popleft()

        # Compute sampling rate and window duration
        n_samples = len(self.buf_timestamps)
        if n_samples >= 2:
            buffer_duration = self.buf_timestamps[-1] - self.buf_timestamps[0]
            fs = n_samples / max(0.1, buffer_duration)
        else:
            buffer_duration = 0.0
            fs = 30.0

        # Perform pulse extraction as soon as initial buffer is ready
        if n_samples >= 35 and buffer_duration >= MIN_BUFFER_SEC:
            r_arr = np.array(self.buf_r)
            g_arr = np.array(self.buf_g)
            b_arr = np.array(self.buf_b)

            # Extract optical pulse waveform
            filtered_pulse = extract_pulse_signal(r_arr, g_arr, b_arr, fs, algorithm=self.algorithm)
            self.latest_pulse_waveform = filtered_pulse

            # Estimate heart rate
            est = self.hr_estimator.estimate(filtered_pulse, fs, apply_smoothing=True)
            self.latest_estimate = est

            # Evaluate signal quality index
            sqi_eval = SignalQualityEvaluator.evaluate(
                face_detected=face_detected,
                motion_velocity=motion_velocity,
                skin_pixel_ratio=skin_ratio,
                spectral_snr_db=est.snr_db,
                buffer_duration_sec=buffer_duration,
                samples_count=n_samples
            )
            self.latest_quality = sqi_eval

            # Record points in session history
            if est.bpm is not None:
                self.history_bpm_records.append((elapsed, est.bpm, sqi_eval.sqi))
            
            self.history_sqi_records.append((elapsed, sqi_eval.sqi, sqi_eval.status_category))

        else:
            # Buffer accumulating
            sqi_eval = SignalQualityEvaluator.evaluate(
                face_detected=face_detected,
                motion_velocity=motion_velocity,
                skin_pixel_ratio=skin_ratio,
                spectral_snr_db=0.0,
                buffer_duration_sec=buffer_duration,
                samples_count=n_samples
            )
            self.latest_quality = sqi_eval
            self.latest_estimate = HeartRateEstimate(
                bpm=self.hr_estimator.smoothed_bpm,
                is_valid=False,
                message=f"Acquiring optical pulse ({buffer_duration:.1f}s / {MIN_BUFFER_SEC:.1f}s)..."
            )

    def get_session_statistics(self) -> Dict[str, any]:
        """Calculates summary statistics and personalized cardiac training zones."""
        valid_bpms = [rec[1] for rec in self.history_bpm_records]
        valid_count = len(valid_bpms)
        elapsed = round(self._active_elapsed_sec, 1)

        if valid_count > 0:
            avg_bpm = round(float(np.mean(valid_bpms)), 1)
            min_bpm = round(float(np.min(valid_bpms)), 1)
            max_bpm = round(float(np.max(valid_bpms)), 1)
        else:
            avg_bpm = None
            min_bpm = None
            max_bpm = None

        usable_pct = 0.0
        if elapsed > 0 and self.history_sqi_records:
            reliable_sqi_count = sum(1 for _, sqi, _ in self.history_sqi_records if sqi >= 40.0)
            usable_pct = round(min(100.0, (reliable_sqi_count / len(self.history_sqi_records)) * 100.0), 1)

        max_hr = get_max_heart_rate(self.user_age)
        zone_info = get_heart_rate_zone(avg_bpm, self.user_age)

        return {
            "session_id": self.session_id,
            "label": self.user_label,
            "user_label": self.user_label,
            "user_age": self.user_age,
            "user_gender": self.user_gender,
            "activity_state": self.activity_state,
            "max_hr": max_hr,
            "target_zone": zone_info["zone_name"],
            "zone_info": zone_info,
            "duration_sec": elapsed,
            "target_duration_sec": self.target_duration_sec,
            "valid_count": valid_count,
            "avg_bpm": avg_bpm,
            "min_bpm": min_bpm,
            "max_bpm": max_bpm,
            "usable_pct": usable_pct,
            "algorithm": self.algorithm,
            "start_time": self.start_wall_time.isoformat() if self.start_wall_time else "",
            "end_time": (self.end_wall_time or datetime.now()).isoformat()
        }
