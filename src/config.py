"""
config.py — Central configuration and constants for PulseVision.
Defines physiological bands, camera parameters, quality thresholds,
user health profile standards, and UI styling tokens.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
DB_PATH = DATA_DIR / "pulsevision.db"
MODEL_PATH = BASE_DIR / "face_landmarker.task"

# Ensure data dir exists
os.makedirs(DATA_DIR, exist_ok=True)

# Application Metadata
APP_NAME = "PulseVision"
APP_SUBTITLE = "Contactless Heart Rate Monitoring & Analytics"
APP_VERSION = "2.0.0"

# User Profile Defaults
DEFAULT_USER_NAME = "Guest User"
DEFAULT_USER_AGE = 25
DEFAULT_USER_GENDER = "Prefer not to say"
DEFAULT_ACTIVITY_STATE = "Resting / Relaxed"

GENDER_OPTIONS = ["Male", "Female", "Non-Binary", "Prefer not to say"]
ACTIVITY_OPTIONS = [
    "🛋️ Resting / Relaxed",
    "🧘 Post-Meditation / Deep Breathing",
    "💻 Working / Desk Study",
    "🏃 Post-Exercise / Cardio Recovery",
    "☕ Post-Caffeine / Alert"
]

# Camera Settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30.0

# Physiological Heart Rate Bounds (Adult human range)
# 42 BPM = 0.70 Hz, 210 BPM = 3.50 Hz
HR_LOW_HZ = 0.70
HR_HIGH_HZ = 3.50
BPM_MIN = 42.0
BPM_MAX = 210.0

# Signal Processing Settings
BUFFER_WINDOW_SEC = 9.0        # Sliding pulse extraction window
MIN_BUFFER_SEC = 3.0           # Minimum duration before showing live estimates
MIN_SAMPLES_COUNT = 45         # Minimum frame samples needed
BUTTERWORTH_ORDER = 3          # Order of Butterworth bandpass filter
FFT_ZERO_PADDING = 2048        # Zero-padded FFT length for fine frequency resolution

# rPPG Extraction Algorithms
ALG_ADAPTIVE = "ADAPTIVE (Ensemble)"  # Robust Green + POS fusion
ALG_GREEN = "GREEN"                   # Pure Green channel baseline
ALG_CHROM = "CHROM"                   # Chrominance-based method (de Haan & Jeanne 2013)
ALG_POS = "POS"                       # Plane-Orthogonal-to-Skin (Wang et al. 2017)
AVAILABLE_ALGORITHMS = [ALG_ADAPTIVE, ALG_GREEN, ALG_POS, ALG_CHROM]
DEFAULT_ALGORITHM = ALG_ADAPTIVE

# Realistic Signal Quality Index (SQI) Thresholds for Webcam rPPG
SQI_EXCELLENT = 75.0
SQI_GOOD = 55.0
SQI_FAIR = 35.0
MIN_SNR_DB = -4.5              # Realistic rPPG in-band SNR threshold (dB)
MAX_MOTION_VELOCITY = 28.0     # Maximum acceptable face displacement in px/frame
MIN_SKIN_PIXEL_RATIO = 0.25    # Minimum ratio of valid skin pixels in ROI

# UI Theme Colors (Modern Glassmorphic Dark Navy & Cyan)
COLOR_BG_DARK = "#070B19"
COLOR_CARD_BG = "#0E172E"
COLOR_CARD_BORDER = "#1E2C4F"
COLOR_PRIMARY_TEAL = "#00F2FE"
COLOR_SECONDARY_CYAN = "#48CAE4"
COLOR_EMERALD_GREEN = "#10B981"
COLOR_AMBER_WARNING = "#F59E0B"
COLOR_CORAL_ALERT = "#EF4444"
COLOR_PURPLE_ACCENT = "#8B5CF6"
COLOR_TEXT_MAIN = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"


# Standard Cardiovascular Training Zones
CARDIAC_ZONES = {
    "Z0": {"name": "Resting Baseline", "min_pct": 0, "max_pct": 50, "color": "#94A3B8"},
    "Z1": {"name": "Zone 1: Warm Up", "min_pct": 50, "max_pct": 60, "color": "#48CAE4"},
    "Z2": {"name": "Zone 2: Fat Burn", "min_pct": 60, "max_pct": 70, "color": "#10B981"},
    "Z3": {"name": "Zone 3: Aerobic", "min_pct": 70, "max_pct": 80, "color": "#F59E0B"},
    "Z4": {"name": "Zone 4: Anaerobic", "min_pct": 80, "max_pct": 90, "color": "#F97316"},
    "Z5": {"name": "Zone 5: Maximum", "min_pct": 90, "max_pct": 100, "color": "#EF4444"},
}


def get_max_heart_rate(age: int) -> int:
    """Calculates age-predicted maximum heart rate (Fox formula: 220 - age)."""
    clean_age = max(5, min(120, int(age or 25)))
    return 220 - clean_age


def get_heart_rate_zone(bpm: Optional[float], age: int) -> Dict[str, any]:
    """
    Computes cardiovascular training/intensity zone based on age-predicted max HR.
    Returns: {zone_name, zone_code, min_bpm, max_bpm, color_hex, color, intensity_pct, percent_max, description}
    """
    if bpm is None or bpm <= 0:
        return {
            "zone_name": "Calibrating",
            "zone_code": "Z0",
            "intensity_pct": 0.0,
            "percent_max": 0.0,
            "color_hex": COLOR_TEXT_MUTED,
            "color": COLOR_TEXT_MUTED,
            "description": "Acquiring physiological signal..."
        }

    max_hr = get_max_heart_rate(age)
    intensity_pct = round((bpm / max_hr) * 100.0, 1)

    if bpm < 0.50 * max_hr:
        zone_name = "Resting Baseline"
        code = "Z0"
        color = "#94A3B8"
        desc = "Resting cardiac activity below aerobic thresholds. Recovery state."
    elif bpm < 0.60 * max_hr:
        zone_name = "Zone 1: Warm Up (50-60%)"
        code = "Z1"
        color = "#48CAE4"
        desc = "Active recovery and metabolic priming. Very light exertion."
    elif bpm < 0.70 * max_hr:
        zone_name = "Zone 2: Fat Burn (60-70%)"
        code = "Z2"
        color = "#10B981"
        desc = "Sustainable aerobic endurance. Optimal lipid oxidation."
    elif bpm < 0.80 * max_hr:
        zone_name = "Zone 3: Aerobic (70-80%)"
        code = "Z3"
        color = "#F59E0B"
        desc = "Moderate cardio effort. Enhances cardiac stroke volume."
    elif bpm < 0.90 * max_hr:
        zone_name = "Zone 4: Anaerobic (80-90%)"
        code = "Z4"
        color = "#F97316"
        desc = "High metabolic demand. Lactate threshold training."
    else:
        zone_name = "Zone 5: Maximum (90-100%)"
        code = "Z5"
        color = "#EF4444"
        desc = "Peak cardiovascular exertion. Maximum aerobic capacity (VO2 Max)."

    return {
        "zone_name": zone_name,
        "zone_code": code,
        "intensity_pct": intensity_pct,
        "percent_max": intensity_pct,
        "color_hex": color,
        "color": color,
        "description": desc
    }

