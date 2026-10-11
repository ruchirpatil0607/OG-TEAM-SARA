"""
test_session_manager.py — Unit tests for session lifecycle and statistics.
"""

import time
from src.session_manager import SessionManager, SessionState


def test_session_state_transitions():
    """Verify clean state transitions: IDLE -> MONITORING -> PAUSED -> MONITORING -> COMPLETED."""
    sm = SessionManager()
    assert sm.state == SessionState.IDLE

    sm.start_session(user_label="Test", duration_sec=30.0)
    assert sm.state == SessionState.MONITORING

    sm.pause_session()
    assert sm.state == SessionState.PAUSED

    sm.resume_session()
    assert sm.state == SessionState.MONITORING

    sm.stop_session()
    assert sm.state == SessionState.COMPLETED


def test_session_demographics_and_zones():
    """Verify that user age, gender, and activity state drive Max HR and Cardiac Zones."""
    sm = SessionManager()
    sm.start_session(
        user_label="Alice",
        user_age=30,
        user_gender="Female",
        activity_state="🏃 Post-Exercise / Cardio Recovery",
        duration_sec=60.0
    )

    assert sm.user_age == 30
    assert sm.user_gender == "Female"
    assert sm.activity_state == "🏃 Post-Exercise / Cardio Recovery"

    # Simulate cardiac data averaging 140 BPM (Zone 3/4 for age 30: max_hr = 190)
    sm.history_bpm_records = [
        (1.0, 138.0, 85.0),
        (2.0, 140.0, 90.0),
        (3.0, 142.0, 88.0),
    ]
    sm._active_elapsed_sec = 5.0

    stats = sm.get_session_statistics()
    assert stats["user_age"] == 30
    assert stats["max_hr"] == 190  # 220 - 30
    assert stats["avg_bpm"] == 140.0
    assert "Zone" in stats["target_zone"]


def test_session_statistics_calculation():
    """Verify average, minimum, and maximum BPM are calculated strictly from valid records."""
    sm = SessionManager()
    sm.start_session(user_label="StatsTest", duration_sec=60.0)

    # Manually populate valid BPM records: 70, 75, 80
    sm.history_bpm_records = [
        (1.0, 70.0, 85.0),
        (2.0, 75.0, 90.0),
        (3.0, 80.0, 88.0),
    ]
    sm.history_sqi_records = [
        (1.0, 85.0, "EXCELLENT"),
        (2.0, 90.0, "EXCELLENT"),
        (3.0, 88.0, "EXCELLENT"),
    ]
    sm._active_elapsed_sec = 10.0

    stats = sm.get_session_statistics()
    assert stats["valid_count"] == 3
    assert stats["avg_bpm"] == 75.0
    assert stats["min_bpm"] == 70.0
    assert stats["max_bpm"] == 80.0
    assert stats["usable_pct"] == 100.0


def test_empty_session_statistics():
    """Empty sessions must report None for averages without throwing exceptions."""
    sm = SessionManager()
    sm.start_session(user_label="EmptyTest", duration_sec=60.0)
    stats = sm.get_session_statistics()

    assert stats["valid_count"] == 0
    assert stats["avg_bpm"] is None
    assert stats["min_bpm"] is None
    assert stats["max_bpm"] is None
