"""
test_database.py — Unit tests for SQLite database manager with demographics.
"""

import tempfile
import os
from src.database import DatabaseManager


def test_database_crud():
    """Verify database creation, insert, read, list, and delete with demographics."""
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_path = temp_db.name
    temp_db.close()

    try:
        db = DatabaseManager(db_path=temp_path)

        # 1. Insert session with demographic fields
        stats = {
            "session_id": "TEST-001",
            "user_label": "Unit Test Subject",
            "user_age": 28,
            "user_gender": "Female",
            "activity_state": "🛋️ Resting / Relaxed",
            "max_hr": 192,
            "target_zone": "Zone 1 (Warm Up)",
            "start_time": "2026-10-09T12:00:00",
            "end_time": "2026-10-09T12:01:00",
            "duration_sec": 60.0,
            "algorithm": "ADAPTIVE (Ensemble)",
            "avg_bpm": 74.2,
            "min_bpm": 68.0,
            "max_bpm": 81.0,
            "valid_count": 45,
            "usable_pct": 92.5
        }
        bpm_records = [(1.0, 72.0, 85.0), (2.0, 74.0, 88.0)]
        pulse_wave = [0.1, 0.2, 0.3, -0.1]

        success = db.save_session(stats, bpm_records, pulse_wave)
        assert success is True

        # 2. Retrieve session and verify demographics
        retrieved = db.get_session("TEST-001")
        assert retrieved is not None
        assert retrieved["user_label"] == "Unit Test Subject"
        assert retrieved["user_age"] == 28
        assert retrieved["user_gender"] == "Female"
        assert retrieved["activity_state"] == "🛋️ Resting / Relaxed"
        assert retrieved["max_hr"] == 192
        assert "Resting" in retrieved["target_zone"]
        assert retrieved["avg_bpm"] == 74.2
        assert len(retrieved["bpm_series"]) == 2

        # 3. List sessions
        sessions_list = db.list_sessions()
        assert len(sessions_list) == 1
        assert sessions_list[0]["session_id"] == "TEST-001"
        assert sessions_list[0]["user_age"] == 28

        # 4. Search sessions
        filtered = db.list_sessions(search="Unit Test")
        assert len(filtered) == 1

        # 5. Delete session
        del_success = db.delete_session("TEST-001")
        assert del_success is True
        assert db.get_session("TEST-001") is None

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
