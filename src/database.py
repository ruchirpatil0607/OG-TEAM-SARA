"""
database.py — SQLite Persistent Storage and Session History for PulseVision.
Handles session recording, retrieval, searching, comparison, and deletion with full user demographics.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional
from src.config import DB_PATH, get_max_heart_rate, get_heart_rate_zone


class DatabaseManager:
    """Manages SQLite session records, schema initialization, and querying."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates tables, indices, and performs schema migrations for demographic columns."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_label TEXT,
                    user_age INTEGER,
                    user_gender TEXT,
                    activity_state TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    duration_sec REAL,
                    algorithm TEXT,
                    avg_bpm REAL,
                    min_bpm REAL,
                    max_bpm REAL,
                    valid_count INTEGER,
                    usable_pct REAL,
                    bpm_series_json TEXT,
                    pulse_waveform_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Perform safe column additions if upgrading existing database
            cursor.execute("PRAGMA table_info(sessions);")
            columns = [row["name"] for row in cursor.fetchall()]

            if "user_age" not in columns:
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN user_age INTEGER;")
                except Exception:
                    pass
            if "user_gender" not in columns:
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN user_gender TEXT;")
                except Exception:
                    pass
            if "activity_state" not in columns:
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN activity_state TEXT;")
                except Exception:
                    pass

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);")
            conn.commit()

    def save_session(
        self,
        stats: Dict[str, Any],
        bpm_records: List[Any],
        pulse_waveform: Optional[Any] = None
    ) -> bool:
        """Saves completed session summary, demographics, and detailed time series."""
        try:
            bpm_json = json.dumps([
                {"t": round(rec[0], 2), "bpm": round(rec[1], 1), "sqi": round(rec[2], 1)}
                for rec in bpm_records
            ])

            if pulse_waveform is not None and len(pulse_waveform) > 0:
                step = max(1, len(pulse_waveform) // 300)
                sampled_wave = [round(float(v), 4) for v in pulse_waveform[::step]]
                wave_json = json.dumps(sampled_wave)
            else:
                wave_json = json.dumps([])

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (
                        session_id, user_label, user_age, user_gender, activity_state,
                        start_time, end_time, duration_sec, algorithm,
                        avg_bpm, min_bpm, max_bpm, valid_count, usable_pct,
                        bpm_series_json, pulse_waveform_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stats["session_id"],
                    stats.get("user_label", stats.get("label", "Guest")),
                    stats.get("user_age", 25),
                    stats.get("user_gender", "Prefer not to say"),
                    stats.get("activity_state", "Resting"),
                    stats.get("start_time", ""),
                    stats.get("end_time", ""),
                    stats.get("duration_sec", 0.0),
                    stats.get("algorithm", "ADAPTIVE"),
                    stats.get("avg_bpm"),
                    stats.get("min_bpm"),
                    stats.get("max_bpm"),
                    stats.get("valid_count", 0),
                    stats.get("usable_pct", 0.0),
                    bpm_json,
                    wave_json
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"[DatabaseManager] Save error: {e}")
            return False

    def list_sessions(self, search: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        """List past sessions ordered by creation date descending."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if search:
                    cursor.execute("""
                        SELECT session_id, user_label, user_age, user_gender, activity_state,
                               start_time, duration_sec, algorithm, avg_bpm, min_bpm, max_bpm,
                               valid_count, usable_pct, created_at
                        FROM sessions
                        WHERE user_label LIKE ? OR session_id LIKE ? OR activity_state LIKE ?
                        ORDER BY created_at DESC LIMIT ?
                    """, (f"%{search}%", f"%{search}%", f"%{search}%", limit))
                else:
                    cursor.execute("""
                        SELECT session_id, user_label, user_age, user_gender, activity_state,
                               start_time, duration_sec, algorithm, avg_bpm, min_bpm, max_bpm,
                               valid_count, usable_pct, created_at
                        FROM sessions
                        ORDER BY created_at DESC LIMIT ?
                    """, (limit,))
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    item = dict(row)
                    age = item.get("user_age") or 25
                    item["max_hr"] = get_max_heart_rate(age)
                    item["target_zone"] = get_heart_rate_zone(item.get("avg_bpm"), age)["zone_name"]
                    results.append(item)
                return results
        except Exception as e:
            print(f"[DatabaseManager] Query error: {e}")
            return []

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details of a specific session."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    res = dict(row)
                    res["bpm_series"] = json.loads(res.get("bpm_series_json") or "[]")
                    res["pulse_waveform"] = json.loads(res.get("pulse_waveform_json") or "[]")
                    age = res.get("user_age") or 25
                    res["max_hr"] = get_max_heart_rate(age)
                    z_info = get_heart_rate_zone(res.get("avg_bpm"), age)
                    res["target_zone"] = z_info["zone_name"]
                    res["zone_info"] = z_info
                    return res
                return None
        except Exception as e:
            print(f"[DatabaseManager] Get error: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a single session."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
            return True
        except Exception as e:
            print(f"[DatabaseManager] Delete error: {e}")
            return False

    def clear_all_sessions(self) -> bool:
        """Deletes all session records."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions;")
                conn.commit()
            return True
        except Exception as e:
            print(f"[DatabaseManager] Clear error: {e}")
            return False
