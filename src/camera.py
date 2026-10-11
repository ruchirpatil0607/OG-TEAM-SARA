"""
camera.py — Thread-safe webcam capture manager for PulseVision.
Handles video streaming, accurate timestamping, frame rate estimation,
graceful device disconnection, and frame-identity tracking to eliminate duplicate samples.
"""

import threading
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.config import DEFAULT_CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, TARGET_FPS


class CameraManager:
    """Thread-safe camera capture manager with real timestamp tracking and frame deduplication."""

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_paused = False
        
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp: float = 0.0
        self.frame_id: int = 0
        self._last_served_frame_id: int = -1

        self.fps_smoothed = TARGET_FPS
        self._last_fps_time = time.time()
        self._fps_counter = 0

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.error_message: Optional[str] = None

    def start(self) -> bool:
        """Start camera capture."""
        with self._lock:
            if self.is_running:
                return True
            
            try:
                # Use DirectShow backend on Windows for fast opening
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_index)
                
                if not self.cap.isOpened():
                    self.error_message = f"Could not open camera at index {self.camera_index}."
                    return False

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

                self.is_running = True
                self.is_paused = False
                self.error_message = None
                self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._thread.start()
                return True

            except Exception as e:
                self.error_message = f"Camera initialization error: {str(e)}"
                return False

    def pause(self):
        """Pause frame acquisition without releasing camera."""
        with self._lock:
            self.is_paused = True

    def resume(self):
        """Resume frame acquisition."""
        with self._lock:
            self.is_paused = False

    def _capture_loop(self):
        """Background thread loop continuously reading frames."""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.03)
                continue

            if not self.cap or not self.cap.isOpened():
                self.error_message = "Camera disconnected."
                break

            ret, frame = self.cap.read()
            now = time.time()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            with self._lock:
                self.latest_frame = frame
                self.latest_timestamp = now
                self.frame_id += 1
                self._fps_counter += 1

                # Update smoothed FPS every 1.0 second
                dt = now - self._last_fps_time
                if dt >= 1.0:
                    self.fps_smoothed = self._fps_counter / dt
                    self._fps_counter = 0
                    self._last_fps_time = now

            # Sleep slightly to match target FPS and not burn CPU
            time.sleep(0.005)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float, bool]:
        """
        Read the latest captured frame and timestamp.
        Returns: (success, frame_bgr, timestamp, is_new_frame)
        """
        with self._lock:
            if not self.is_running or self.latest_frame is None:
                return False, None, 0.0, False
            
            is_new = (self.frame_id != self._last_served_frame_id)
            self._last_served_frame_id = self.frame_id
            return True, self.latest_frame.copy(), self.latest_timestamp, is_new

    def get_fps(self) -> float:
        with self._lock:
            return max(15.0, min(60.0, self.fps_smoothed))

    def stop(self):
        """Stop background capture and release resources."""
        with self._lock:
            self.is_running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.latest_frame = None

    def release(self):
        self.stop()

    @staticmethod
    def list_available_cameras(max_probe: int = 3) -> List[int]:
        """Probe available camera indices."""
        available = []
        for i in range(max_probe):
            temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not temp_cap.isOpened():
                temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                available.append(i)
                temp_cap.release()
        return available if available else [0]
