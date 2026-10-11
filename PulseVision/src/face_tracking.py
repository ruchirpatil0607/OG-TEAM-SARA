"""
face_tracking.py — Face detection, landmark tracking, and motion analysis for PulseVision.
Uses MediaPipe FaceLandmarker to detect 478 3D landmarks, track motion stability,
and verify facial positioning for rPPG.
"""

import math
import os
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import MODEL_PATH


class FaceDetectionResult:
    """Encapsulates face detection coordinates, landmarks, and motion metrics."""
    def __init__(self, detected: bool = False):
        self.detected = detected
        self.bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self.landmarks: Optional[List[Tuple[int, int]]] = None  # Pixel coords
        self.motion_velocity: float = 0.0                      # px/frame displacement
        self.is_centered: bool = False
        self.is_proper_size: bool = False
        self.alignment_status: str = "No Face"                 # Human-readable prompt


class FaceTracker:
    """High-accuracy face tracker utilizing MediaPipe FaceLandmarker with motion analysis."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = str(model_path or MODEL_PATH)
        self.detector = None
        self.prev_center: Optional[Tuple[float, float]] = None
        self._init_detector()

    def _init_detector(self):
        """Initialize MediaPipe FaceLandmarker."""
        if not os.path.exists(self.model_path):
            alt_path = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")
            if os.path.exists(alt_path):
                self.model_path = os.path.abspath(alt_path)

        if os.path.exists(self.model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=self.model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[FaceTracker] Warning initializing FaceLandmarker: {e}")
                self.detector = None

    def process_frame(self, bgr_frame: np.ndarray) -> FaceDetectionResult:
        """
        Detect face and compute facial landmarks and motion velocity.
        bgr_frame: Input image (BGR)
        """
        h, w, _ = bgr_frame.shape
        res = FaceDetectionResult(detected=False)

        if self.detector is not None:
            try:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection = self.detector.detect(mp_image)

                if detection and detection.face_landmarks:
                    face_lms = detection.face_landmarks[0]
                    pixel_lms = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]
                    res.landmarks = pixel_lms
                    res.detected = True

                    # Calculate bounding box
                    xs = [pt[0] for pt in pixel_lms]
                    ys = [pt[1] for pt in pixel_lms]
                    min_x, max_x = max(0, min(xs)), min(w - 1, max(xs))
                    min_y, max_y = max(0, min(ys)), min(h - 1, max(ys))
                    bw = max_x - min_x
                    bh = max_y - min_y
                    res.bbox = (min_x, min_y, bw, bh)

                    # Center and size checks
                    cx = min_x + bw / 2.0
                    cy = min_y + bh / 2.0
                    res.is_centered = (0.25 * w <= cx <= 0.75 * w) and (0.20 * h <= cy <= 0.80 * h)
                    face_area_ratio = (bw * bh) / (w * h)
                    res.is_proper_size = 0.08 <= face_area_ratio <= 0.55

                    # Motion velocity computation
                    if self.prev_center is not None:
                        dx = cx - self.prev_center[0]
                        dy = cy - self.prev_center[1]
                        res.motion_velocity = math.hypot(dx, dy)
                    self.prev_center = (cx, cy)

                    # Alignment prompt
                    if not res.is_centered:
                        res.alignment_status = "Center your face in the frame"
                    elif not res.is_proper_size:
                        if face_area_ratio < 0.08:
                            res.alignment_status = "Move closer to the camera"
                        else:
                            res.alignment_status = "Move slightly further back"
                    elif res.motion_velocity > 15.0:
                        res.alignment_status = "Stay still for accurate reading"
                    else:
                        res.alignment_status = "Optimal alignment"

                    return res

            except Exception as e:
                pass

        # Fallback heuristic if MediaPipe detector is unavailable
        res.detected = False
        res.alignment_status = "No face detected"
        self.prev_center = None
        return res

    def close(self):
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None
