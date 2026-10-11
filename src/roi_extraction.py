"""
roi_extraction.py — Facial skin Region of Interest (ROI) extraction for PulseVision.
Isolates Forehead, Left Cheek, and Right Cheek skin patches, filters non-skin artifacts,
and extracts mean/median RGB color time-series.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.face_tracking import FaceDetectionResult


class ROIExtractor:
    """Extracts Forehead, Left Cheek, and Right Cheek skin patches with noise filtering."""

    @staticmethod
    def extract_rois(bgr_frame: np.ndarray, face_res: FaceDetectionResult) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract image crops for 'forehead', 'left_cheek', 'right_cheek'.
        Returns dictionary mapping ROI names to BGR patches.
        """
        h, w, _ = bgr_frame.shape
        rois = {"forehead": None, "left_cheek": None, "right_cheek": None, "combined": None}
        
        if not face_res.detected or not face_res.bbox:
            return rois

        fx, fy, fw, fh = face_res.bbox

        if face_res.landmarks and len(face_res.landmarks) >= 468:
            lms = face_res.landmarks
            # Landmark 10: Forehead top, 107: Left eyebrow top, 336: Right eyebrow top
            p_top = lms[10]
            p_leb = lms[107]
            p_reb = lms[336]
            
            # Forehead crop
            fh_y1 = max(0, min(p_top[1], fy))
            fh_y2 = max(fh_y1 + 10, min(p_leb[1], p_reb[1]) - 5)
            fh_x1 = max(0, min(p_leb[0], p_top[0] - int(0.20 * fw)))
            fh_x2 = min(w - 1, max(p_reb[0], p_top[0] + int(0.20 * fw)))
            
            if fh_x2 > fh_x1 + 10 and fh_y2 > fh_y1 + 10:
                rois["forehead"] = bgr_frame[fh_y1:fh_y2, fh_x1:fh_x2]

            # Landmark 117, 101, 50: Left Cheek (anatomical left of face / right in mirrored frame)
            # Below eye (landmark 145), outer of nose (landmark 1)
            p_leye_bottom = lms[145]
            p_nose = lms[1]
            lc_y1 = min(h - 1, p_leye_bottom[1] + 10)
            lc_y2 = min(h - 1, lc_y1 + int(0.22 * fh))
            lc_x1 = max(0, min(lms[205][0], lms[117][0]))
            lc_x2 = min(w - 1, max(p_leye_bottom[0], lc_x1 + int(0.20 * fw)))
            if lc_x2 > lc_x1 + 8 and lc_y2 > lc_y1 + 8:
                rois["left_cheek"] = bgr_frame[lc_y1:lc_y2, lc_x1:lc_x2]

            # Right Cheek
            p_reye_bottom = lms[374]
            rc_y1 = min(h - 1, p_reye_bottom[1] + 10)
            rc_y2 = min(h - 1, rc_y1 + int(0.22 * fh))
            rc_x1 = max(0, min(p_reye_bottom[0] - int(0.10 * fw), lms[346][0]))
            rc_x2 = min(w - 1, max(lms[425][0], rc_x1 + int(0.20 * fw)))
            if rc_x2 > rc_x1 + 8 and rc_y2 > rc_y1 + 8:
                rois["right_cheek"] = bgr_frame[rc_y1:rc_y2, rc_x1:rc_x2]

        else:
            # Geometric proportional fallback based on bounding box
            # Forehead: top 15% to 32%, center 50%
            fh_y1 = max(0, fy + int(0.12 * fh))
            fh_y2 = max(fh_y1 + 10, fy + int(0.30 * fh))
            fh_x1 = max(0, fx + int(0.25 * fw))
            fh_x2 = min(w - 1, fx + int(0.75 * fw))
            rois["forehead"] = bgr_frame[fh_y1:fh_y2, fh_x1:fh_x2]

            # Left Cheek: 50% to 70% height, 15% to 38% width
            lc_y1 = max(0, fy + int(0.50 * fh))
            lc_y2 = max(lc_y1 + 10, fy + int(0.70 * fh))
            lc_x1 = max(0, fx + int(0.15 * fw))
            lc_x2 = min(w - 1, fx + int(0.38 * fw))
            rois["left_cheek"] = bgr_frame[lc_y1:lc_y2, lc_x1:lc_x2]

            # Right Cheek: 50% to 70% height, 62% to 85% width
            rc_y1 = max(0, fy + int(0.50 * fh))
            rc_y2 = max(rc_y1 + 10, fy + int(0.70 * fh))
            rc_x1 = max(0, fx + int(0.62 * fw))
            rc_x2 = min(w - 1, fx + int(0.85 * fw))
            rois["right_cheek"] = bgr_frame[rc_y1:rc_y2, rc_x1:rc_x2]

        return rois

    @staticmethod
    def filter_skin_pixels(roi_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Creates a binary mask selecting plausible skin pixels and rejecting shadows/specular glare.
        Returns: (masked_roi_pixels, skin_pixel_ratio)
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return np.empty((0, 3)), 0.0

        # Discard extreme dark (hair/shadow) and extreme saturation (glare)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        valid_mask = (gray > 35) & (gray < 245)

        total_pixels = roi_bgr.shape[0] * roi_bgr.shape[1]
        valid_count = np.count_nonzero(valid_mask)
        ratio = valid_count / float(total_pixels) if total_pixels > 0 else 0.0

        if valid_count > 0:
            skin_pixels = roi_bgr[valid_mask]
            return skin_pixels, ratio
        return np.empty((0, 3)), 0.0

    @classmethod
    def extract_color_means(cls, bgr_frame: np.ndarray, face_res: FaceDetectionResult) -> Tuple[Optional[Tuple[float, float, float]], float]:
        """
        Extract spatial mean (R, G, B) across all valid facial skin ROIs.
        Returns: ((mean_R, mean_G, mean_B), skin_pixel_ratio)
        """
        rois = cls.extract_rois(bgr_frame, face_res)
        all_skin_pixels = []
        ratios = []

        for name in ["forehead", "left_cheek", "right_cheek"]:
            patch = rois.get(name)
            if patch is not None and patch.size > 0:
                pixels, ratio = cls.filter_skin_pixels(patch)
                if len(pixels) > 0:
                    all_skin_pixels.append(pixels)
                    ratios.append(ratio)

        if not all_skin_pixels:
            return None, 0.0

        combined_pixels = np.vstack(all_skin_pixels)
        # BGR to RGB
        mean_b = float(np.mean(combined_pixels[:, 0]))
        mean_g = float(np.mean(combined_pixels[:, 1]))
        mean_r = float(np.mean(combined_pixels[:, 2]))
        avg_ratio = float(np.mean(ratios)) if ratios else 0.0

        return (mean_r, mean_g, mean_b), avg_ratio

    @classmethod
    def draw_roi_overlays(cls, frame_bgr: np.ndarray, face_res: FaceDetectionResult) -> np.ndarray:
        """Renders glowing facial bounding box and Forehead / Cheek ROI markers on frame."""
        overlay = frame_bgr.copy()
        h, w, _ = frame_bgr.shape

        if not face_res.detected or not face_res.bbox:
            # Draw subtle guiding dashed oval
            cv2.ellipse(overlay, (w // 2, h // 2), (int(w * 0.18), int(h * 0.26)),
                        0, 0, 360, (100, 100, 120), 2, cv2.LINE_AA)
            cv2.putText(overlay, "Position Face Inside Marker", (w // 2 - 120, h // 2 + int(h * 0.32)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 220), 1, cv2.LINE_AA)
            return overlay

        fx, fy, fw, fh = face_res.bbox
        # Sleek rounded face box
        color_box = (254, 242, 0) if face_res.is_centered else (100, 150, 255) # BGR: Cyan vs Amber
        cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), color_box, 2, cv2.LINE_AA)

        # Draw ROI regions
        rois = cls.extract_rois(frame_bgr, face_res)
        # We can extract and highlight specific ROI bounding boxes
        if face_res.landmarks:
            lms = face_res.landmarks
            # Forehead marker
            p_top = lms[10]
            p_leb = lms[107]
            p_reb = lms[336]
            fh_y1 = max(0, min(p_top[1], fy))
            fh_y2 = max(fh_y1 + 10, min(p_leb[1], p_reb[1]) - 5)
            fh_x1 = max(0, min(p_leb[0], p_top[0] - int(0.20 * fw)))
            fh_x2 = min(w - 1, max(p_reb[0], p_top[0] + int(0.20 * fw)))
            cv2.rectangle(overlay, (fh_x1, fh_y1), (fh_x2, fh_y2), (0, 242, 254), 1, cv2.LINE_AA)
            cv2.putText(overlay, "Forehead", (fh_x1, max(15, fh_y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 242, 254), 1, cv2.LINE_AA)

            # Left Cheek marker
            p_leye_bottom = lms[145]
            lc_y1 = min(h - 1, p_leye_bottom[1] + 10)
            lc_y2 = min(h - 1, lc_y1 + int(0.22 * fh))
            lc_x1 = max(0, min(lms[205][0], lms[117][0]))
            lc_x2 = min(w - 1, max(p_leye_bottom[0], lc_x1 + int(0.20 * fw)))
            cv2.rectangle(overlay, (lc_x1, lc_y1), (lc_x2, lc_y2), (72, 202, 228), 1, cv2.LINE_AA)

            # Right Cheek marker
            p_reye_bottom = lms[374]
            rc_y1 = min(h - 1, p_reye_bottom[1] + 10)
            rc_y2 = min(h - 1, rc_y1 + int(0.22 * fh))
            rc_x1 = max(0, min(p_reye_bottom[0] - int(0.10 * fw), lms[346][0]))
            rc_x2 = min(w - 1, max(lms[425][0], rc_x1 + int(0.20 * fw)))
            cv2.rectangle(overlay, (rc_x1, rc_y1), (rc_x2, rc_y2), (72, 202, 228), 1, cv2.LINE_AA)

        # Draw status prompt badge at top of preview
        status_text = face_res.alignment_status
        badge_color = (16, 185, 129) if "Optimal" in status_text else (0, 165, 255)
        cv2.rectangle(overlay, (10, 10), (w - 10, 36), (20, 25, 40), -1)
        cv2.putText(overlay, f"Tracking: {status_text}", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_color, 1, cv2.LINE_AA)

        return overlay
