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
    def draw_roi_overlays(cls, frame_bgr: np.ndarray, face_res: FaceDetectionResult, show_mesh: bool = True, show_rois: bool = True) -> np.ndarray:
        """Renders glowing facial landmarks and Forehead / Cheek circular ROI markers matching reference specification."""
        overlay = frame_bgr.copy()
        h, w, _ = frame_bgr.shape

        if not face_res.detected or not face_res.bbox:
            # Draw subtle guiding dashed oval
            cv2.ellipse(overlay, (w // 2, h // 2), (int(w * 0.18), int(h * 0.26)),
                        0, 0, 360, (0, 242, 254), 1, cv2.LINE_AA)
            cv2.putText(overlay, "Center Face in Frame", (w // 2 - 85, h // 2 + int(h * 0.32)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
            return overlay

        fx, fy, fw, fh = face_res.bbox

        # 1. Draw Facial Landmark Points (Mesh overlay)
        if show_mesh and face_res.landmarks:
            lms = face_res.landmarks
            # Draw subsampled landmark points in glowing cyan/emerald
            step = 3 if len(lms) > 100 else 1
            for i in range(0, len(lms), step):
                pt = lms[i]
                cv2.circle(overlay, pt, 1, (0, 242, 254), -1, cv2.LINE_AA)

        # 2. Draw Circular Cheek & Forehead ROI highlights with alpha blend
        if show_rois:
            roi_layer = overlay.copy()

            if face_res.landmarks and len(face_res.landmarks) >= 468:
                lms = face_res.landmarks
                # Forehead center & radius
                p_top = lms[10]
                p_leb = lms[107]
                p_reb = lms[336]
                fh_cx = (p_leb[0] + p_reb[0]) // 2
                fh_cy = max(10, (p_top[1] + min(p_leb[1], p_reb[1])) // 2)
                fh_r = max(12, int(fw * 0.12))

                # Left Cheek center & radius
                p_leye_bottom = lms[145]
                lc_cx = max(10, int(lms[117][0] + lms[205][0]) // 2)
                lc_cy = min(h - 10, int(p_leye_bottom[1] + fh * 0.12))
                lc_r = max(14, int(fw * 0.11))

                # Right Cheek center & radius
                p_reye_bottom = lms[374]
                rc_cx = min(w - 10, int(lms[346][0] + lms[425][0]) // 2)
                rc_cy = min(h - 10, int(p_reye_bottom[1] + fh * 0.12))
                rc_r = max(14, int(fw * 0.11))

                # Draw translucent circular highlight discs
                for (cx, cy, r) in [(fh_cx, fh_cy, fh_r), (lc_cx, lc_cy, lc_r), (rc_cx, rc_cy, rc_r)]:
                    cv2.circle(roi_layer, (cx, cy), r, (180, 110, 0), -1, cv2.LINE_AA) # Deep blue/cyan fill
                    cv2.circle(roi_layer, (cx, cy), r, (254, 242, 0), 2, cv2.LINE_AA) # Glowing cyan border

            else:
                # Proportional circular fallback
                fh_cx = fx + fw // 2
                fh_cy = max(10, fy + int(0.22 * fh))
                fh_r = max(14, int(0.12 * fw))

                lc_cx = fx + int(0.28 * fw)
                lc_cy = fy + int(0.58 * fh)
                lc_r = max(14, int(0.11 * fw))

                rc_cx = fx + int(0.72 * fw)
                rc_cy = fy + int(0.58 * fh)
                rc_r = max(14, int(0.11 * fw))

                for (cx, cy, r) in [(fh_cx, fh_cy, fh_r), (lc_cx, lc_cy, lc_r), (rc_cx, rc_cy, rc_r)]:
                    cv2.circle(roi_layer, (cx, cy), r, (180, 110, 0), -1, cv2.LINE_AA)
                    cv2.circle(roi_layer, (cx, cy), r, (254, 242, 0), 2, cv2.LINE_AA)

            # Blend with 35% opacity for realistic medical-imaging HUD
            cv2.addWeighted(roi_layer, 0.40, overlay, 0.60, 0, overlay)

        return overlay
