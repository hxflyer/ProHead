"""
MediaPipe-based face alignment helper.
Rotation: eye→mouth axis → vertical.
Scale: eye_center to jaw (chin landmark 152) = FACE_SCALE_RATIO * image_size.
Translation: mean of 5 face landmarks → image center.
Fallback: if MediaPipe fails, use GT landmark file indices.
"""
import math
import os
from typing import Optional

os.environ.setdefault("GLOG_minloglevel", "3")  # suppress MediaPipe/TFLite noise

import numpy as np

# eye_center → bottom_jaw = 30% of image height after alignment
FACE_SCALE_RATIO = 0.40

# MediaPipe Face Landmarker 478-pt model indices (with iris)
_MP_RIGHT_EYE   = 468   # right iris center
_MP_LEFT_EYE    = 473   # left iris center
_MP_NOSE        = 1     # nose tip
_MP_RIGHT_MOUTH = 61    # right mouth corner
_MP_LEFT_MOUTH  = 291   # left mouth corner
_MP_JAW         = 152   # chin bottom

# GT landmark file indices for fallback
_GT_5PT_INDICES = (2219, 2194, 1993, 1896, 1839)  # eye_r, eye_l, nose, mouth_r, mouth_l
_GT_JAW_INDEX   = 1788

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "pretrained", "face_landmarker.task")

_mp_landmarker = None


def _get_landmarker():
    global _mp_landmarker
    if _mp_landmarker is None:
        import mediapipe as mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        _mp_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return _mp_landmarker


class Align5PtHelper:
    """MediaPipe-based face alignment with scale/rotation/translate."""

    def __init__(
        self,
        image_size: int,
        indices=None,           # unused, kept for compat
        jitter_std: float = 0.0,  # unused
        target_norm: Optional[np.ndarray] = None,  # unused, kept for compat
        output_scale: float = 1.0,  # unused, kept for compat
        direction_shift: float = 0.0,  # unused, kept for compat
        scale_jitter: float = 0.0,
        translate_jitter: float = 0.0,
        lm_jitter: float = 0.01,
        y_offset: float = 0.0,
    ):
        self.image_size = int(image_size)
        self.target_norm = None if target_norm is None else np.asarray(target_norm, dtype=np.float32)
        self.output_scale = float(output_scale)
        self.direction_shift = float(direction_shift)
        self.scale_jitter = float(max(0.0, scale_jitter))
        self.translate_jitter = float(max(0.0, translate_jitter))
        self.lm_jitter = float(max(0.0, lm_jitter))
        self.y_offset = float(y_offset)

    # ------------------------------------------------------------------
    # MediaPipe detection → 6 key points, with GT fallback
    # ------------------------------------------------------------------

    def detect_landmarks(
        self,
        image_rgb: np.ndarray,
        fallback_lm_px: Optional[np.ndarray] = None,
    ) -> tuple[Optional[np.ndarray], str]:
        """Run MediaPipe face landmarker on an RGB image.
        Returns (lm6, source) where:
          lm6: [6, 2] float32 pixel coords [right_eye, left_eye, nose, right_mouth, left_mouth, jaw]
               or None if both MediaPipe and fallback fail.
          source: 'mediapipe', 'landmark_gt', or 'none'
        fallback_lm_px: [N, 2] float32 GT pixel coords used when MediaPipe fails.
        """
        # Prefer GT landmark file
        if fallback_lm_px is not None:
            indices = list(_GT_5PT_INDICES) + [_GT_JAW_INDEX]
            if max(indices) < len(fallback_lm_px):
                lm6 = fallback_lm_px[indices].astype(np.float32)
                return lm6, "landmark_gt"

        # Fallback to MediaPipe
        import mediapipe as mp
        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = _get_landmarker().detect(mp_image)
        if result.face_landmarks:
            lm = result.face_landmarks[0]

            def pt(idx):
                return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

            lm6 = np.stack([
                pt(_MP_RIGHT_EYE), pt(_MP_LEFT_EYE), pt(_MP_NOSE),
                pt(_MP_RIGHT_MOUTH), pt(_MP_LEFT_MOUTH), pt(_MP_JAW),
            ], axis=0)
            return lm6, "mediapipe"

        return None, "none"

    # ------------------------------------------------------------------
    # 5 key points for output (drop jaw, keep first 5)
    # ------------------------------------------------------------------

    def extract_key5_from_lm68(self, lm6: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Return (pts [5,2], valid [5]) — first 5 of the 6-pt detection result."""
        if lm6 is None:
            return np.zeros((5, 2), dtype=np.float32), np.zeros(5, dtype=np.float32)
        return lm6[:5].astype(np.float32), np.ones(5, dtype=np.float32)

    # ------------------------------------------------------------------
    # Alignment matrix
    # ------------------------------------------------------------------

    def estimate_alignment_matrix(
        self,
        lm68: Optional[np.ndarray],   # [6, 2] from detect_landmarks
        src_w: int,
        src_h: int,
        split: str,
    ) -> np.ndarray:
        is_train = split == 'train'
        image_size = self.image_size
        c = np.array([image_size * 0.5, image_size * 0.5], dtype=np.float32)
        lm6 = lm68

        if lm6 is not None:
            if is_train and self.lm_jitter > 0.0:
                lm6 = lm6 + np.random.normal(0.0, self.lm_jitter * max(src_w, src_h), lm6.shape).astype(np.float32)
            eye_center   = (lm6[0] + lm6[1]) * 0.5
            mouth_center = (lm6[3] + lm6[4]) * 0.5
            jaw          = lm6[5]

            # Rotation: align eye→mouth to vertical
            dx = float(mouth_center[0] - eye_center[0])
            dy = float(mouth_center[1] - eye_center[1])
            dist_em = max(math.hypot(dx, dy), 1.0)
            cos_a = dy / dist_em
            sin_a = dx / dist_em

            # Scale: project jaw onto eye→mouth axis for true vertical distance
            dxj = float(jaw[0] - eye_center[0])
            dyj = float(jaw[1] - eye_center[1])
            dist_jaw = max(dxj * sin_a + dyj * cos_a, 1.0)
            s = FACE_SCALE_RATIO * float(image_size) / dist_jaw

            # Translation: x = -(nose - (eye_center + mouth_center)), y = (eye_center + jaw) / 2 + 100
            nose = lm6[2]
            src_center = np.array([
                eye_center[0] + mouth_center[0] - nose[0],
                (eye_center[1] + jaw[1]) * 0.5 - 100.0,
            ], dtype=np.float32)
        else:
            cos_a, sin_a = 1.0, 0.0
            s = float(image_size) / float(max(src_w, src_h))
            src_center = np.array([src_w * 0.5, src_h * 0.5], dtype=np.float32)

        r00, r01 = s * cos_a, -s * sin_a
        r10, r11 = s * sin_a,  s * cos_a
        tx = c[0] - (r00 * src_center[0] + r01 * src_center[1])
        ty = c[1] - (r10 * src_center[0] + r11 * src_center[1]) + self.y_offset
        m = np.array([[r00, r01, tx], [r10, r11, ty]], dtype=np.float32)

        if is_train:
            if self.scale_jitter > 0.0:
                js = float(np.clip(1.0 + np.random.normal(0.0, self.scale_jitter), 0.8, 1.2))
                m[:, :2] *= js
                m[:, 2] = js * m[:, 2] + (1.0 - js) * c
            if self.translate_jitter > 0.0:
                m[:, 2] += np.random.normal(0.0, self.translate_jitter * image_size, 2).astype(np.float32)

        return m

    # ------------------------------------------------------------------
    # Point / geometry transform helpers (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def transform_points_px(points_px: np.ndarray, m: np.ndarray) -> np.ndarray:
        out = points_px.copy().astype(np.float32)
        finite_mask = np.isfinite(out).all(axis=1)
        if not finite_mask.any():
            return out
        ones = np.ones((int(finite_mask.sum()), 1), dtype=np.float32)
        pts_h = np.concatenate([out[finite_mask], ones], axis=1)
        out[finite_mask] = pts_h @ m.T
        return out

    def apply_alignment_to_geometry(
        self,
        geom: Optional[np.ndarray],
        m: np.ndarray,
        src_w: int,
        src_h: int,
    ) -> Optional[np.ndarray]:
        if geom is None:
            return None
        out = geom.copy()
        pts_px = out[:, 3:5].astype(np.float32).copy()
        valid_mask = np.isfinite(pts_px).all(axis=1) & np.all(pts_px >= 0.0, axis=1)
        if valid_mask.any():
            pts_valid_px = pts_px[valid_mask]
            pts_valid_px[:, 0] *= float(src_w)
            pts_valid_px[:, 1] *= float(src_h)
            pts_valid_px = self.transform_points_px(pts_valid_px, m)
            out[valid_mask, 3] = pts_valid_px[:, 0] / float(self.image_size)
            out[valid_mask, 4] = pts_valid_px[:, 1] / float(self.image_size)
        return out
