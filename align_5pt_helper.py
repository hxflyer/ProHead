import os
import numpy as np
import cv2
from typing import Optional


class Align5PtHelper:
    """Fixed 5-point alignment utilities for geometry datasets."""

    def __init__(
        self,
        image_size: int,
        indices: np.ndarray,
        target_norm: np.ndarray,
        jitter_std: float,
        output_scale: float,
        direction_shift: float,
    ):
        self.image_size = int(image_size)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_norm = np.asarray(target_norm, dtype=np.float32)
        self.jitter_std = float(max(0.0, jitter_std))
        self.output_scale = float(np.clip(output_scale, 0.1, 2.0))
        self.direction_shift = float(np.clip(direction_shift, 0.0, 0.5))

    def extract_five_points_px(self, landmarks: Optional[np.ndarray], w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Extract 5 anchor points in pixel space from full landmark array."""
        pts = np.full((5, 2), np.nan, dtype=np.float32)
        valid = np.zeros((5,), dtype=bool)
        if landmarks is None:
            return pts, valid

        for i, lm_idx in enumerate(self.indices.tolist()):
            if lm_idx < 0 or lm_idx >= len(landmarks):
                continue
            uv = landmarks[lm_idx, 3:5]
            if not np.isfinite(uv).all():
                continue
            px = np.array([uv[0] * float(w), uv[1] * float(h)], dtype=np.float32)
            pts[i] = px
            valid[i] = True
        return pts, valid

    @staticmethod
    def compute_head_center_px(landmarks: Optional[np.ndarray], w: int, h: int) -> Optional[np.ndarray]:
        """Compute robust head center from full landmark bbox center (pixel space)."""
        if landmarks is None or len(landmarks) == 0:
            return None
        pts = landmarks[:, 3:5].astype(np.float32).copy()
        finite = np.isfinite(pts).all(axis=1)
        if not finite.any():
            return None
        pts = pts[finite]
        pts[:, 0] *= float(w)
        pts[:, 1] *= float(h)
        pmin = np.min(pts, axis=0)
        pmax = np.max(pts, axis=0)
        return 0.5 * (pmin + pmax)

    @staticmethod
    def is_extreme_pose(five_pts_px: np.ndarray, five_valid: np.ndarray) -> bool:
        """Heuristic extreme-pose detector (profile / top-down / bottom-up-like compression)."""
        pts = five_pts_px[five_valid]
        if len(pts) < 2:
            return True
        extent = float(np.max(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)))
        if extent < 1e-6:
            return True

        eye_ratio = 1.0
        if five_valid[0] and five_valid[1]:
            eye_dist = float(np.linalg.norm(five_pts_px[0] - five_pts_px[1]))
            eye_ratio = eye_dist / extent

        mouth_ratio = 1.0
        if five_valid[3] and five_valid[4]:
            mouth_dist = float(np.linalg.norm(five_pts_px[3] - five_pts_px[4]))
            mouth_ratio = mouth_dist / extent

        span = np.ptp(pts, axis=0)
        anis = float(min(span[0], span[1]) / max(max(span[0], span[1]), 1e-6))
        return (eye_ratio < 0.22) or (mouth_ratio < 0.22) or (anis < 0.18)

    @staticmethod
    def estimate_face_direction(five_pts_px: np.ndarray, five_valid: np.ndarray) -> np.ndarray:
        """Estimate 2D face direction from 5 points."""
        if five_pts_px.shape != (5, 2) or five_valid.shape != (5,):
            return np.zeros((2,), dtype=np.float32)
        if not five_valid[2]:
            return np.zeros((2,), dtype=np.float32)

        context_idx = [0, 1, 3, 4]
        context = []
        for i in context_idx:
            if five_valid[i]:
                context.append(five_pts_px[i])
        if len(context) == 0:
            return np.zeros((2,), dtype=np.float32)

        context = np.asarray(context, dtype=np.float32)
        center = np.mean(context, axis=0)
        nose = five_pts_px[2].astype(np.float32)

        valid_pts = five_pts_px[five_valid]
        if len(valid_pts) >= 2:
            extent = float(np.max(np.linalg.norm(valid_pts[:, None, :] - valid_pts[None, :, :], axis=2)))
        else:
            extent = 1.0
        extent = max(extent, 1.0)

        d = (nose - center) / extent
        d = np.clip(d, -1.0, 1.0)
        return d.astype(np.float32)

    def estimate_alignment_matrix(
        self,
        five_pts_px: np.ndarray,
        five_valid: np.ndarray,
        src_w: int,
        src_h: int,
        split: str,
        head_center_px: Optional[np.ndarray] = None,
        is_extreme_pose: bool = False,
        face_direction: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Estimate similarity transform from sparse anchors to canonical face pose."""
        def apply_center_scale(m_in: np.ndarray, scale: float) -> np.ndarray:
            c = np.array([0.5 * float(self.image_size), 0.5 * float(self.image_size)], dtype=np.float32)
            a = m_in[:, :2]
            b = m_in[:, 2]
            m_out = m_in.copy().astype(np.float32)
            m_out[:, :2] = scale * a
            m_out[:, 2] = scale * b + (1.0 - scale) * c
            return m_out

        def apply_head_recentering(m_in: np.ndarray) -> np.ndarray:
            if head_center_px is None or not np.isfinite(head_center_px).all():
                return m_in
            c = np.array([0.5 * float(self.image_size), 0.5 * float(self.image_size)], dtype=np.float32)
            cur = m_in[:, :2] @ head_center_px.astype(np.float32) + m_in[:, 2]
            shift = c - cur
            w_shift = 1.0 if is_extreme_pose else 0.35
            m_out = m_in.copy().astype(np.float32)
            m_out[:, 2] += w_shift * shift
            if face_direction is not None:
                fd = np.asarray(face_direction, dtype=np.float32)
                if fd.shape == (2,) and np.isfinite(fd).all():
                    m_out[:, 2] += self.direction_shift * float(self.image_size) * np.clip(fd, -1.0, 1.0)
            return m_out

        if five_pts_px.shape != (5, 2) or five_valid.shape != (5,):
            return None

        src_all = five_pts_px.astype(np.float32).copy()
        dst_all = (self.target_norm * float(self.image_size)).astype(np.float32)

        if five_valid[0] and five_valid[1] and src_all[0, 0] > src_all[1, 0]:
            src_all[[0, 1]] = src_all[[1, 0]]
        if five_valid[3] and five_valid[4] and src_all[3, 0] > src_all[4, 0]:
            src_all[[3, 4]] = src_all[[4, 3]]

        valid_idx = np.where(five_valid)[0]
        if len(valid_idx) == 0:
            sx = float(self.image_size) / float(max(1, src_w))
            sy = float(self.image_size) / float(max(1, src_h))
            m_no_kp = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float32)
            m_no_kp = apply_center_scale(m_no_kp, self.output_scale)
            return apply_head_recentering(m_no_kp)

        src = src_all[valid_idx].copy()
        dst = dst_all[valid_idx].copy()

        if five_valid[0] and five_valid[1]:
            eye_dist = float(np.linalg.norm(src_all[0] - src_all[1]))
            eye_floor = 0.06 * float(min(src_w, src_h))
            if eye_dist < eye_floor:
                kept = [i for i in valid_idx.tolist() if i not in (0, 1)]
                src_list = [src_all[i] for i in kept]
                dst_list = [dst_all[i] for i in kept]
                src_list.append(0.5 * (src_all[0] + src_all[1]))
                dst_list.append(0.5 * (dst_all[0] + dst_all[1]))
                src = np.asarray(src_list, dtype=np.float32)
                dst = np.asarray(dst_list, dtype=np.float32)

        if split == 'train' and self.jitter_std > 0.0:
            jitter_px = self.jitter_std * float(min(src_w, src_h))
            src += np.random.normal(0.0, jitter_px, size=src.shape).astype(np.float32)

        src_center = np.mean(src, axis=0)
        dst_center = np.mean(dst, axis=0)

        if len(src) >= 2:
            src_extent = float(np.max(np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)))
            dst_extent = float(np.max(np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=2)))
        else:
            src_extent = 0.0
            dst_extent = 0.0

        extent_floor = 0.18 * float(min(src_w, src_h))
        src_extent_safe = max(src_extent, extent_floor)
        if dst_extent <= 1e-6:
            scale_ref = float(self.image_size) / float(max(1, max(src_w, src_h)))
        else:
            scale_ref = dst_extent / src_extent_safe
        scale_ref = float(np.clip(scale_ref, 0.45, 1.8))

        m = None
        if len(src) >= 2:
            m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

        if m is not None and m.shape == (2, 3) and np.isfinite(m).all():
            a, b = float(m[0, 0]), float(m[0, 1])
            scale_est = float(np.sqrt(a * a + b * b))
            if scale_est < 1e-6:
                scale_final = scale_ref
                rot = np.eye(2, dtype=np.float32)
            else:
                rot = np.array([[a, b], [-b, a]], dtype=np.float32) / scale_est
                lower = max(0.45, 0.7 * scale_ref)
                upper = min(1.8, 1.35 * scale_ref)
                scale_final = float(np.clip(scale_est, lower, upper))

            t = dst_center - scale_final * (rot @ src_center)
            m_final = np.zeros((2, 3), dtype=np.float32)
            m_final[:, :2] = scale_final * rot
            m_final[:, 2] = t
            m_final = apply_center_scale(m_final, self.output_scale)
            return apply_head_recentering(m_final)

        m_final = np.array(
            [
                [scale_ref, 0.0, dst_center[0] - scale_ref * src_center[0]],
                [0.0, scale_ref, dst_center[1] - scale_ref * src_center[1]],
            ],
            dtype=np.float32,
        )
        m_final = apply_center_scale(m_final, self.output_scale)
        return apply_head_recentering(m_final)

    @staticmethod
    def transform_points_px(points_px: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Apply affine matrix to point array [N,2] in pixels."""
        if points_px is None:
            return None
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
        """Transform geometry 2D coords with affine; keep 3D coords unchanged."""
        if geom is None:
            return None
        out = geom.copy()
        pts_px = out[:, 3:5].astype(np.float32).copy()
        pts_px[:, 0] *= float(src_w)
        pts_px[:, 1] *= float(src_h)
        pts_px = self.transform_points_px(pts_px, m)
        out[:, 3] = pts_px[:, 0] / float(self.image_size)
        out[:, 4] = pts_px[:, 1] / float(self.image_size)
        return out
