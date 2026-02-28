import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimDRLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss for SimDR with out-of-range diagnostics.
    Target coordinates are not clamped.
    """

    def __init__(
        self,
        k_bins: int = 256,
        sigma: float = 2.0,
        min_3d: float = -1.0,
        max_3d: float = 1.0,
        min_2d: float = -1.0,
        max_2d: float = 1.0,
    ):
        super().__init__()
        self.k_bins = int(k_bins)
        self.sigma = float(sigma)

        self.register_buffer("min_3d", torch.tensor(float(min_3d)))
        self.register_buffer("max_3d", torch.tensor(float(max_3d)))
        self.register_buffer("min_2d", torch.tensor(float(min_2d)))
        self.register_buffer("max_2d", torch.tensor(float(max_2d)))

        # Last batch out-of-range ratio per dimension [x, y, z, u, v]
        self.last_oob_ratio = None

    def generate_target_simdr(self, coords: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        norm_coords = (coords - min_val) / (max_val - min_val) * (self.k_bins - 1)
        x = torch.arange(self.k_bins, device=coords.device).float()
        mu = norm_coords.unsqueeze(-1)
        logits = -((x - mu) ** 2) / (2 * self.sigma ** 2)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def measure_oob(coords: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        return ((coords < min_val) | (coords > max_val)).float().mean()

    def forward(self, pred_logits: torch.Tensor, target_coords: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d, k = pred_logits.shape
        if d != 5:
            raise ValueError(f"SimDRLoss expects 5 dimensions, got {d}.")

        targets_list = []
        oob_list = []

        for i in range(3):
            coords_i = target_coords[..., i]
            oob_list.append(self.measure_oob(coords_i, self.min_3d, self.max_3d))
            targets_list.append(self.generate_target_simdr(coords_i, self.min_3d, self.max_3d))

        for i in range(3, 5):
            coords_i = target_coords[..., i]
            oob_list.append(self.measure_oob(coords_i, self.min_2d, self.max_2d))
            targets_list.append(self.generate_target_simdr(coords_i, self.min_2d, self.max_2d))

        target_dist = torch.stack(targets_list, dim=2)
        self.last_oob_ratio = torch.stack(oob_list).detach()

        log_probs = F.log_softmax(pred_logits, dim=-1)
        loss = F.kl_div(log_probs, target_dist, reduction="none", log_target=False)
        loss = loss.sum(dim=-1)

        if weights is not None:
            loss = loss * weights
            return loss.sum() / (weights.sum() + 1e-6)
        return loss.mean()


class WingLoss(nn.Module):
    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = float(w)
        self.epsilon = float(epsilon)
        self.C = self.w - self.w * math.log(1 + self.w / self.epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        scale = 512.0
        y_pred = pred * scale
        y_true = target * scale

        diff = (y_pred - y_true).abs()
        idx_small = diff < self.w
        idx_big = diff >= self.w

        loss = torch.zeros_like(diff)
        loss[idx_small] = self.w * torch.log(1 + diff[idx_small] / self.epsilon)
        loss[idx_big] = diff[idx_big] - self.C

        if weights is not None:
            loss = loss * weights
            return loss.sum() / (weights.sum() + 1e-6)
        return loss.mean()


def compute_weighted_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    l1 = (pred - target).abs().float()
    if weights is not None:
        w = weights.to(device=l1.device, dtype=torch.float32)
        return (l1 * w).sum(dtype=torch.float32) / (w.sum(dtype=torch.float32) + 1e-6)
    return l1.mean()


def is_finite_tensor(t: torch.Tensor | None) -> bool:
    if t is None:
        return True
    return bool(torch.isfinite(t).all().item())


def model_parameters_are_finite(module: nn.Module) -> bool:
    m = module.module if hasattr(module, "module") else module
    for p in m.parameters():
        if p is not None and (not torch.isfinite(p).all()):
            return False
    return True


@dataclass
class _MetricState:
    total: float = 0.0
    count: float = 0.0


class MetricAccumulator:
    """Tracks weighted sums/counts and returns robust means."""

    def __init__(self):
        self._states: dict[str, _MetricState] = {}

    @staticmethod
    def _to_float(v) -> float:
        if isinstance(v, torch.Tensor):
            return float(v.detach().item()) if v.numel() == 1 else float(v.detach().sum().item())
        return float(v)

    def update_sum_count(self, name: str, sum_value, count_value) -> None:
        s = self._to_float(sum_value)
        c = self._to_float(count_value)
        st = self._states.setdefault(name, _MetricState())
        st.total += s
        st.count += c

    def get_sum(self, name: str) -> float:
        st = self._states.get(name)
        return 0.0 if st is None else st.total

    def get_count(self, name: str) -> float:
        st = self._states.get(name)
        return 0.0 if st is None else st.count

    def mean(self, name: str) -> float | None:
        st = self._states.get(name)
        if st is None or st.count <= 0:
            return None
        return st.total / max(st.count, 1e-6)

    def has(self, name: str) -> bool:
        return self.get_count(name) > 0

