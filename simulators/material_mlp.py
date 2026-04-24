"""Per-point material and grip MLPs, ported from PhysTwin/stage2.

- MaterialMLP: position -> (E, nu), sigmoid-bounded
- GripMLP:     position -> [0, 1] scalar per-point grip weight
- WeightedBoundary: Kaolin Boundary subclass with per-pinned-index weights
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from kaolin.physics.utils import scene_forces as _scene_forces


MATERIAL_BOUNDS = {"E": (1e3, 1e7), "nu": (0.35, 0.49)}


class MaterialMLP(nn.Module):
    def __init__(self, initial_E: float, initial_nu: float,
                 hidden_dim: int = 64, num_layers: int = 3, bounds: dict | None = None):
        super().__init__()
        self.bounds = bounds or MATERIAL_BOUNDS
        self.log_E_min = float(np.log(self.bounds["E"][0]))
        self.log_E_max = float(np.log(self.bounds["E"][1]))
        nu_min, nu_max = self.bounds["nu"]

        layers = []
        in_dim = 3
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 2)

        # Bias init so output ≈ (initial_E, initial_nu) at x=0 (pre-training).
        target_log = float(np.clip(np.log(initial_E), self.log_E_min + 1e-4, self.log_E_max - 1e-4))
        norm_E = (target_log - self.log_E_min) / (self.log_E_max - self.log_E_min)
        norm_E = float(np.clip(norm_E, 0.01, 0.99))
        nu_clipped = float(np.clip(initial_nu, nu_min + 1e-4, nu_max - 1e-4))
        norm_nu = (nu_clipped - nu_min) / (nu_max - nu_min)
        norm_nu = float(np.clip(norm_nu, 0.01, 0.99))
        with torch.no_grad():
            self.head.bias[0] = float(np.log(norm_E / (1.0 - norm_E)))
            self.head.bias[1] = float(np.log(norm_nu / (1.0 - norm_nu)))

    def forward(self, x: torch.Tensor):
        feats = self.net(x)
        raw = self.head(feats)
        log_E = self.log_E_min + (self.log_E_max - self.log_E_min) * torch.sigmoid(raw[:, 0])
        E = torch.exp(log_E)
        nu_min, nu_max = self.bounds["nu"]
        nu = nu_min + (nu_max - nu_min) * torch.sigmoid(raw[:, 1])
        return E, nu


class GripMLP(nn.Module):
    """Position -> [0, 1] per-point grip weight. Multiplied by a base penalty."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WeightedBoundary(_scene_forces.Boundary):
    """Boundary penalty with per-pinned-index weight.

    Stores a weight tensor aligned with pinned_indices. `_energy` multiplies
    each pinned point's squared-distance penalty by its weight. Gradient and
    hessian are computed via autograd on _energy, so the weights propagate.
    """
    def __init__(self):
        super().__init__()
        self.pinned_weights = None  # (num_pinned,)

    def set_pinned_weights(self, weights: torch.Tensor):
        self.pinned_weights = weights

    def _energy(self, x):
        pt_wise_en = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if self.pinned_indices is None:
            return pt_wise_en
        sq = torch.sum((x[self.pinned_indices] - self.pinned_vertices) ** 2, dim=1)
        if self.pinned_weights is not None:
            sq = sq * self.pinned_weights
        pt_wise_en[self.pinned_indices] = sq
        return pt_wise_en
