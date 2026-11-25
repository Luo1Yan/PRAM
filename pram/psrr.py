"""Equal-edge spectral composition and Gaussian matching coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import PRAMConfig
from .mcr import MCRStats


def regularized_power(matrix: Tensor, power: float, config: PRAMConfig) -> Tensor:
    """Apply a spectral power after PSD projection and relative ridge addition."""
    matrix = matrix.to(dtype=torch.float64)
    matrix = 0.5 * (matrix + matrix.mT)
    trace_mean = matrix.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    ridge = config.relative_ridge * trace_mean.clamp_min(config.scale_floor)
    eigenvalues, vectors = torch.linalg.eigh(matrix)
    shifted = eigenvalues.clamp_min(0) + ridge.unsqueeze(-1)
    factors = torch.where(shifted > 0, shifted.pow(power), 0.0)
    return (vectors * factors.unsqueeze(-2)) @ vectors.mT


@dataclass(frozen=True)
class SpectralGroup:
    group_index: int
    projection_a: Tensor
    projection_b: Tensor
    sigma: Tensor

    def features(
        self, centered_scaled: Tensor, side: str, weight: float, config: PRAMConfig
    ) -> tuple[Tensor, Tensor]:
        """Factor the pairwise LLR into a dot product and endpoint offsets."""
        projection = self.projection_a if side == "a" else self.projection_b
        coordinates = centered_scaled @ projection
        c = self.sigma.clamp(0, config.corr_cap)
        linear = weight * c / (1.0 - c.square())
        features = coordinates * linear.sqrt().unsqueeze(1)
        offset = coordinates.new_zeros(coordinates.size(1))
        if config.use_quadratic:
            penalty = (0.5 * c * linear).unsqueeze(1)
            offset = -(coordinates.square() * penalty).sum(dim=(0, 2))
        return features.transpose(0, 1).flatten(1), offset


@dataclass(frozen=True)
class PSRRState:
    groups: list[SpectralGroup]


def fit_psrr(mcr: MCRStats, n_states: int, config: PRAMConfig) -> PSRRState:
    """Retain leading paired directions and their unclipped singular values."""
    groups = []
    for index, group in enumerate(mcr.groups):
        if group.parts == 0 and mcr.n_parts > 0 and not config.use_global:
            continue
        a, b = group.a.regularized, group.b.regularized
        pivot = 0.5 * (a.pp + b.pp)
        kernel = a.xp @ regularized_power(pivot, -1.0, config) @ b.xp.mT
        wa = regularized_power(a.xx, -0.5, config)
        wb = regularized_power(b.xx, -0.5, config)
        u, sigma, vh = torch.linalg.svd(wa @ kernel @ wb, full_matrices=False)
        k = min(config.max_directions, sigma.size(1))
        if mcr.single_vector_pivot:
            k = min(k, max(1, n_states - 1))
        groups.append(SpectralGroup(
            index, wa @ u[:, :, :k], wb @ vh[:, :k, :].mT, sigma[:, :k].contiguous()
        ))
    return PSRRState(groups)
