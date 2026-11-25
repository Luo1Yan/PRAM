"""Edge centering, modality scaling, and joint cross-part shrinkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .config import PRAMConfig

if TYPE_CHECKING:
    from .model import ModalityInput


@dataclass(frozen=True)
class Blocks:
    """Endpoint self, endpoint-to-pivot, and pivot self blocks, grouped by part."""

    xx: Tensor
    xp: Tensor
    pp: Tensor

    def scaled(self, sx: Tensor, sp: Tensor) -> Blocks:
        return Blocks(self.xx / sx.square(), self.xp / (sx * sp), self.pp / sp.square())

    def shrunk(self, alpha: float) -> Blocks:
        if alpha == 0:
            return self

        def blend(block: Tensor) -> Tensor:
            return torch.lerp(block, block.mean(dim=0, keepdim=True), alpha)

        return Blocks(blend(self.xx), blend(self.xp), blend(self.pp))


@dataclass(frozen=True)
class EdgeStats:
    x_mean: Tensor
    p_mean: Tensor
    x_scale: Tensor
    p_scale: Tensor
    sample: Blocks
    regularized: Blocks
    x_broadcast: bool
    p_broadcast: bool


@dataclass(frozen=True)
class MCRGroup:
    """A zero part count identifies the separate global component."""

    parts: int
    a: EdgeStats
    b: EdgeStats


@dataclass(frozen=True)
class MCRStats:
    groups: list[MCRGroup]
    n_parts: int
    alpha: float
    single_vector_pivot: bool


@dataclass(frozen=True)
class _RawEdge:
    x_mean: Tensor
    p_mean: Tensor
    sample: Blocks
    variance: Blocks | None


def _center(vectors: Tensor) -> tuple[Tensor, Tensor]:
    vectors = vectors.to(dtype=torch.float64)
    origin = vectors[:, :1, :]
    mean = origin[:, 0, :] + (vectors - origin).mean(dim=1)
    return mean, vectors - mean.unsqueeze(1)


def _raw_edge(x: Tensor, p: Tensor, estimate_variance: bool) -> _RawEdge:
    xm, x = _center(x)
    pm, p = _center(p)
    n = x.size(1)
    sample = Blocks(x.mT @ x / n, x.mT @ p / n, p.mT @ p / n)
    variance = None
    if estimate_variance:
        x2 = x.square().sum(dim=2)
        p2 = p.square().sum(dim=2)

        def psi(left: Tensor, right: Tensor, block: Tensor) -> Tensor:
            fourth = (left * right).mean(dim=1)
            return ((fourth - block.square().sum(dim=(1, 2))) / n).clamp_min(0)

        variance = Blocks(
            psi(x2, x2, sample.xx),
            psi(x2, p2, sample.xp),
            psi(p2, p2, sample.pp),
        )
    return _RawEdge(xm, pm, sample, variance)


def _energy(self_blocks: Tensor) -> Tensor:
    return self_blocks.diagonal(dim1=1, dim2=2).mean()


def _shrinkage(entries: list[tuple[Tensor, Tensor, int]]) -> float:
    if not entries:
        return 1.0
    numerator = entries[0][0].new_zeros(())
    denominator = numerator.clone()
    for block, variance, weight in entries:
        # Subtracting a common origin preserves exact zero for identical parts.
        differences = block - block[:1]
        differences = differences - differences.mean(dim=0, keepdim=True)
        dispersion = differences.square().sum()
        numerator += torch.where(dispersion > 0, weight * variance.sum(), 0.0)
        denominator += weight * dispersion
    if denominator.item() == 0:
        return 1.0
    return (numerator / denominator).clamp(0, 1).item()


def _fit_group(
    parts: int,
    a: ModalityInput,
    pa: ModalityInput,
    pb: ModalityInput,
    b: ModalityInput,
    config: PRAMConfig,
) -> tuple[MCRGroup, float]:
    pa_vectors, pb_vectors = pa.vectors(parts), pb.vectors(parts)
    if pa_vectors.size(2) != pb_vectors.size(2):
        raise ValueError("Pivot feature widths must agree across fitting edges")
    estimate = parts > 0 and config.use_shrinkage
    raw_a = _raw_edge(a.vectors(parts), pa_vectors, estimate)
    raw_b = _raw_edge(b.vectors(parts), pb_vectors, estimate)
    floor = config.scale_floor
    sa = _energy(raw_a.sample.xx).clamp_min(floor).sqrt()
    sb = _energy(raw_b.sample.xx).clamp_min(floor).sqrt()
    pooled = len(a) * _energy(raw_a.sample.pp) + len(b) * _energy(raw_b.sample.pp)
    sp = (pooled / (len(a) + len(b))).clamp_min(floor).sqrt()

    entries = []
    edges = []
    for raw, sx, endpoint, pivot in ((raw_a, sa, a, pa), (raw_b, sb, b, pb)):
        sample = raw.sample.scaled(sx, sp)
        xb = parts > 0 and endpoint.local_feat is None
        ppb = parts > 0 and pivot.local_feat is None
        edges.append((raw, sx, sample, xb, ppb))
        if raw.variance is not None:
            variance = raw.variance.scaled(sx.square(), sp.square())
            if not xb:
                entries.append((sample.xx, variance.xx, 1))
            if not (xb and ppb):
                entries.append((sample.xp, variance.xp, 2))
            if not ppb:
                entries.append((sample.pp, variance.pp, 1))

    alpha = _shrinkage(entries) if estimate else 0.0
    fitted = [
        EdgeStats(raw.x_mean, raw.p_mean, sx, sp, sample, sample.shrunk(alpha), xb, ppb)
        for raw, sx, sample, xb, ppb in edges
    ]
    return MCRGroup(parts, fitted[0], fitted[1]), alpha


def fit_mcr(
    ap_a: ModalityInput,
    ap_p: ModalityInput,
    pb_p: ModalityInput,
    pb_b: ModalityInput,
    config: PRAMConfig,
) -> MCRStats:
    """Fit both edges with one local intensity and row-pooled pivot scales."""
    if len(ap_a) != len(ap_p) or len(pb_p) != len(pb_b):
        raise ValueError("Each fitting edge requires paired rows")
    if min(len(ap_a), len(pb_b)) < 1:
        raise ValueError("Both fitting edges require at least one row")
    grids = {
        x.local_feat.size(1)
        for x in (ap_a, ap_p, pb_p, pb_b)
        if config.use_local and x.local_feat is not None
    }
    if len(grids) > 1:
        raise ValueError("Local part counts must agree across modalities")
    n_parts = next(iter(grids), 0)
    global_group, _ = _fit_group(0, ap_a, ap_p, pb_p, pb_b, config)
    groups = [global_group]
    alpha = 0.0
    if n_parts:
        local_group, alpha = _fit_group(n_parts, ap_a, ap_p, pb_p, pb_b, config)
        groups.append(local_group)
    single_pivot = not (ap_p.multiple_parts or pb_p.multiple_parts)
    return MCRStats(groups, n_parts, alpha, single_pivot)
