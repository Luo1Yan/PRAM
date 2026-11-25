"""Base scores, state posteriors, reference distributions, and task readouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import PRAMConfig, sampled_indices
from .mcr import EdgeStats, MCRStats
from .psrr import PSRRState, regularized_power

if TYPE_CHECKING:
    from .model import ModalityInput


def _state_ids(x: Tensor, decimals: int) -> tuple[Tensor, Tensor]:
    rounded = torch.round(x.to(dtype=torch.float64), decimals=decimals)
    return torch.unique(rounded, sorted=True, return_inverse=True, dim=0)


def pivot_partition(
    ap_p: ModalityInput, pb_p: ModalityInput, config: PRAMConfig
) -> tuple[Tensor, Tensor, Tensor]:
    x = torch.cat((ap_p.global_feat, pb_p.global_feat))
    states, ids = _state_ids(x, config.state_decimals)
    return states, ids[:len(ap_p)], ids[len(ap_p):]


def _repeat_index(m: int, n: int) -> float:
    return max(0.0, 1.0 - math.log(m) / math.log(n)) if n >= 3 else 0.0


def _folds(n: int, seed: int, device: torch.device):
    order = sampled_indices(n, n, device, seed)
    split = (n + 1) // 2
    left, right = order[:split], order[split:]
    return ((left, right), (right, left))


def _descriptors(x: ModalityInput, mean_local: bool) -> list[Tensor]:
    result = [x.global_feat]
    if mean_local:
        local = x.global_feat if x.local_feat is None else x.local_feat.mean(dim=1)
        result.append(local)
    return result


@dataclass(frozen=True)
class LDAVoter:
    center: Tensor
    scale: Tensor
    coefficient: Tensor
    intercept: Tensor

    def log_posterior(self, x: Tensor) -> Tensor:
        logits = ((x - self.center) / self.scale) @ self.coefficient + self.intercept
        return logits.log_softmax(dim=1)


def _fit_voter(x: Tensor, y: Tensor, counts: Tensor, config: PRAMConfig) -> LDAVoter:
    x = x.to(dtype=torch.float64)
    n, d = x.shape
    center = x[0] + (x - x[0]).mean(dim=0)
    z = x - center
    scale = z.square().mean().clamp_min(config.scale_floor).sqrt()
    z = z / scale
    counts = counts.to(dtype=z.dtype)
    log_prior = (counts / n).log()
    total_energy = z.square().sum()
    if counts.numel() == 1 or total_energy.item() == 0:
        return LDAVoter(center, scale, z.new_zeros(d, len(counts)), log_prior)

    means = z.new_zeros(len(counts), d)
    means.index_add_(0, y, z)
    means /= counts.unsqueeze(1)
    residual = z - means[y]
    if len(counts) >= n - 1 or residual.square().sum() <= 1e-12 * total_energy:
        residual = z
    covariance = residual.T @ residual / n
    theta = covariance.diagonal().mean()
    identity = torch.eye(d, device=x.device, dtype=x.dtype)
    dispersion = (covariance - theta * identity).square().sum()
    variance = (
        residual.square().sum(dim=1).square().mean() - covariance.square().sum()
    ) / n
    beta = torch.where(dispersion > 0, (variance / dispersion).clamp(0, 1), 1.0)
    regularized = (1.0 - beta) * covariance + (beta + config.lda_diag_eps) * theta * identity
    coefficient = torch.linalg.solve(regularized, means.T)
    intercept = log_prior - 0.5 * (means * coefficient.T).sum(dim=1)
    return LDAVoter(center, scale, coefficient, intercept)


@dataclass(frozen=True)
class PosteriorPool:
    classes: Tensor
    voters: list[LDAVoter]
    weights: Tensor

    def log_posterior(self, x: ModalityInput) -> Tensor:
        descriptors = _descriptors(x, len(self.voters) > 1)
        logs = [v.log_posterior(d) for v, d in zip(self.voters, descriptors)]
        if len(logs) == 1:
            return logs[0]
        pooled = (torch.stack(logs, dim=1) * self.weights.view(1, -1, 1)).sum(dim=1)
        return pooled.log_softmax(dim=1)

    def predict(self, x: ModalityInput) -> Tensor:
        return self.classes[self.log_posterior(x).argmax(dim=1)]


def _pool_weights(
    descriptors: list[Tensor], y: Tensor, n_classes: int, config: PRAMConfig
) -> Tensor:
    nv = len(descriptors)
    equal = descriptors[0].new_full((nv,), 1.0 / nv, dtype=torch.float64)
    if nv == 1 or n_classes == 1 or len(y) < 2:
        return equal

    held_out = []
    for train, test in _folds(len(y), 0, y.device):
        present, local_y, counts = torch.unique(
            y[train], sorted=True, return_inverse=True, return_counts=True
        )
        lookup = y.new_full((n_classes,), -1)
        lookup[present] = torch.arange(len(present), device=y.device)
        targets = lookup[y[test]]
        eligible = targets >= 0
        rows = test[eligible]
        if rows.numel():
            logs = [
                _fit_voter(d[train], local_y, counts, config).log_posterior(d[rows])
                for d in descriptors
            ]
            held_out.append((torch.stack(logs, dim=1), targets[eligible]))
    if not held_out:
        return equal

    n_eligible = sum(target.numel() for _, target in held_out)
    with torch.enable_grad():
        logits = torch.zeros_like(equal, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [logits], max_iter=config.lbfgs_max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            weights = logits.softmax(dim=0).view(1, -1, 1)
            loss = sum(
                F.cross_entropy((logs * weights).sum(dim=1), target, reduction="sum")
                for logs, target in held_out
            ) / n_eligible
            loss.backward()
            return loss

        optimizer.step(closure)
        return logits.detach().softmax(dim=0)


def _fit_pool(x: ModalityInput, labels: Tensor, config: PRAMConfig) -> PosteriorPool:
    classes, y, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    descriptors = _descriptors(x, config.use_local and x.multiple_parts)
    weights = _pool_weights(descriptors, y, len(classes), config)
    voters = [_fit_voter(d, y, counts, config) for d in descriptors]
    return PosteriorPool(classes, voters, weights)


@dataclass(frozen=True)
class _Support:
    ids: Tensor
    rows_a: Tensor
    rows_b: Tensor
    log_prior: Tensor


def _shared_support(ya: Tensor, yb: Tensor, n_states: int) -> _Support:
    ca = torch.bincount(ya, minlength=n_states)
    cb = torch.bincount(yb, minlength=n_states)
    shared = (ca > 0) & (cb > 0)
    ids = shared.nonzero(as_tuple=True)[0]
    rows_a = shared[ya].nonzero(as_tuple=True)[0]
    rows_b = shared[yb].nonzero(as_tuple=True)[0]
    frequencies = (ca[ids] + cb[ids]).to(dtype=torch.float64)
    log_prior = (frequencies / frequencies.sum()).log() if ids.numel() else frequencies
    return _Support(ids, rows_a, rows_b, log_prior)


def _compatibility(a: Tensor, b: Tensor, log_prior: Tensor) -> Tensor:
    """Evaluate log-sum-exp in tiles bounded in query, candidate, and state product."""
    b = b - log_prior
    max_pairs = max(1, (1 << 20) // log_prior.numel())
    if len(a) * len(b) <= max_pairs:
        return torch.logsumexp(a.unsqueeze(1) + b.unsqueeze(0), dim=2)
    result = a.new_empty(len(a), len(b))
    cols = min(len(b), math.isqrt(max_pairs))
    rows = max_pairs // cols
    for i in range(0, len(a), rows):
        left = a[i:i + rows].unsqueeze(1)
        for j in range(0, len(b), cols):
            right = b[j:j + cols].unsqueeze(0)
            result[i:i + rows, j:j + cols] = torch.logsumexp(left + right, dim=2)
    return result


@dataclass(frozen=True)
class Moments:
    mean: Tensor
    std: Tensor


def _moments(values: Tensor, dim: int, floor: float) -> Moments:
    variance, mean = torch.var_mean(values, dim=dim, unbiased=values.size(dim) > 1)
    return Moments(mean, variance.clamp_min(0).sqrt().clamp_min(floor))


@dataclass(frozen=True)
class Association:
    a: PosteriorPool
    b: PosteriorPool
    log_prior: Tensor
    moments: Moments


def _fit_association(
    a: ModalityInput,
    b: ModalityInput,
    ya: Tensor,
    yb: Tensor,
    support: _Support,
    bank_a: ModalityInput,
    bank_b: ModalityInput,
    config: PRAMConfig,
) -> Association:
    pool_a = _fit_pool(a.take(support.rows_a), ya[support.rows_a], config)
    pool_b = _fit_pool(b.take(support.rows_b), yb[support.rows_b], config)
    if config.use_standardization:
        scores = _compatibility(
            pool_a.log_posterior(bank_a), pool_b.log_posterior(bank_b), support.log_prior
        )
        moments = _moments(scores.flatten(), 0, config.scale_floor)
    else:
        moments = Moments(support.log_prior.new_zeros(()), support.log_prior.new_ones(()))
    return Association(pool_a, pool_b, support.log_prior, moments)


def _class_semantics(
    a: ModalityInput,
    pa: ModalityInput,
    pb: ModalityInput,
    b: ModalityInput,
    bank_a: ModalityInput,
    bank_b: ModalityInput,
    config: PRAMConfig,
) -> tuple[float, float, float, Association | None]:
    states, ya = _state_ids(a.global_feat, config.state_decimals)
    m, n = len(states), len(a)
    w = _repeat_index(m, n)
    if n < 3 or m < 2 or w == 0:
        return w, 0.0, 0.0, None

    correct = 0
    for train, test in _folds(n, config.class_prompt_cv_seed, ya.device):
        pool = _fit_pool(pa.take(train), ya[train], config)
        predictions = pool.predict(pa.take(test))
        correct += (predictions == ya[test]).sum().item()
    chance = 1.0 / m
    chi = min(1.0, max(0.0, (correct / n - chance) / (1.0 - chance)))
    if chi == 0:
        return w, chi, 0.0, None

    yb = _fit_pool(pa, ya, config).predict(pb)
    support = _shared_support(ya, yb, m)
    if support.ids.numel() == 0:
        return w, chi, 0.0, None
    route = _fit_association(a, b, ya, yb, support, bank_a, bank_b, config)
    return w, chi, w * chi, route


@dataclass(frozen=True)
class CovarianceMaps:
    a: Tensor
    b: Tensor


def _covariance_map(edge: EdgeStats, config: PRAMConfig) -> Tensor:
    raw_pp = edge.sample.pp * edge.p_scale.square()
    raw_xp = edge.sample.xp * (edge.x_scale * edge.p_scale)
    return raw_xp @ regularized_power(raw_pp, -0.5, config)


@dataclass(frozen=True)
class BaseFeatures:
    vector: Tensor
    offset: Tensor | None


def _base_features(
    x: ModalityInput,
    side: str,
    mcr: MCRStats,
    psrr: PSRRState,
    maps: list[CovarianceMaps],
    omega: float,
    config: PRAMConfig,
) -> list[BaseFeatures]:
    correlation = []
    covariance = []
    offset = x.global_feat.new_zeros(len(x))
    component_count = sum(group.sigma.size(0) for group in psrr.groups)
    for index, spectral in enumerate(psrr.groups):
        group = mcr.groups[spectral.group_index]
        edge = group.a if side == "a" else group.b
        vectors = x.vectors(group.parts)
        if vectors.size(2) != edge.x_mean.size(1):
            raise ValueError(f"Feature width differs from the fitted width for endpoint {side}")
        centered = vectors - edge.x_mean.unsqueeze(1)
        features, bias = spectral.features(
            centered / edge.x_scale, side, omega if group.parts else 1.0, config
        )
        correlation.append(features)
        offset = offset + bias
        if maps:
            projection = maps[index].a if side == "a" else maps[index].b
            mapped = centered @ projection
            norm = torch.linalg.vector_norm(mapped, dim=2, keepdim=True)
            unit = mapped / norm.clamp_min(config.scale_floor)
            covariance.append(unit.transpose(0, 1).flatten(1))
    result = [BaseFeatures(torch.cat(correlation, dim=1), offset)]
    if covariance:
        vector = torch.cat(covariance, dim=1) / math.sqrt(component_count)
        result.append(BaseFeatures(vector, None))
    return result


@dataclass(frozen=True)
class MSFState:
    m: int
    shared_states: Tensor
    omega: float
    w: float
    o: float
    w_A: float
    chi_A: float
    tau_A: float
    covariance_maps: list[CovarianceMaps]
    pivot_route: Association | None
    class_route: Association | None
    bank_indices_a: Tensor
    bank_indices_b: Tensor
    bank_a: list[BaseFeatures]
    bank_b: list[BaseFeatures]


def fit_msf(
    ap_a: ModalityInput,
    ap_p: ModalityInput,
    pb_p: ModalityInput,
    pb_b: ModalityInput,
    partition: tuple[Tensor, Tensor, Tensor],
    mcr: MCRStats,
    psrr: PSRRState,
    config: PRAMConfig,
    is_class_prompt: bool,
) -> MSFState:
    states, ya, yb = partition
    support = _shared_support(ya, yb, len(states))
    w = _repeat_index(len(states), len(ya) + len(yb))
    o = 0.5 * (len(support.rows_a) / len(ya) + len(support.rows_b) / len(yb))
    multi = sum((ap_a.multiple_parts, ap_p.multiple_parts or pb_p.multiple_parts,
                 pb_b.multiple_parts))
    omega = 1.0 / mcr.n_parts if mcr.n_parts and multi == 1 else 1.0

    maps = []
    if w > 0 and config.use_covariance_score:
        for spectral in psrr.groups:
            group = mcr.groups[spectral.group_index]
            maps.append(CovarianceMaps(
                _covariance_map(group.a, config), _covariance_map(group.b, config)
            ))

    ia = sampled_indices(len(ya), config.max_bank_size, ya.device)
    ib = sampled_indices(len(yb), config.max_bank_size, yb.device)
    bank_a, bank_b = ap_a.take(ia), pb_b.take(ib)
    pivot_route = None
    if o > 0 and config.use_pivot_associations:
        pivot_route = _fit_association(
            ap_a, pb_b, ya, yb, support, bank_a, bank_b, config
        )
    w_a = chi_a = tau_a = 0.0
    class_route = None
    if is_class_prompt and config.use_class_semantics:
        w_a, chi_a, tau_a, class_route = _class_semantics(
            ap_a, ap_p, pb_p, pb_b, bank_a, bank_b, config
        )

    base_a, base_b = [], []
    if config.use_standardization:
        base_a = _base_features(bank_a, "a", mcr, psrr, maps, omega, config)
        base_b = _base_features(bank_b, "b", mcr, psrr, maps, omega, config)
    return MSFState(
        len(states), states[support.ids], omega, w, o, w_a, chi_a, tau_a,
        maps, pivot_route, class_route, ia, ib, base_a, base_b
    )


def _dot(a: BaseFeatures, b: BaseFeatures, i: slice, j: slice) -> Tensor:
    scores = a.vector[i] @ b.vector[j].T
    if a.offset is not None and b.offset is not None:
        scores += a.offset[i].unsqueeze(1)
        scores += b.offset[j].unsqueeze(0)
    return scores


def _conditional(
    fixed: BaseFeatures,
    bank: BaseFeatures,
    left: bool,
    chunk_size: int | None,
    floor: float,
) -> Moments:
    n = len(fixed.vector)
    step = min(chunk_size or n, max(1, (1 << 20) // len(bank.vector)))
    means, stds = [], []
    for start in range(0, n, step):
        rows = slice(start, start + step)
        if left:
            values = _dot(fixed, bank, rows, slice(None))
            moments = _moments(values, 1, floor)
        else:
            values = _dot(bank, fixed, slice(None), rows)
            moments = _moments(values, 0, floor)
        means.append(moments.mean)
        stds.append(moments.std)
    return Moments(torch.cat(means), torch.cat(stds))


@dataclass(frozen=True)
class _BaseTerm:
    a: BaseFeatures
    b: BaseFeatures
    ma: Moments | None
    mb: Moments | None
    weight: float

    def score(self, i: slice, j: slice) -> Tensor:
        raw = _dot(self.a, self.b, i, j)
        if self.ma is None or self.mb is None:
            return raw
        left = (raw - self.ma.mean[i, None]) / self.ma.std[i, None]
        right = (raw - self.mb.mean[j]) / self.mb.std[j]
        return 0.5 * (left + right)


def score_msf(
    a: ModalityInput,
    b: ModalityInput,
    mcr: MCRStats,
    psrr: PSRRState,
    msf: MSFState,
    config: PRAMConfig,
    task: str,
    chunk_size: int | None,
) -> Tensor:
    """Score A-major pairs using retrieval, class-prompt, or label-transfer gates."""
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")
    output = torch.empty(len(a), len(b), dtype=torch.float32, device=a.global_feat.device)
    if not len(a) or not len(b):
        return output

    o = msf.o if task != "retrieval" and msf.pivot_route is not None else 0.0
    tau = msf.tau_A if task == "classification" else 0.0
    base_weight = (1.0 - tau) * (1.0 - o)
    base_terms = []
    if base_weight > 0:
        left = _base_features(a, "a", mcr, psrr, msf.covariance_maps, msf.omega, config)
        right = _base_features(b, "b", mcr, psrr, msf.covariance_maps, msf.omega, config)
        for k, (fa, fb) in enumerate(zip(left, right)):
            ma = mb = None
            if config.use_standardization:
                ma = _conditional(fa, msf.bank_b[k], True, chunk_size, config.scale_floor)
                mb = _conditional(fb, msf.bank_a[k], False, chunk_size, config.scale_floor)
            base_terms.append(_BaseTerm(fa, fb, ma, mb, base_weight * (msf.w if k else 1.0)))

    auxiliary = []
    for weight, route in (((1.0 - tau) * o, msf.pivot_route), (tau, msf.class_route)):
        if weight > 0 and route is not None:
            auxiliary.append((weight, route, route.a.log_posterior(a), route.b.log_posterior(b)))

    rows, cols = chunk_size or len(a), chunk_size or len(b)
    for start in range(0, len(a), rows):
        i = slice(start, min(start + rows, len(a)))
        for stop in range(0, len(b), cols):
            j = slice(stop, min(stop + cols, len(b)))
            block = a.global_feat.new_zeros(i.stop - i.start, j.stop - j.start)
            for term in base_terms:
                block.add_(term.score(i, j), alpha=term.weight)
            for weight, route, qa, qb in auxiliary:
                raw = _compatibility(qa[i], qb[j], route.log_prior)
                standardized = (raw - route.moments.mean) / route.moments.std
                block.add_(standardized, alpha=weight)
            output[i, j] = block
    return output
