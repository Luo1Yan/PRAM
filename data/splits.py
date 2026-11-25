"""Reproducible sampling on fitting edges and stratified wearable partitions."""

import numpy as np

from .schema import RelationPack


def split_disjoint_halves(
    pack: RelationPack, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    fixed = pack.ap_idx is not None and pack.pb_idx is not None
    if fixed and (seed == 0 or not len(pack.train_idx) or pack.separate_edges):
        return np.sort(pack.ap_idx), np.sort(pack.pb_idx)
    if pack.separate_edges:
        raise ValueError("Separate collections require explicit ap_idx and pb_idx")
    if len(pack.train_idx) < 2:
        raise ValueError("At least two training rows are required to split the edges")
    order = np.random.default_rng(seed).permutation(pack.train_idx)
    cut = len(order) // 2
    return np.sort(order[:cut]), np.sort(order[cut:])


def subsample_pair_budget(
    ap_idx: np.ndarray, pb_idx: np.ndarray, fraction: float, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    rng = np.random.default_rng(seed)

    def sample(idx):
        if len(idx) < 2:
            raise ValueError("Each budgeted edge requires at least two pairs")
        count = max(2, int(round(fraction * len(idx))))
        return np.sort(rng.choice(idx, count, replace=False))

    return sample(ap_idx), sample(pb_idx)


def split_wearable_c3t(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Allocate test, validation and HAR totals by stratified largest remainder.

    Totals are rounded once on the full corpus. Each allocation uses the remaining
    class populations. Equal remainders are broken randomly by the split RNG.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or not len(labels):
        raise ValueError("Wearable labels must be a nonempty integer vector")
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(labels == c)) for c in np.unique(labels)]
    n = len(labels)
    result = {}
    for key, size in (("test", round(0.1 * n)), ("val", round(0.1 * n)), ("har", round(0.4 * n))):
        counts = np.array([len(pool) for pool in pools], dtype=np.int64)
        numerators = counts * size
        allocation, remainder = np.divmod(numerators, counts.sum())
        order = np.lexsort((rng.random(len(counts)), -remainder))
        allocation[order[:size - allocation.sum()]] += 1
        result[key] = np.sort(np.concatenate([p[:k] for p, k in zip(pools, allocation)]))
        pools = [p[k:] for p, k in zip(pools, allocation)]
    result["align"] = np.sort(np.concatenate(pools))
    return result
