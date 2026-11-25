"""Tie-aware retrieval and classification metrics, reported as fractions."""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import torch
from torch import Tensor

from .config import sampled_indices


def _matrix(scores: Tensor) -> None:
    if scores.ndim != 2 or min(scores.shape) == 0:
        raise ValueError("Evaluation requires a nonempty query-by-candidate score matrix")
    if scores.is_complex() or torch.isnan(scores).any():
        raise ValueError("Scores must be ordered real values")


def _vector(values: Tensor, size: int, device: torch.device, name: str) -> Tensor:
    values = torch.as_tensor(values, device=device)
    if values.ndim != 1 or values.numel() != size:
        raise ValueError(f"{name} must have one entry per row")
    return values


def _category_hit(tied: int, positives: int, draws: int) -> float:
    if positives == 0 or draws == 0:
        return 0.0
    if draws > tied - positives:
        return 1.0
    total = comb(tied, draws)
    return (total - comb(tied - positives, draws)) / total


def expected_recall_at_k(
    scores: Tensor,
    k: int = 1,
    positive_indices: Tensor | None = None,
    *,
    query_labels: Tensor | None = None,
    candidate_labels: Tensor | None = None,
) -> float:
    """Use paired positives by default, or same-label positives with both label vectors."""
    _matrix(scores)
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    nq, nc = scores.shape
    k = min(k, nc)
    if query_labels is None and candidate_labels is None:
        if positive_indices is None:
            if nq > nc:
                raise ValueError("Supply the paired candidate index for each query")
            indices = torch.arange(nq, device=scores.device)
        else:
            indices = _vector(positive_indices, nq, scores.device, "positive_indices")
            if indices.is_floating_point() or indices.is_complex() or indices.dtype == torch.bool:
                raise ValueError("Positive indices must be integers")
            indices = indices.long()
            if (indices < 0).any() or (indices >= nc).any():
                raise ValueError("Positive index outside the candidate gallery")
        target = scores.gather(1, indices[:, None])
        above = (scores > target).sum(dim=1)
        tied = (scores == target).sum(dim=1)
        return ((k - above).double() / tied).clamp(0, 1).mean().item()

    if query_labels is None or candidate_labels is None or positive_indices is not None:
        raise ValueError("Category retrieval requires both label vectors and a single relevance rule")
    queries = _vector(query_labels, nq, scores.device, "query_labels")
    candidates = _vector(candidate_labels, nc, scores.device, "candidate_labels")
    positive = queries[:, None] == candidates
    if scores.is_floating_point():
        lowest = -torch.inf
    elif scores.dtype == torch.bool:
        lowest = False
    else:
        lowest = torch.iinfo(scores.dtype).min
    best = scores.masked_fill(~positive, lowest).amax(dim=1, keepdim=True)
    above = (scores > best).sum(dim=1)
    in_block = scores == best
    tied = in_block.sum(dim=1)
    positives = (in_block & positive).sum(dim=1)
    draws = (k - above).clamp_min(0).minimum(tied)
    counts = torch.stack((tied, positives, draws), dim=1).tolist()
    return sum(_category_hit(*row) for row in counts) / nq


def evaluate_bidirectional_retrieval(
    scores: Tensor,
    ks: Sequence[int] = (1, 5, 10),
    labels_a: Tensor | None = None,
    labels_b: Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Return directional recalls and their unweighted mean from A-major scores."""
    _matrix(scores)
    if labels_a is None and labels_b is None and scores.size(0) != scores.size(1):
        raise ValueError("Bidirectional instance retrieval requires aligned pairs")
    ab, ba, mean = {}, {}, {}
    for k in ks:
        key = f"R@{k}"
        ab[key] = expected_recall_at_k(scores, k, query_labels=labels_a, candidate_labels=labels_b)
        ba[key] = expected_recall_at_k(scores.T, k, query_labels=labels_b, candidate_labels=labels_a)
        mean[key] = 0.5 * (ab[key] + ba[key])
    return {"a_to_b": ab, "b_to_a": ba, "mean": mean}


def ordered_knn_votes(scores: Tensor, gallery_labels: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Count votes using the supplied gallery order to break neighbor ties."""
    if k < 1 or scores.ndim != 2 or scores.size(1) < 1:
        raise ValueError("Voting requires a positive k and a nonempty gallery")
    labels = _vector(gallery_labels, scores.size(1), scores.device, "gallery_labels")
    classes, codes = torch.unique(labels, sorted=True, return_inverse=True)
    order = scores.argsort(dim=1, descending=True, stable=True)[:, :min(k, len(labels))]
    neighbors = codes[order]
    votes = torch.zeros(len(scores), len(classes), dtype=torch.long, device=scores.device)
    votes.scatter_add_(1, neighbors, torch.ones_like(neighbors))
    return classes, votes


def evaluate_classification(
    scores: Tensor,
    targets: Tensor,
    candidate_labels: Tensor | None = None,
    *,
    protocol: str = "class_prompt",
    knn_k: int = 10,
    max_gallery_size: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate query rows against prompt columns or a pooled labeled training gallery."""
    _matrix(scores)
    targets = _vector(targets, len(scores), scores.device, "targets")
    if candidate_labels is None:
        if protocol == "knn":
            raise ValueError("Label transfer requires gallery labels")
        labels = torch.arange(scores.size(1), device=scores.device)
    else:
        labels = _vector(candidate_labels, scores.size(1), scores.device, "candidate_labels")
    if protocol == "knn":
        if max_gallery_size < 1:
            raise ValueError("max_gallery_size must be positive")
        index = sampled_indices(len(labels), max_gallery_size, scores.device, seed)
        labels, scores = ordered_knn_votes(scores[:, index], labels[index], knn_k)
    elif protocol != "class_prompt":
        raise ValueError("protocol must be 'class_prompt' or 'knn'")

    winners = scores == scores.amax(dim=1, keepdim=True)
    hits = (winners & (targets[:, None] == labels)).sum(dim=1)
    credit = hits.double() / winners.sum(dim=1)
    classes, inverse, counts = torch.unique(
        targets, sorted=True, return_inverse=True, return_counts=True
    )
    sums = credit.new_zeros(len(classes))
    sums.index_add_(0, inverse, credit)
    return {"accuracy": credit.mean().item(), "balanced_accuracy": (sums / counts).mean().item()}


def evaluate_bidirectional_knn(
    scores_a_to_b: Tensor,
    scores_b_to_a: Tensor,
    targets_a: Tensor,
    targets_b: Tensor,
    gallery_labels_b: Tensor,
    gallery_labels_a: Tensor,
    *,
    knn_k: int = 10,
    max_gallery_size: int = 10000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Average the two transfer directions using each modality's labeled gallery."""
    options = dict(protocol="knn", knn_k=knn_k, max_gallery_size=max_gallery_size, seed=seed)
    ab = evaluate_classification(scores_a_to_b, targets_a, gallery_labels_b, **options)
    ba = evaluate_classification(scores_b_to_a, targets_b, gallery_labels_a, **options)
    return {"a_to_b": ab, "b_to_a": ba, "mean": {k: 0.5 * (ab[k] + ba[k]) for k in ab}}
