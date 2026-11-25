"""Fit on the two pivot edges and evaluate held-out endpoint relations."""

from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import normalize

from data import RelationPack, slice_modality, split_disjoint_halves, subsample_pair_budget
from pram import ModalityInput, PRAM, PRAMConfig, evaluate_bidirectional_retrieval, evaluate_classification
from pram.metrics import evaluate_bidirectional_knn


def percentages(metrics):
    return {
        key: percentages(value) if isinstance(value, dict) else 100.0 * value
        for key, value in metrics.items()
    }


def fit_edges(pack, config, ap, pb, device, *, seed=0, shuffle=False):
    a = slice_modality(pack.a_global, pack.a_local, ap)
    pa = slice_modality(pack.p_global, pack.p_local, ap)
    pp = slice_modality(*pack.pb_pivot, pb)
    b = slice_modality(pack.b_global, pack.b_local, pb)
    if shuffle:
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def permute(pivot):
            if not pivot.multiple_parts:
                return pivot
            local = pivot.local_feat
            order = torch.stack([
                torch.randperm(local.size(1), generator=generator) for _ in range(len(pivot))
            ])
            return ModalityInput(pivot.global_feat, local[torch.arange(len(pivot))[:, None], order])

        pa, pp = permute(pa), permute(pp)
    return PRAM(config, device=device).fit(a, pa, pp, b, is_class_prompt=pack.metric == "category")


def _measure(pack, config, retrieval_score, classification_score):
    test, train, labels = pack.test_idx, pack.train_idx, pack.labels
    if not len(test):
        raise ValueError("Relation evaluation requires held-out rows")
    category = pack.metric == "category"
    if category and labels is None:
        raise ValueError("Category retrieval requires labels")
    test_labels = torch.as_tensor(labels[test]) if category else None
    retrieval = evaluate_bidirectional_retrieval(
        retrieval_score(test, test), labels_a=test_labels, labels_b=test_labels,
    )
    classification = None
    if category:
        rows = np.union1d(train, test)
        classes, first = np.unique(labels[rows], return_index=True)
        scores = classification_score(rows[first], test)
        classification = evaluate_classification(
            scores.T, test_labels, torch.as_tensor(classes), protocol="class_prompt",
        )
    elif labels is not None:
        queries = test[labels[test] >= 0]
        gallery = train[labels[train] >= 0]
        if len(gallery) > config.max_gallery_size:
            gallery = np.sort(np.random.default_rng(0).choice(
                gallery, config.max_gallery_size, replace=False,
            ))
        if len(queries) and len(gallery):
            targets = torch.as_tensor(labels[queries])
            candidates = torch.as_tensor(labels[gallery])
            ab = classification_score(queries, gallery)
            ba = classification_score(gallery, queries).T
            classification = evaluate_bidirectional_knn(
                ab, ba, targets, targets, candidates, candidates,
                knn_k=config.knn_k, max_gallery_size=config.max_gallery_size, seed=0,
            )
    return {
        "retrieval": percentages(retrieval),
        "classification": percentages(classification) if classification is not None else None,
    }


def evaluate_relation(
    pack: RelationPack, config: PRAMConfig, *, seed: int = 0, device: str = "cpu",
    chunk_size: int | None = None, include_frozen_cosine: bool = True,
    shuffle_pivot_parts: bool = False, budget_fraction: float | None = None,
) -> dict:
    started = perf_counter()
    ap, pb = split_disjoint_halves(pack, seed)
    if budget_fraction is not None:
        ap, pb = subsample_pair_budget(ap, pb, budget_fraction, seed)
    fit_start = perf_counter()
    model = fit_edges(pack, config, ap, pb, device, seed=seed, shuffle=shuffle_pivot_parts)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    fit_seconds = perf_counter() - fit_start

    def scorer(method):
        def score(rows_a, rows_b):
            return method(
                slice_modality(pack.a_global, pack.a_local, rows_a),
                slice_modality(pack.b_global, pack.b_local, rows_b),
                chunk_size=chunk_size,
            )
        return score

    metrics = _measure(pack, config, scorer(model.score_retrieval), scorer(model.score_classification))
    frozen = None
    if include_frozen_cosine and pack.a_global.shape[1] == pack.b_global.shape[1]:
        def cosine(rows_a, rows_b):
            a = normalize(torch.as_tensor(pack.a_global[rows_a], device=device), dim=1)
            b = normalize(torch.as_tensor(pack.b_global[rows_b], device=device), dim=1)
            return a @ b.T

        frozen = _measure(pack, config, cosine, cosine)
    gates = {key: float(getattr(model.msf, key)) for key in ("omega", "w", "o", "tau_A")}
    gates["alpha"] = float(model.mcr.alpha)
    return {
        "name": pack.name, "backbone": pack.backbone, "dataset": pack.dataset,
        "metric": pack.metric, **metrics, "frozen_cosine": frozen, "gates": gates,
        "fit_seconds": fit_seconds, "total_seconds": perf_counter() - started,
    }
