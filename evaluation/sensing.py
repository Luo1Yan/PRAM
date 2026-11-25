"""Paired-frame localization, cross-device localization and wearable recognition."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from data import RelationPack, slice_modality, split_disjoint_halves, split_wearable_c3t
from pram import ModalityInput, PRAM, PRAMConfig, evaluate_bidirectional_retrieval
from .runner import fit_edges, percentages


def localization_recall(
    scores: Tensor, query_xy: np.ndarray, db_xy: np.ndarray, thresholds: Sequence[float],
) -> dict[str, float]:
    """Report top-one success percentages with inclusive Euclidean distance bounds."""
    query_xy, db_xy = np.asarray(query_xy, dtype=np.float64), np.asarray(db_xy, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) < 1:
        raise ValueError("Localization requires a nonempty query-by-database matrix")
    if query_xy.shape != (scores.size(0), 2) or db_xy.shape != (scores.size(1), 2):
        raise ValueError("Coordinate rows must follow score matrix order")
    if torch.isnan(scores).any() or not np.isfinite(query_xy).all() or not np.isfinite(db_xy).all():
        raise ValueError("Invalid localization scores or coordinates")
    best = scores.argmax(1).cpu().numpy()
    distances = np.linalg.norm(query_xy - db_xy[best], axis=1)
    recalls = {}
    for threshold in thresholds:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Distance thresholds must be finite and nonnegative")
        recalls[f"{threshold:g}m"] = float(100 * np.mean(distances <= threshold))
    return recalls


def evaluate_robot_localization(
    pack: RelationPack, config: PRAMConfig, thresholds: Sequence[float] = (5.0, 15.0, 25.0),
    *, seed: int = 0, device: str = "cpu", chunk_size: int | None = None,
) -> dict:
    if pack.coords is None:
        raise ValueError("Robot localization requires aligned frame coordinates")
    endpoints = (pack.a_name.lower(), pack.b_name.lower())
    if set(endpoints) != {"thermal", "depth"}:
        raise ValueError("Robot localization requires thermal and depth endpoints")
    ap, pb = split_disjoint_halves(pack, seed)
    model = fit_edges(pack, config, ap, pb, device)
    rows = pack.test_idx
    scores = model.score_retrieval(
        slice_modality(pack.a_global, pack.a_local, rows),
        slice_modality(pack.b_global, pack.b_local, rows),
        chunk_size=chunk_size,
    )
    thermal = scores if endpoints[0] == "thermal" else scores.T
    xy = pack.coords[rows]
    return {
        "name": pack.name, "dataset": pack.dataset, "backbone": pack.backbone,
        "thermal_to_depth": localization_recall(thermal, xy, xy, thresholds),
        "exact_retrieval": percentages(evaluate_bidirectional_retrieval(scores, ks=(1, 10))),
    }


def _selected_xy(dedicated, shared, rows):
    if dedicated is not None:
        return dedicated if len(dedicated) == len(rows) else dedicated[rows]
    if shared is None:
        raise ValueError("Supply coordinates for each database and query set")
    return shared[rows]


def evaluate_vehicle_localization(
    pack: RelationPack, config: PRAMConfig, thresholds: Sequence[float] = (5.0, 15.0, 25.0),
    *, device: str = "cpu", chunk_size: int | None = None,
) -> dict:
    if pack.ap_idx is None or pack.pb_idx is None or pack.db_idx is None or not pack.query_sets:
        raise ValueError("Vehicle localization requires both fitting edges, db_idx and query_sets")
    model = fit_edges(pack, config, pack.ap_idx, pack.pb_idx, device)
    database = slice_modality(pack.a_global, pack.a_local, pack.db_idx)
    db_xy = _selected_xy(pack.db_coords, pack.coords, pack.db_idx)
    results = {}
    for name, rows in pack.query_sets.items():
        queries = slice_modality(pack.b_global, pack.b_local, rows)
        scores = model.score_retrieval(database, queries, chunk_size=chunk_size).T
        query_xy = _selected_xy((pack.query_coords or {}).get(name), pack.coords, rows)
        results[name] = localization_recall(scores, query_xy, db_xy, thresholds)
    return {
        "name": pack.name, "dataset": pack.dataset, "backbone": pack.backbone,
        "query_sets": results,
    }


def evaluate_wearable_imu(
    pack: RelationPack, config: PRAMConfig, seeds: Sequence[int] = (1, 2, 3),
    *, device: str = "cpu", chunk_size: int | None = None,
) -> dict:
    labels = pack.labels
    if labels is None or not len(labels):
        raise ValueError("Wearable recognition requires action labels")
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Wearable action labels must be contiguous and start at zero")
    if not len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Supply distinct wearable split seeds")
    identity = torch.eye(len(classes), dtype=torch.float32)
    prompts = ModalityInput(identity)
    per_seed = {}
    for seed in seeds:
        split = split_wearable_c3t(labels, seed)
        har, align, test = (split[key] for key in ("har", "align", "test"))
        if not min(len(har), len(align), len(test)):
            raise ValueError("HAR, alignment and test splits must all contain rows")
        model = PRAM(config, device=device).fit(
            ModalityInput(identity[torch.from_numpy(labels[har])]),
            slice_modality(pack.p_global, pack.p_local, har),
            slice_modality(pack.p_global, pack.p_local, align),
            slice_modality(pack.b_global, pack.b_local, align),
            is_class_prompt=False,
        )
        scores = model.score_classification(
            prompts, slice_modality(pack.b_global, pack.b_local, test), chunk_size=chunk_size,
        ).T
        ranking = scores.argsort(dim=1, descending=True, stable=True)
        targets = torch.as_tensor(labels[test], device=scores.device)[:, None]
        per_seed[str(seed)] = {
            f"Top-{k}": 100 * (ranking[:, :k] == targets).any(1).double().mean().item()
            for k in (1, 3)
        }
    return {
        "name": pack.name, "dataset": pack.dataset, "backbone": pack.backbone,
        "per_seed": per_seed,
        "mean": {key: float(np.mean([r[key] for r in per_seed.values()])) for key in ("Top-1", "Top-3")},
    }
