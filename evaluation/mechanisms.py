"""Paired budget comparisons, pivot correspondence and fitting-split sensitivity."""

from dataclasses import replace

from data import RelationPack
from pram import PRAMConfig
from .runner import evaluate_relation


def run_mechanism_suite(
    pack: RelationPack, base_config: PRAMConfig, *,
    pair_budgets=(0.02, 0.05, 0.20, 1.0), budget_seeds=(0, 1),
    test_shuffled_pivot=True, test_five_seeds=False, device="cpu", chunk_size=None,
) -> dict:
    cache = {}

    def measure(config, seed, fraction=1.0, shuffle=False):
        key = (config, seed, fraction, shuffle)
        if key not in cache:
            cache[key] = evaluate_relation(
                pack, config, seed=seed, device=device, chunk_size=chunk_size,
                budget_fraction=None if fraction == 1.0 else fraction,
                shuffle_pivot_parts=shuffle, include_frozen_cosine=False,
            )
        return cache[key]

    budgets = {}
    no_shrinkage = replace(base_config, use_shrinkage=False)
    for fraction in pair_budgets:
        budgets[str(float(fraction))] = {
            str(seed): {
                "pram_full": measure(base_config, seed, fraction),
                "wo_shrinkage": measure(no_shrinkage, seed, fraction),
            }
            for seed in budget_seeds
        }
    shuffled = None
    pivots = (pack.p_local, pack.pb_pivot[1])
    if test_shuffled_pivot and any(local is not None and local.shape[1] > 1 for local in pivots):
        shuffled = {
            "aligned": measure(base_config, 0),
            "shuffled": measure(base_config, 0, shuffle=True),
        }
    return {
        "name": pack.name, "dataset": pack.dataset, "backbone": pack.backbone,
        "pair_budgets": budgets, "shuffled_pivot": shuffled,
        "five_seeds": {str(s): measure(base_config, s) for s in range(5)} if test_five_seeds else None,
    }
