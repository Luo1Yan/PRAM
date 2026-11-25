"""Component ablations on identical training, test and gallery rows."""

from dataclasses import replace

from .runner import evaluate_relation


ABLATION_VARIANTS = {
    "pram_full": {},
    "global_only": {"use_local": False},
    "wo_global": {"use_global": False},
    "wo_shrinkage": {"use_shrinkage": False},
    "wo_quadratic": {"use_quadratic": False},
    "wo_covariance_score": {"use_covariance_score": False},
    "wo_standardization": {"use_standardization": False},
    "wo_pivot_associations": {"use_pivot_associations": False},
    "wo_class_semantics": {"use_class_semantics": False},
}


def run_ablation_suite(
    pack, base_config, variants=None, *, seed=0, device="cpu", chunk_size=None,
) -> dict:
    names = list(ABLATION_VARIANTS if variants is None else variants)
    if not names or len(set(names)) != len(names) or set(names) - ABLATION_VARIANTS.keys():
        raise ValueError("Supply a nonempty selection of distinct ablation variants")
    results = {}
    frozen = None
    for i, name in enumerate(names):
        config = replace(base_config, **ABLATION_VARIANTS[name])
        result = evaluate_relation(
            pack, config, seed=seed, device=device, chunk_size=chunk_size,
            include_frozen_cosine=i == 0,
        )
        baseline = result.pop("frozen_cosine")
        if i == 0:
            frozen = baseline
        results[name] = result
    return {
        "name": pack.name, "dataset": pack.dataset, "backbone": pack.backbone,
        "variants": results, "frozen_cosine": frozen,
    }
