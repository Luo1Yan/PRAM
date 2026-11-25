"""Run a frozen-feature benchmark from a JSON experiment configuration."""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean

import torch

from data import load_relation_pack, resolve_pack_path
from evaluation import (
    evaluate_relation, evaluate_robot_localization, evaluate_vehicle_localization,
    evaluate_wearable_imu, run_ablation_suite, run_mechanism_suite,
)
from pram import PRAMConfig


_MODES = {"main", "ablation", "sensing_robot", "sensing_vehicle", "sensing_wearable", "mechanism"}


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _relations(config, requested):
    entries = {}
    for item in config["relations"]:
        item = {"pack": item} if isinstance(item, str) else dict(item)
        path = item.get("pack") or item.get("pack_file") or item.get("file")
        if not isinstance(path, str) or not path:
            raise ValueError("Each relation requires a pack filename")
        name = item.get("name", Path(path).stem)
        if name in entries:
            raise ValueError(f"Duplicate relation {name}")
        entries[name] = {**item, "name": name, "pack": path}
    selected = set(requested or entries)
    if selected - entries.keys():
        raise ValueError(f"Unknown relations: {sorted(selected - entries.keys())}")
    if not selected:
        raise ValueError("The experiment requires at least one relation")
    return [entry for name, entry in entries.items() if name in selected]


def _benchmark_means(records, relations, mode):
    totals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    masks = {r["name"]: r.get("report", ["retrieval", "classification"]) for r in relations}
    for row in records:
        methods = dict(row["variants"]) if mode == "ablation" else {"pram_full": row}
        if row["frozen_cosine"] is not None:
            methods["frozen_cosine"] = row["frozen_cosine"]
        for method, values in methods.items():
            bucket = totals[row["backbone"]][method]
            if "retrieval" in masks[row["relation"]]:
                bucket["R@10"].append(values["retrieval"]["mean"]["R@10"])
            cls = values["classification"]
            if "classification" in masks[row["relation"]] and cls is not None:
                bucket["Top-1"].append(cls.get("mean", cls)["accuracy"])
    return {
        backbone: {
            method: {metric: fmean(scores) for metric, scores in metrics.items()}
            for method, metrics in methods.items()
        }
        for backbone, methods in totals.items()
    }


def _summary(values, path=()):
    for key, value in values.items():
        if isinstance(value, dict) and key not in ("a_to_b", "b_to_a", "gates"):
            yield from _summary(value, (*path, key))
        elif isinstance(value, (int, float)) and (
            key in ("R@1", "R@10", "accuracy", "Top-1", "Top-3") or key.endswith("m")
        ):
            yield "/".join((*path, key)), value


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pram_config", type=Path)
    parser.add_argument("--data_dir", type=Path, default=Path("./data_cache"))
    parser.add_argument("--hf_repo")
    parser.add_argument("--relations", nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    experiment = _read(args.config)
    mode = experiment["mode"]
    if mode not in _MODES:
        raise ValueError(f"Unknown experiment mode {mode}")
    options = experiment.get("pram", {})
    if isinstance(options, str):
        options = _read(args.config.parent / options)
    if args.pram_config is not None:
        options = {**options, **_read(args.pram_config)}
    config = PRAMConfig(**options)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select --device cpu")
    chunk = args.chunk_size if args.chunk_size is not None else experiment.get("chunk_size")
    if chunk is not None and (type(chunk) is not int or chunk < 1):
        raise ValueError("chunk_size must be a positive integer")
    common = {"device": device, "chunk_size": chunk}
    defaults = [1, 2, 3] if mode == "sensing_wearable" else [0]
    if mode == "mechanism":
        defaults = experiment.get("budget_seeds", [0, 1])
    seeds = args.seeds or experiment.get("seeds", defaults)
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Supply distinct nonnegative integer seeds")
    repo = args.hf_repo or experiment.get("hf_repo")
    revision = experiment.get("revision", "main")
    relations = _relations(experiment, args.relations)
    records = []
    for spec in relations:
        pack = load_relation_pack(resolve_pack_path(spec["pack"], args.data_dir, repo, revision))
        identity = {"relation": spec["name"], "pack": spec["pack"]}
        thresholds = spec.get("thresholds", experiment.get("thresholds", [5, 15, 25]))
        if mode == "sensing_vehicle":
            result = evaluate_vehicle_localization(pack, config, thresholds, **common)
        elif mode == "sensing_wearable":
            result = evaluate_wearable_imu(pack, config, seeds, **common)
        elif mode == "mechanism":
            result = run_mechanism_suite(
                pack, config, pair_budgets=experiment.get("pair_budgets", [0.02, 0.05, 0.2, 1.0]),
                budget_seeds=seeds, test_shuffled_pivot=experiment.get("test_shuffled_pivot", True),
                test_five_seeds=experiment.get("test_five_seeds", False), **common,
            )
            if experiment.get("sensitivity"):
                result["sensitivity"] = {
                    name: evaluate_relation(
                        pack, replace(config, **changes), seed=seeds[0],
                        include_frozen_cosine=False, **common,
                    )
                    for name, changes in experiment["sensitivity"].items()
                }
        else:
            for seed in seeds:
                if mode == "main":
                    result = evaluate_relation(pack, config, seed=seed, **common)
                elif mode == "ablation":
                    result = run_ablation_suite(pack, config, experiment.get("variants"), seed=seed, **common)
                else:
                    result = evaluate_robot_localization(pack, config, thresholds, seed=seed, **common)
                records.append({**identity, "seed": seed, **result})
            continue
        records.append({**identity, **result})

    report = {
        "mode": mode, "device": device, "chunk_size": chunk, "pram": asdict(config),
        "seeds": [] if mode == "sensing_vehicle" else list(seeds), "hf_repo": repo,
        "revision": revision, "experiment": experiment, "results": records,
    }
    if mode in ("main", "ablation"):
        report["means"] = _benchmark_means(records, relations, mode)
    if args.output is not None:
        contents = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents + "\n", encoding="utf-8")
    print("Relation\tSeed\tMetric\tPercent")
    for record in records:
        for metric, value in _summary(record):
            print(f"{record['relation']}\t{record.get('seed', '-')}\t{metric}\t{value:.3f}")
    return report


if __name__ == "__main__":
    main()
