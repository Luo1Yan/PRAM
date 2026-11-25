"""Compare frozen cosine and PRAM on independently projected Gaussian parts."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pram import (
    ModalityInput,
    PRAM,
    PRAMConfig,
    evaluate_bidirectional_retrieval,
    evaluate_classification,
)


@dataclass(frozen=True)
class Benchmark:
    ap_a: ModalityInput
    ap_p: ModalityInput
    pb_p: ModalityInput
    pb_b: ModalityInput
    test_a: ModalityInput
    test_b: ModalityInput
    prompts_a: ModalityInput
    samples_b: ModalityInput
    labels_b: Tensor


def make_benchmark(seed: int = 0, parts: int = 6) -> Benchmark:
    """Draw disjoint fitting collections and independent retrieval and prompt tasks."""
    if parts < 1:
        raise ValueError("parts must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = 4

    def noise(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, device="cpu", dtype=torch.float64)

    def frame(*shape: int) -> Tensor:
        q, r = torch.linalg.qr(noise(*shape), mode="reduced")
        signs = r.diagonal(dim1=-2, dim2=-1).sign().unsqueeze(-2)
        return q * signs

    global_frames = [frame(parts * width, width) for _ in range(3)]
    local_frames = [frame(parts, width, width) for _ in range(3)]

    def latent(n: int) -> Tensor:
        return noise(n, parts, width)

    def observe(z: Tensor, modality: int) -> ModalityInput:
        global_feat = z.flatten(1) @ global_frames[modality]
        local_feat = torch.einsum("ngr,grd->ngd", z, local_frames[modality])
        return ModalityInput(
            (global_feat + 0.03 * noise(*global_feat.shape)).float(),
            (local_feat + 0.03 * noise(*local_feat.shape)).float(),
        )

    za, zb, test = latent(160), latent(184), latent(72)
    prototypes = latent(16)
    labels = torch.arange(len(prototypes)).repeat_interleave(4)
    samples = (prototypes[labels] + 0.20 * latent(len(labels))) / math.sqrt(1.04)
    return Benchmark(
        observe(za, 0), observe(za, 1), observe(zb, 1), observe(zb, 2),
        observe(test, 0), observe(test, 2), observe(prototypes, 0), observe(samples, 2), labels
    )


def run_benchmark(data: Benchmark, device: torch.device | str) -> dict[str, dict[str, float]]:
    def cosine(a: ModalityInput, b: ModalityInput) -> Tensor:
        x = F.normalize(a.global_feat.to(device), dim=1)
        y = F.normalize(b.global_feat.to(device), dim=1)
        return x @ y.T

    def metrics(retrieval: Tensor, classification: Tensor) -> dict[str, float]:
        recall = evaluate_bidirectional_retrieval(retrieval, ks=(1,))
        accuracy = evaluate_classification(classification.T, data.labels_b)
        return {
            "a_to_b_r1": recall["a_to_b"]["R@1"],
            "b_to_a_r1": recall["b_to_a"]["R@1"],
            "mean_r1": recall["mean"]["R@1"],
            **accuracy,
        }

    results = {"Frozen Cosine": metrics(
        cosine(data.test_a, data.test_b), cosine(data.prompts_a, data.samples_b)
    )}
    for name, use_local in (("Global-only PRAM", False), ("Full PRAM", True)):
        model = PRAM(PRAMConfig(use_local=use_local), device=device).fit(
            data.ap_a, data.ap_p, data.pb_p, data.pb_b, is_class_prompt=True
        )
        results[name] = metrics(
            model.score_retrieval(data.test_a, data.test_b),
            model.score_classification(data.prompts_a, data.samples_b),
        )
    return results


def main(argv=None) -> dict[str, dict[str, float]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parts", type=int, default=6)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    threads = torch.get_num_threads()
    limit_threads = device.type == "cpu" and threads != 1
    if limit_threads:
        torch.set_num_threads(1)
    try:
        results = run_benchmark(make_benchmark(args.seed, args.parts), device)
    finally:
        if limit_threads:
            torch.set_num_threads(threads)
    print(f"{'Method':<18} {'A->B R@1':>10} {'B->A R@1':>10} {'Mean R@1':>10} "
          f"{'Prompt acc.':>12} {'Balanced acc.':>14}")
    for name, row in results.items():
        print(f"{name:<18} {row['a_to_b_r1']:>10.1%} {row['b_to_a_r1']:>10.1%} "
              f"{row['mean_r1']:>10.1%} {row['accuracy']:>12.1%} "
              f"{row['balanced_accuracy']:>14.1%}")
    return results


if __name__ == "__main__":
    main()
