"""Estimator controls and reproducible row sampling."""

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True)
class PRAMConfig:
    """Numerical conventions and component switches for PRAM."""

    relative_ridge: float = 0.5
    max_directions: int = 256
    max_bank_size: int = 2000
    max_gallery_size: int = 10000
    knn_k: int = 10
    state_decimals: int = 4
    lbfgs_max_iter: int = 200
    scale_floor: float = 1e-12
    corr_cap: float = 1.0 - 1e-6
    lda_diag_eps: float = 1e-6
    class_prompt_cv_seed: int = 7
    use_global: bool = True
    use_local: bool = True
    use_shrinkage: bool = True
    use_quadratic: bool = True
    use_covariance_score: bool = True
    use_standardization: bool = True
    use_pivot_associations: bool = True
    use_class_semantics: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_directions", "max_bank_size", "max_gallery_size",
            "knn_k", "lbfgs_max_iter",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.state_decimals) is not int or self.state_decimals < 0:
            raise ValueError("state_decimals must be a nonnegative integer")
        if not isfinite(self.relative_ridge) or self.relative_ridge < 0:
            raise ValueError("relative_ridge must be finite and nonnegative")
        for name in ("scale_floor", "lda_diag_eps"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.corr_cap < 1:
            raise ValueError("corr_cap must lie in [0, 1)")
        if not self.use_global and not self.use_local:
            raise ValueError("At least one of use_global or use_local must be True")


def sampled_indices(
    n: int, limit: int, device: torch.device | str, seed: int = 0
) -> Tensor:
    """Draw an ordered sample without replacement using a CPU generator."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(n, generator=generator, device="cpu")
    return indices[:limit].to(device=device)
