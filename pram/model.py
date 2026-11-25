"""Public fitting, scoring, and prediction interfaces."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace

import torch
from torch import Tensor

from .config import PRAMConfig, sampled_indices
from .mcr import MCRStats, fit_mcr
from .metrics import ordered_knn_votes
from .msf import MSFState, fit_msf, pivot_partition, score_msf
from .psrr import PSRRState, fit_psrr


@dataclass(frozen=True)
class ModalityInput:
    """Official global features [N, d0] and aligned local features [N, G, d]."""

    global_feat: Tensor
    local_feat: Tensor | None = None

    def __post_init__(self) -> None:
        g, local = self.global_feat, self.local_feat
        if not isinstance(g, Tensor) or g.ndim != 2 or g.size(1) < 1 or g.is_complex():
            raise ValueError("global_feat must be a real matrix of positive width")
        if local is not None:
            if not isinstance(local, Tensor) or local.ndim != 3 or local.size(0) != len(g):
                raise ValueError("local_feat must have shape [N, G, d]")
            if min(local.shape[1:]) < 1 or local.is_complex() or local.device != g.device:
                raise ValueError("Local features require positive dimensions and the global device")

    def __len__(self) -> int:
        return self.global_feat.size(0)

    @property
    def multiple_parts(self) -> bool:
        return self.local_feat is not None and self.local_feat.size(1) > 1

    def vectors(self, parts: int) -> Tensor:
        """Return component-major features, broadcasting a single vector as needed."""
        if parts == 0:
            return self.global_feat.unsqueeze(0)
        if self.local_feat is None:
            return self.global_feat.unsqueeze(0).expand(parts, -1, -1)
        if self.local_feat.size(1) != parts:
            raise ValueError("Input local grid differs from the fitted grid")
        return self.local_feat.transpose(0, 1)

    def take(self, index: Tensor | slice) -> ModalityInput:
        local = None if self.local_feat is None else self.local_feat[index]
        return ModalityInput(self.global_feat[index], local)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> ModalityInput:
        def convert(x: Tensor) -> Tensor:
            return x.detach().to(device=device, dtype=dtype)

        local = None if self.local_feat is None else convert(self.local_feat)
        return ModalityInput(convert(self.global_feat), local)


class PRAM:
    """Fit separate A–P and P–B edges and return A-major pair-score matrices."""

    def __init__(
        self, config: PRAMConfig | None = None, device: torch.device | str | None = None
    ) -> None:
        self.config = config if config is not None else PRAMConfig()
        self.device = torch.device(device) if device is not None else None
        self.mcr: MCRStats | None = None
        self.psrr: PSRRState | None = None
        self.msf: MSFState | None = None

    @torch.inference_mode(False)
    @torch.no_grad()
    def fit(
        self,
        ap_a: ModalityInput,
        ap_p: ModalityInput,
        pb_p: ModalityInput,
        pb_b: ModalityInput,
        is_class_prompt: bool = False,
    ) -> PRAM:
        """Estimate all relations and reference distributions from the two fitting edges."""
        device = self.device if self.device is not None else ap_a.global_feat.device
        a, pa, pb, b = (x.to(device, torch.float64) for x in (ap_a, ap_p, pb_p, pb_b))
        mcr = fit_mcr(a, pa, pb, b, self.config)
        partition = pivot_partition(pa, pb, self.config)
        psrr = fit_psrr(mcr, len(partition[0]), self.config)
        msf = fit_msf(a, pa, pb, b, partition, mcr, psrr, self.config, is_class_prompt)
        self.mcr, self.psrr, self.msf = mcr, psrr, msf
        self.device = a.global_feat.device
        return self

    def to(self, device: torch.device | str) -> PRAM:
        """Move all fitted tensors, including voters and reference banks."""
        target = torch.device(device)
        if self.device == target:
            return self
        tensors = {}

        def move(value):
            if isinstance(value, Tensor):
                key = id(value)
                if key not in tensors:
                    tensors[key] = value.to(device=target)
                return tensors[key]
            if isinstance(value, list):
                return [move(item) for item in value]
            if is_dataclass(value):
                return replace(value, **{f.name: move(getattr(value, f.name)) for f in fields(value)})
            return value

        self.mcr, self.psrr, self.msf = move(self.mcr), move(self.psrr), move(self.msf)
        self.device = target
        return self

    @torch.no_grad()
    def _score(
        self, query_a: ModalityInput, cand_b: ModalityInput, task: str, chunk_size: int | None
    ) -> Tensor:
        if self.mcr is None or self.psrr is None or self.msf is None or self.device is None:
            raise RuntimeError("Fit PRAM before scoring")
        a = query_a.to(self.device, torch.float64)
        b = cand_b.to(self.device, torch.float64)
        return score_msf(a, b, self.mcr, self.psrr, self.msf, self.config, task, chunk_size)

    def score_retrieval(
        self, query_a: ModalityInput, cand_b: ModalityInput, chunk_size: int | None = None
    ) -> Tensor:
        """Return the float32 retrieval readout R."""
        return self._score(query_a, cand_b, "retrieval", chunk_size)

    def score_classification(
        self, query_a: ModalityInput, cand_b: ModalityInput, chunk_size: int | None = None
    ) -> Tensor:
        """Return the float32 classification readout S."""
        return self._score(query_a, cand_b, "classification", chunk_size)

    def predict_class_prompts(
        self,
        prompts_a: ModalityInput,
        samples_b: ModalityInput,
        prompt_labels: Tensor | None = None,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Return the winning A prompt label for each B sample, taking the first tied prompt."""
        if not len(prompts_a):
            raise ValueError("Class-prompt prediction requires at least one prompt")
        scores = self.score_classification(prompts_a, samples_b, chunk_size)
        winners = scores.argmax(dim=0)
        if prompt_labels is None:
            return winners
        labels = torch.as_tensor(prompt_labels, device=scores.device)
        if labels.ndim != 1 or len(labels) != len(prompts_a):
            raise ValueError("Supply one label per prompt")
        return labels[winners]

    def predict_knn(
        self,
        query_a: ModalityInput,
        cand_b: ModalityInput,
        gallery_labels: Tensor,
        chunk_size: int | None = None,
        *,
        gallery_side: str = "b",
    ) -> Tensor:
        """Transfer labels with S0. A tied majority returns the lowest winning label."""
        if gallery_side not in ("a", "b"):
            raise ValueError("gallery_side must be 'a' or 'b'")
        gallery = cand_b if gallery_side == "b" else query_a
        labels = torch.as_tensor(gallery_labels, device=gallery.global_feat.device)
        if labels.ndim != 1 or len(labels) != len(gallery) or not len(labels):
            raise ValueError("Supply one label per row of a nonempty training gallery")
        order = sampled_indices(len(gallery), self.config.max_gallery_size, labels.device)
        selected = gallery.take(order)
        if gallery_side == "b":
            scores = self._score(query_a, selected, "transfer", chunk_size)
        else:
            scores = self._score(selected, cand_b, "transfer", chunk_size).T
        classes, votes = ordered_knn_votes(scores, labels[order], self.config.knn_k)
        return classes[votes.argmax(dim=1)]
