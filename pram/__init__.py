"""Partwise Reasoning across Modalities."""

from .config import PRAMConfig
from .mcr import MCRStats
from .metrics import (
    evaluate_bidirectional_retrieval,
    evaluate_classification,
    expected_recall_at_k,
)
from .model import ModalityInput, PRAM
from .msf import MSFState
from .psrr import PSRRState
from .registration import align_spatial_grids, estimate_spatial_offset

__version__ = "1.0.0"

__all__ = [
    "PRAM",
    "PRAMConfig",
    "ModalityInput",
    "MCRStats",
    "PSRRState",
    "MSFState",
    "expected_recall_at_k",
    "evaluate_bidirectional_retrieval",
    "evaluate_classification",
    "estimate_spatial_offset",
    "align_spatial_grids",
]
