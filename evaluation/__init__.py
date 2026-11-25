"""Evaluation protocols for frozen feature packs."""

from .ablations import ABLATION_VARIANTS, run_ablation_suite
from .mechanisms import run_mechanism_suite
from .runner import evaluate_relation
from .sensing import (
    evaluate_robot_localization,
    evaluate_vehicle_localization,
    evaluate_wearable_imu,
    localization_recall,
)

__all__ = [
    "ABLATION_VARIANTS",
    "evaluate_relation",
    "run_ablation_suite",
    "run_mechanism_suite",
    "localization_recall",
    "evaluate_robot_localization",
    "evaluate_vehicle_localization",
    "evaluate_wearable_imu",
]
