"""Centralized PDAP configuration (Hydra structured configs + activation registry)."""

from __future__ import annotations

from .schema import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    EnvConfig,
    ExperimentConfig,
)
from .activations import ACTIVATIONS, get_activation
from .store import register_configs

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "EnvConfig",
    "ExperimentConfig",
    "ACTIVATIONS",
    "get_activation",
    "register_configs",
]
