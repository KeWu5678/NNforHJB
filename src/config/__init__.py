"""Centralized PDAP configuration (Hydra structured configs + activation registry)."""

from __future__ import annotations

from .schema import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    EnvConfig,
    ExperimentConfig,
)
from .activations import ACTIVATIONS, ActivationSpec, get_activation, matern52
from .store import register_configs

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "EnvConfig",
    "ExperimentConfig",
    "ACTIVATIONS",
    "ActivationSpec",
    "get_activation",
    "matern52",
    "register_configs",
]
