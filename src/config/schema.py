"""Typed configuration schema for a PDAP run (Hydra structured configs).

Four sections compose into :class:`ExperimentConfig`:

  * ``model``    — a registered model: structure + insertion rule + hyperparameters.
  * ``training`` — how the model is fit: outer PDAP loop + SSN solver + insertion
    numeric constants.
  * ``data``     — the data source (a ``.npy`` path with keys ``x, v, dv``).
  * ``env``      — runtime: seed + logging.

Every default equals the value currently in force for the VDP signed-profile
baseline (``scripts/run_activation_experiment.py``) and the hardcoded library
literals, so the default ``ExperimentConfig`` reproduces today's behavior.

The config is **domain-agnostic** — it describes the PDAP model and how it is
trained, not any specific control problem. The only problem-specific input is
``data.path``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """A registered model = structure + insertion rule + hyperparameters.

    ``kind`` and ``insertion`` form the model identity (mirroring the
    ``registry.py`` ``ALIASES``: signed/profile, semiconcave/profile,
    signed/finite_step). ``activation`` is a registry name (resolved to a
    callable + its ``use_sphere`` geometry at build time); ``use_sphere=None``
    derives the geometry from the activation, an explicit bool overrides it.
    """

    # identity
    kind: str = "signed"          # "signed" | "semiconcave"
    insertion: str = "profile"    # "profile" | "finite_step"
    # structure
    activation: str = "relu"      # name resolved via src.config.activations
    power: float = 1.0
    loss_weights: str = "h1"      # "l2" | "h1" | (resolved to a (w1, w2) tuple)
    use_sphere: Optional[bool] = None
    c_init: float = 1.0           # semiconcave only
    # regularization hyperparameters
    alpha: float = 1e-5
    gamma: float = 0.0
    th: float = 0.5


@dataclass
class TrainingConfig:
    """How the model is fit: outer PDAP loop + SSN solver + insertion constants."""

    # outer PDAP loop
    num_iterations: int = 10
    num_insertion: int = 50
    max_insert: int = 15
    prune_merge_tol: float = 1e-3
    threshold: float = 1e-5       # legacy: recorded by PDAP.fit, not used in the loop
    decorrelation: bool = False
    # SSN solver (src/SSN/optimizer.py defaults + the hardcoded iterations=20)
    lr: float = 1.0
    method: str = "levenberg_marquardt"   # "levenberg_marquardt" | "steihaug_cg"
    max_ls_iter: int = 500
    tolerance_ls: float = 1.0 + 1e-8
    tolerance_grad: float = 0.0
    sigmamax: float = 10.0
    fit_outer_iterations: int = 20
    display_every: int = 2
    # insertion numeric constants (src/PDAP/insertion.py)
    ins_merge_tol: float = 1e-2
    lbfgs_lr: float = 1e-2
    lbfgs_steps: int = 200
    newton_tol: float = 1e-12
    newton_max_iter: int = 50


@dataclass
class DataConfig:
    """The data source: a ``.npy`` with keys ``x, v, dv``.

    ``path`` is resolved relative to ``DATA_DIR`` (absolute paths allowed).
    PDAP splits train/validation internally, so no split knob is needed here.
    """

    path: str = "VDP_beta_0.1_grid_30x30.npy"


@dataclass
class EnvConfig:
    """Runtime: random seed + logging.

    Fixed (not configured): device is CPU-only (no GPU path exists) and dtype is
    float64 (hardcoded across PDAP/models/SSN). Surfacing those is future work.
    """

    seed: int = 42
    verbose: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    name: str = "run"


__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "EnvConfig",
    "ExperimentConfig",
]
