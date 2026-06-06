"""Typed configuration schema for a PDAP run (Hydra structured configs).

Four sections compose into :class:`ExperimentConfig`:

  * ``model``    — a registered model: structure + insertion rule + hyperparameters.
  * ``training`` — how the model is fit: outer PDAP loop + SSN solver + insertion
    numeric constants.
  * ``data``     — the data source (a key-based ``.npy``/``.npz`` path with
    arrays ``x``, ``v``, and ``dv``).
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
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """A registered model = structure + insertion rule + hyperparameters.

    ``kind`` and ``insertion`` form the model identity (the ``conf/model/*.yaml``
    config group: signed/profile, semiconcave/profile, signed/finite_step).
    ``activation`` is a registry name resolved to a callable at build time;
    ``use_sphere`` is set by hand to match the activation's geometry.
    """

    # identity
    kind: str = "signed"          # "signed" | "semiconcave"
    insertion: str = "profile"    # "profile" | "finite_step"
    # structure
    activation: str = "relu"      # name resolved via src.config.activations
    power: float = 1.0
    # (w1, w2) = (value loss weight, gradient loss weight); l2 = (1, 0), h1 = (1, 1)
    loss_weights: Tuple[float, float] = (1.0, 1.0)
    # sample candidate directions on S^d — valid only for positively-homogeneous
    # activations (relu, abs, ...); set by hand to match the chosen activation.
    use_sphere: bool = True
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
    prune_amp_tol: float = 1e-8
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
    """The data source: a key-based ``.npy`` or ``.npz`` with ``x``, ``v``, ``dv``.

    ``path`` is a bare filename under ``DATA_DIR`` (see ``src.paths``); absolute
    paths are allowed. Resolution happens in ``src.data.load_value_samples``.
    The default points at the existing legacy VDP ``.npy``; new OpenLoop
    generators save ``.npz`` files with the same keys.
    ``train_fraction`` is the train/validation split applied in
    ``src.data.split_value_samples`` (first fraction trains, rest validates).
    ``normalize`` applies max-abs scaling (with chain-rule gradient transform)
    at load time; see ``PDAP.from_config``.
    """

    path: str = "VDP_beta_0.1_grid_30x30.npy"
    train_fraction: float = 0.9
    normalize: bool = True


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
