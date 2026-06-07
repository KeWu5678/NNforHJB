"""Tests for the Hydra config system: composition, model groups, config→PDAP."""

from __future__ import annotations

import numpy as np
import torch
from hydra import compose, initialize

import src.config.store  # noqa: F401  — registers `config_schema`
from src.config import get_activation
from src.PDAP import PDAP


def _data() -> dict:
    rng = np.random.default_rng(0)
    return {
        "x": rng.standard_normal((20, 2)),
        "v": rng.standard_normal((20, 1)),
        "dv": rng.standard_normal((20, 2)),
    }


def test_compose_defaults() -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
    assert cfg.model.kind == "signed"
    assert cfg.model.insertion == "profile"
    assert cfg.model.activation == "relu"
    assert cfg.model.power == 1.0
    assert cfg.model.alpha == 1e-5
    assert cfg.model.use_sphere is True
    assert cfg.training.num_iterations == 10
    assert cfg.training.max_ls_iter == 500
    assert cfg.training.ins_merge_tol == 1e-2
    assert cfg.data.path.endswith("VDP_beta_0.1_grid_30x30.npy")
    assert cfg.data.normalize is True
    assert cfg.env.seed == 42


def test_model_groups() -> None:
    with initialize(version_base=None, config_path="../conf"):
        sc = compose(config_name="config", overrides=["model=semiconcave"])
        fs = compose(config_name="config", overrides=["model=finite_step"])
    assert sc.model.kind == "semiconcave"
    assert sc.model.insertion == "profile"
    # finite_step config group = signed + finite_step
    assert fs.model.kind == "signed"
    assert fs.model.insertion == "finite_step"


def test_config_builds_pdap() -> None:
    """PDAP reads its identity + hyperparameters straight off the composed config."""
    data = _data()
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["model.gamma=0.5", "env.verbose=false"])

    pdap = PDAP(cfg, data)

    # objective hyperparameters live on the trainer, not the model
    assert pdap.objective.alpha == cfg.model.alpha == 1e-5
    assert pdap.objective.gamma == 0.5
    assert pdap.insertion_kind == "profile"
    assert type(pdap.model).__name__ == "SignedModel"
    assert pdap.model.power == cfg.model.power
    assert pdap.activation_fn is torch.relu
    # solver settings live on the trainer; insertion constants on the PDAP loop
    assert pdap.solver.max_ls_iter == 500
    assert pdap.fit_outer_iterations == 20
    assert pdap.ins_merge_tol == 1e-2


def test_from_config_loads_data() -> None:
    """from_config loads (and normalizes) the dataset named in cfg.data."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["env.verbose=false"])
    pdap = PDAP.from_config(cfg)
    assert pdap.input_dim == 2


def test_activation_resolver() -> None:
    assert get_activation("relu") is torch.relu
    assert callable(get_activation("matern52"))


def test_use_sphere_is_explicit() -> None:
    data = _data()
    with initialize(version_base=None, config_path="../conf"):
        default = compose(config_name="config", overrides=["env.verbose=false"])
        override = compose(config_name="config", overrides=["model.use_sphere=false", "env.verbose=false"])
    assert PDAP(default, data)._use_sphere is True
    assert PDAP(override, data)._use_sphere is False
