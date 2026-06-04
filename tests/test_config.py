"""Tests for the Hydra config system: composition, model registry, from_config."""

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
    assert cfg.training.num_iterations == 10
    assert cfg.training.max_ls_iter == 500
    assert cfg.training.ins_merge_tol == 1e-2
    assert cfg.data.path.endswith("VDP_beta_0.1_grid_30x30.npy")
    assert cfg.env.seed == 42


def test_model_registry() -> None:
    with initialize(version_base=None, config_path="../conf"):
        sc = compose(config_name="config", overrides=["model=semiconcave"])
        fs = compose(config_name="config", overrides=["model=finite_step"])
    assert sc.model.kind == "semiconcave"
    assert sc.model.insertion == "profile"
    # finite_step mirrors registry.py ALIASES["finite_step"]: signed + finite_step
    assert fs.model.kind == "signed"
    assert fs.model.insertion == "finite_step"


def test_from_config_equivalence() -> None:
    """PDAP.from_config matches an explicit constructor call with the same values."""
    data = _data()
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["model.gamma=0.5", "env.verbose=false"])

    from_cfg = PDAP.from_config(cfg, data)
    explicit = PDAP(
        data, alpha=1e-5, gamma=0.5, power=1.0, model="signed", insertion="profile",
        activation=torch.relu, loss_weights="h1", lr=1.0, th=0.5, use_sphere=True,
        c_init=1.0, verbose=False,
    )

    assert from_cfg.alpha == explicit.alpha
    assert from_cfg.insertion_kind == explicit.insertion_kind
    assert type(from_cfg.model).__name__ == type(explicit.model).__name__
    assert from_cfg.model.gamma == explicit.model.gamma == 0.5
    assert from_cfg.model.power == explicit.model.power
    assert from_cfg._use_sphere == explicit._use_sphere is True
    assert from_cfg.activation_fn is explicit.activation_fn is torch.relu
    # surfaced solver/insertion constants default to today's literals
    assert from_cfg.model.max_ls_iter == 500
    assert from_cfg.fit_outer_iterations == 20
    assert from_cfg.ins_merge_tol == 1e-2


def test_activation_resolver() -> None:
    assert get_activation("relu").use_sphere is True
    assert get_activation("matern52").use_sphere is False
    assert get_activation("relu").fn is torch.relu


def test_use_sphere_derived_unless_overridden() -> None:
    data = _data()
    with initialize(version_base=None, config_path="../conf"):
        default = compose(config_name="config", overrides=["env.verbose=false"])
        override = compose(config_name="config", overrides=["model.use_sphere=false", "env.verbose=false"])
    # relu derives use_sphere=True; an explicit override wins.
    assert PDAP.from_config(default, data)._use_sphere is True
    assert PDAP.from_config(override, data)._use_sphere is False
