"""Register the structured-config schema with Hydra's ConfigStore.

Importing this module registers ``ExperimentConfig`` under the name
``config_schema``; ``conf/config.yaml`` pulls it in via its defaults list so the
composed YAML is validated against the typed schema.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from .schema import ExperimentConfig


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=ExperimentConfig)


# Register on import (idempotent — ConfigStore.store overwrites).
register_configs()
