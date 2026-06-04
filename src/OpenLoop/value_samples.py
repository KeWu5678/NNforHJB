"""PDAP-facing open-loop value-sample data object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ValueSamples:
    """Final retained `(x, V(x), dV(x))` samples for PDAP training."""

    x: np.ndarray
    v: np.ndarray
    dv: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=np.float64)
        v = np.asarray(self.v, dtype=np.float64).reshape(-1)
        dv = np.asarray(self.dv, dtype=np.float64)
        if x.size == 0:
            x = np.empty((0, 2), dtype=np.float64)
        if dv.size == 0:
            dv = np.empty((0, 2), dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x must have shape (n, 2)")
        if dv.shape != x.shape:
            raise ValueError("dv must have shape (n, 2)")
        if v.shape != (x.shape[0],):
            raise ValueError("v must have shape (n,)")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "dv", dv)

    @property
    def size(self) -> int:
        return int(self.x.shape[0])

    def to_pdap_dict(self) -> dict[str, np.ndarray]:
        return {
            "x": self.x.copy(),
            "v": self.v.copy(),
            "dv": self.dv.copy(),
        }

    def save_npz(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, x=self.x, v=self.v, dv=self.dv)
        return output_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "ValueSamples":
        with np.load(path) as data:
            return cls(x=data["x"], v=data["v"], dv=data["dv"])

    @classmethod
    def concatenate(cls, chunks: list["ValueSamples"]) -> "ValueSamples":
        if not chunks:
            return cls(
                x=np.empty((0, 2), dtype=np.float64),
                v=np.empty((0,), dtype=np.float64),
                dv=np.empty((0, 2), dtype=np.float64),
            )
        return cls(
            x=np.concatenate([chunk.x for chunk in chunks], axis=0),
            v=np.concatenate([chunk.v for chunk in chunks], axis=0),
            dv=np.concatenate([chunk.dv for chunk in chunks], axis=0),
        )


__all__ = ["ValueSamples"]
