from __future__ import annotations

import numpy as np

from src.data import ValueSampleNormalizer, load_value_samples
from src.metric import format_table


def test_load_value_samples_accepts_structured_npy(tmp_path):
    path = tmp_path / "samples.npy"
    dtype = [("x", "f8", (2,)), ("dv", "f8", (2,)), ("v", "f8")]
    data = np.zeros(3, dtype=dtype)
    data["x"] = [[1, 2], [3, 4], [5, 6]]
    data["v"] = [10, 20, 30]
    data["dv"] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    np.save(path, data)

    samples = load_value_samples(path)

    assert samples["x"].shape == (3, 2)
    assert samples["v"].shape == (3, 1)
    assert samples["dv"].shape == (3, 2)


def test_value_sample_normalizer_uses_chain_rule_for_gradient():
    samples = {
        "x": np.array([[2.0, -4.0]]),
        "v": np.array([[8.0]]),
        "dv": np.array([[3.0, 5.0]]),
    }
    normalizer = ValueSampleNormalizer.fit(samples)
    normalized = normalizer.normalize(samples)

    np.testing.assert_allclose(normalized["x"], [[1.0, -1.0]])
    np.testing.assert_allclose(normalized["v"], [[1.0]])
    np.testing.assert_allclose(normalized["dv"], [[0.75, 2.5]])


def test_format_table_is_markdown_without_pandas():
    text = format_table(
        [{"name": "relu", "score": 1.2345}],
        ["name", "score"],
        formats={"score": "{:.2f}"},
    )

    assert "| name | score |" in text
    assert "| relu | 1.23  |" in text
