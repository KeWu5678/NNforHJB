#!/usr/bin/env python3
"""Check whether large gamma effects align with sparsity or activation class."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def parse_gamma_values(text: str) -> list[float]:
    values = []
    for item in text.split(";"):
        if item.strip():
            _, value = item.strip().split(":")
            values.append(float(value))
    return values


def load_effects(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    rows: dict[str, dict[str, dict[str, str]]] = {}
    with path.open() as file:
        for row in csv.DictReader(file):
            rows.setdefault(row["activation"], {})[row["metric"]] = row
    return rows


def rankdata(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array)
    ranks = np.empty(len(values_array), dtype=float)
    i = 0
    while i < len(values_array):
        j = i
        while j + 1 < len(values_array) and values_array[order[j + 1]] == values_array[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def corr(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if len(left_array) < 2 or np.std(left_array) == 0.0 or np.std(right_array) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def spearman(left: list[float], right: list[float]) -> float:
    return corr(rankdata(left), rankdata(right))


def class_tags(activation: str) -> set[str]:
    tags = set()
    if any(token in activation for token in ["relu", "smoothy", "abs", "x_absx", "sqrt_signed"]):
        tags.add("sharp_or_inactive")
    if any(token in activation for token in ["tanh", "atan", "asinh", "softsign", "sigmoid", "rcip"]):
        tags.add("saturating")
    if any(token in activation for token in ["gelu", "swish", "silu", "mish", "hardswish"]):
        tags.add("smooth_gated")
    if any(token in activation for token in ["cubic", "quartic", "quad", "sp2", "elu2", "squared"]):
        tags.add("polynomial_or_squared")
    return tags


def experiment_rows(path: Path, accuracy_metric: str, score_metric: str) -> list[dict[str, object]]:
    effects = load_effects(path)
    rows = []
    for activation, metrics in effects.items():
        if accuracy_metric not in metrics or "neurons" not in metrics or score_metric not in metrics:
            continue
        rows.append(
            {
                "activation": activation,
                "mean_neurons": float(np.mean(parse_gamma_values(metrics["neurons"]["gamma_values"]))),
                "accuracy_range": float(metrics[accuracy_metric]["abs_range"]),
                "neuron_range": float(metrics["neurons"]["abs_range"]),
                "score_range": float(metrics[score_metric]["abs_range"]),
                "tags": class_tags(activation),
            }
        )
    return rows


def format_row(row: dict[str, object]) -> str:
    tags = ", ".join(sorted(row["tags"])) or "other"
    return (
        f"| `{row['activation']}` | {row['score_range']:.3g} | "
        f"{row['accuracy_range']:.3g} | {row['neuron_range']:.3g} | "
        f"{row['mean_neurons']:.1f} | {tags} |"
    )


def write_section(
    lines: list[str],
    title: str,
    rows: list[dict[str, object]],
) -> None:
    mean_neurons = [float(row["mean_neurons"]) for row in rows]
    lines.extend(
        [
            f"## {title}",
            "",
            "| effect metric | Pearson vs mean neurons | Spearman vs mean neurons |",
            "|:--------------|------------------------:|-------------------------:|",
        ]
    )
    for metric in ["accuracy_range", "neuron_range", "score_range"]:
        values = [float(row[metric]) for row in rows]
        lines.append(f"| `{metric}` | {corr(mean_neurons, values):.3f} | {spearman(mean_neurons, values):.3f} |")

    median_neurons = float(np.median(mean_neurons))
    score_ranges = [float(row["score_range"]) for row in rows]
    score_q75 = float(np.quantile(score_ranges, 0.75))
    score_median = float(np.median(score_ranges))

    lines.extend(
        [
            "",
            f"Median mean neurons: `{median_neurons:.1f}`. 75th percentile score range: `{score_q75:.3g}`.",
            "",
            "### Largest score-range effects",
            "",
            "| activation | score range | accuracy range | neuron range | mean neurons | class |",
            "|:-----------|------------:|---------------:|-------------:|-------------:|:------|",
        ]
    )
    for row in sorted(rows, key=lambda item: float(item["score_range"]), reverse=True)[:12]:
        lines.append(format_row(row))

    counterexamples = [
        row
        for row in rows
        if float(row["score_range"]) >= score_q75
        and float(row["mean_neurons"]) < median_neurons
        and "sharp_or_inactive" not in row["tags"]
    ]
    lines.extend(
        [
            "",
            "### Counterexamples: high gamma effect without high neurons or sharp/inactive class",
            "",
            "| activation | score range | accuracy range | neuron range | mean neurons | class |",
            "|:-----------|------------:|---------------:|-------------:|-------------:|:------|",
        ]
    )
    for row in sorted(counterexamples, key=lambda item: float(item["score_range"]), reverse=True)[:12]:
        lines.append(format_row(row))
    if not counterexamples:
        lines.append("| none in this rerun set | | | | | |")

    quiet_high_neuron = [
        row
        for row in rows
        if (float(row["mean_neurons"]) >= median_neurons or "sharp_or_inactive" in row["tags"])
        and float(row["score_range"]) < score_median
    ]
    lines.extend(
        [
            "",
            "### Counterexamples: high neurons or sharp/inactive class with low gamma effect",
            "",
            "| activation | score range | accuracy range | neuron range | mean neurons | class |",
            "|:-----------|------------:|---------------:|-------------:|-------------:|:------|",
        ]
    )
    for row in sorted(quiet_high_neuron, key=lambda item: float(item["score_range"]))[:12]:
        lines.append(format_row(row))
    if not quiet_high_neuron:
        lines.append("| none in this rerun set | | | | | |")
    lines.append("")


def main() -> int:
    vdp_rows = experiment_rows(ROOT / "ActivationSearch" / "data:VDP" / "analysis" / "gamma_effect.tsv", "h1", "score")
    disc_rows = experiment_rows(ROOT / "ActivationSearch" / "data:analytical" / "gamma_effect.tsv", "near_grad", "near_score")

    lines = [
        "# Gamma Pattern Check",
        "",
        "This checks the hypothesis that gamma has the largest effect when an activation",
        "already uses many neurons or has a sharp inactive/active transition. The check",
        "uses the generated gamma-effect tables and simple name-based activation classes.",
        "",
        "Conclusion: the hypothesis is only locally true for the small VDP rerun set.",
        "It is not a general pattern in the discontinuous-gradient experiment.",
        "",
    ]
    write_section(lines, "VDP HJB", vdp_rows)
    write_section(lines, "Discontinuous Gradient", disc_rows)
    (ROOT / "meta_analysis" / "gamma" / "gamma_pattern_check.md").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
