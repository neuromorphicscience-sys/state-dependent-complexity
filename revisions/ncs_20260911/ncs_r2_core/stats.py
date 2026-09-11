from __future__ import annotations

import itertools
import math

import numpy as np


def graph_equal_weight(values_by_graph: dict[str, list[float]]) -> tuple[np.ndarray, float]:
    if not values_by_graph or any(len(v) == 0 for v in values_by_graph.values()):
        raise ValueError("each graph needs observations")
    graph_means = np.asarray([np.mean(values_by_graph[key]) for key in sorted(values_by_graph)], dtype=float)
    return graph_means, float(graph_means.mean())


def require_graph_level(values: np.ndarray, graph_ids: list[str]) -> None:
    if values.ndim != 1 or len(values) != len(graph_ids) or len(set(graph_ids)) != len(graph_ids):
        raise ValueError("inference input must contain one aggregate per unique graph")


def bootstrap_mean_ci(graph_values, seed: int, replicates: int = 10_000,
                      alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(graph_values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap requires graph-level vector")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, values.size, size=(replicates, values.size))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(sampled, [alpha / 2, 1 - alpha / 2]))


def exact_two_sided_signflip(differences) -> float:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size > 20:
        raise ValueError("exact sign flip expects <=20 graph differences")
    observed = abs(float(values.mean()))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        extreme += abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15
    return extreme / (2 ** values.size)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def bonferroni_one_sided_ucb(graph_losses, simultaneous_states: int = 2) -> float:
    values = np.asarray(graph_losses, dtype=float)
    if values.size < 2:
        raise ValueError("UCB needs at least two graphs")
    # Fixed normal critical value for one-sided alpha=0.05/2 = 0.025 => 97.5%.
    z = 1.959963984540054
    return float(values.mean() + z * values.std(ddof=1) / math.sqrt(values.size))


def sufficiency_label(mid_graph_losses, sparse_graph_losses, epsilon: float = 0.02) -> dict:
    mid_ucb = bonferroni_one_sided_ucb(mid_graph_losses)
    sparse_ucb = bonferroni_one_sided_ucb(sparse_graph_losses)
    return {"mid_ucb": mid_ucb, "sparse_ucb": sparse_ucb, "epsilon": epsilon,
            "sufficient": bool(mid_ucb <= epsilon and sparse_ucb <= epsilon),
            "wording": "simultaneous_mean_loss_within_epsilon" if mid_ucb <= epsilon and sparse_ucb <= epsilon else "sufficiency_not_established"}


def crossover_transform(values, version: str):
    data = np.asarray(values, dtype=float)
    if version == "legacy_unscaled_v1":
        return data
    if version == "legacy_half_scaled_v1":
        return 0.5 * data
    raise ValueError("crossover version must be explicit")

