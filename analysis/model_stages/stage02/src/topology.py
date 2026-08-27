from __future__ import annotations

import math
import random
from typing import Dict, Any, Tuple

import numpy as np
import torch


def _orient_undirected(adj_u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Orient each undirected edge randomly in one direction."""
    n = adj_u.shape[0]
    a = np.zeros((n, n), dtype=np.float32)
    iu, ju = np.where(np.triu(adj_u, 1) > 0)
    if len(iu) == 0:
        return a
    flip = rng.random(len(iu)) < 0.5
    a[iu[flip], ju[flip]] = 1.0
    a[ju[~flip], iu[~flip]] = 1.0
    return a


def make_er(n: int, p: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    a = (rng.random((n, n)) < p).astype(np.float32)
    np.fill_diagonal(a, 0.0)
    modules = np.zeros(n, dtype=np.int32)
    return a, modules


def make_small_world(n: int, p_target: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Watts-Strogatz-like undirected ring lattice, then random orientation.
    p_target controls approximate density; rewiring probability is fixed at 0.15.
    """
    mean_k = max(2, int(round(p_target * (n - 1))))
    if mean_k % 2:
        mean_k += 1
    mean_k = min(mean_k, n - 2 if (n - 2) % 2 == 0 else n - 3)
    half = max(1, mean_k // 2)

    u = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            u[i, j] = 1
            u[j, i] = 1

    beta = 0.15
    # rewire only clockwise edges
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if rng.random() < beta:
                u[i, j] = u[j, i] = 0
                forbidden = set(np.flatnonzero(u[i]).tolist())
                forbidden.add(i)
                candidates = [x for x in range(n) if x not in forbidden]
                if candidates:
                    k = int(rng.choice(candidates))
                    u[i, k] = u[k, i] = 1
                else:
                    u[i, j] = u[j, i] = 1

    a = _orient_undirected(u, rng)
    modules = np.zeros(n, dtype=np.int32)
    return a, modules


def make_scale_free(n: int, p_target: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple Barabasi-Albert-like graph, then random orientation.
    m is chosen from the target mean density.
    """
    target_mean_degree = max(2.0, p_target * (n - 1))
    m = max(1, int(round(target_mean_degree / 2.0)))
    m = min(m, max(1, n // 8))

    u = np.zeros((n, n), dtype=np.uint8)
    seed_n = min(n, max(m + 1, 3))
    for i in range(seed_n):
        for j in range(i + 1, seed_n):
            u[i, j] = u[j, i] = 1

    degree = u.sum(axis=0).astype(np.float64)
    for new in range(seed_n, n):
        probs = degree[:new] + 1.0
        probs = probs / probs.sum()
        chosen = rng.choice(new, size=min(m, new), replace=False, p=probs)
        for old in chosen:
            u[new, old] = u[old, new] = 1
        degree = u.sum(axis=0).astype(np.float64)

    a = _orient_undirected(u, rng)
    modules = np.zeros(n, dtype=np.int32)
    return a, modules


def make_modular(n: int, p_target: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Directed stochastic block network with four modules.
    Mean density is kept close to p_target.
    """
    n_modules = 4
    modules = np.arange(n, dtype=np.int32) % n_modules
    rng.shuffle(modules)

    p_out = max(0.002, p_target * 0.20)
    # solve approximately for p_in so global density ~= p_target
    frac_same = 1.0 / n_modules
    p_in = (p_target - (1.0 - frac_same) * p_out) / frac_same
    p_in = float(np.clip(p_in, p_target, min(0.8, max(p_target, p_target * 4.0))))

    same = modules[:, None] == modules[None, :]
    probs = np.where(same, p_in, p_out)
    a = (rng.random((n, n)) < probs).astype(np.float32)
    np.fill_diagonal(a, 0.0)
    return a, modules


def build_binary_topology(topology: str, n: int, p: float, seed: int):
    rng = np.random.default_rng(seed)
    if topology == "er":
        return make_er(n, p, rng)
    if topology == "small_world":
        return make_small_world(n, p, rng)
    if topology == "scale_free":
        return make_scale_free(n, p, rng)
    if topology == "modular":
        return make_modular(n, p, rng)
    raise ValueError(f"Unknown topology: {topology}")


def node_scores(a: np.ndarray, modules: np.ndarray, strategy: str, rng: np.random.Generator):
    indeg = a.sum(axis=1).astype(np.float64)   # incoming under row-post, col-pre convention
    outdeg = a.sum(axis=0).astype(np.float64)
    total = indeg + outdeg

    if strategy == "random":
        return rng.random(a.shape[0])

    if strategy == "high_degree":
        return total + 1e-6 * rng.random(a.shape[0])

    if strategy == "feedback_hub":
        return (indeg + 1.0) * (outdeg + 1.0) + 1e-6 * rng.random(a.shape[0])

    if strategy == "module_bridge":
        if np.unique(modules).size <= 1:
            return total + 1e-6 * rng.random(a.shape[0])
        cross = np.zeros(a.shape[0], dtype=np.float64)
        for i in range(a.shape[0]):
            other = modules != modules[i]
            cross[i] = a[i, other].sum() + a[other, i].sum()
        return cross + 0.05 * total + 1e-6 * rng.random(a.shape[0])

    if strategy == "cycle_proxy":
        # directed 3-cycle participation proxy using binary matrix products.
        # score_i = (A^3)_ii, plus a weak degree tie-breaker.
        a64 = a.astype(np.float64, copy=False)
        a2 = a64 @ a64
        cyc = np.sum(a2 * a64.T, axis=1)
        return cyc + 0.01 * total + 1e-6 * rng.random(a.shape[0])

    raise ValueError(f"Unknown placement strategy: {strategy}")


def select_hh_nodes(
    a: np.ndarray,
    modules: np.ndarray,
    n_hh: int,
    strategy: str,
    seed: int,
) -> np.ndarray:
    n = a.shape[0]
    n_hh = int(np.clip(n_hh, 0, n))
    if n_hh == 0:
        return np.array([], dtype=np.int64)
    if n_hh == n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed + 99173)

    if strategy == "coverage_greedy":
        # Greedy structural coverage: prioritize nodes that cover many currently
        # uncovered one-hop neighbors, with degree as a tie-breaker.
        neighbors = (a + a.T) > 0
        degree = neighbors.sum(axis=1).astype(np.float64)
        chosen = []
        covered = np.zeros(n, dtype=bool)
        available = np.ones(n, dtype=bool)
        for _ in range(n_hh):
            gains = ((neighbors & (~covered)[None, :]).sum(axis=1)).astype(np.float64)
            score = gains + 0.05 * degree + 1e-6 * rng.random(n)
            score[~available] = -np.inf
            i = int(np.argmax(score))
            chosen.append(i)
            available[i] = False
            covered[i] = True
            covered |= neighbors[i]
        return np.asarray(chosen, dtype=np.int64)

    scores = node_scores(a, modules, strategy, rng)
    order = np.argsort(scores)[::-1]
    return order[:n_hh].astype(np.int64)


def reorder_hh_prefix(a: np.ndarray, modules: np.ndarray, hh_nodes: np.ndarray):
    n = a.shape[0]
    hh_set = set(int(x) for x in hh_nodes.tolist())
    rest = [i for i in range(n) if i not in hh_set]
    perm = np.asarray(list(hh_nodes) + rest, dtype=np.int64)
    a2 = a[np.ix_(perm, perm)]
    modules2 = modules[perm]
    return a2, modules2, perm


def build_reordered_adjacency(job: Dict[str, Any], n: int, device: torch.device):
    topology = job["topology"]
    p = float(job["connection_prob"])
    seed = int(job["seed"])
    n_hh = int(job["n_hh"])
    strategy = job["placement"]

    a, modules = build_binary_topology(topology, n, p, seed)
    hh_nodes = select_hh_nodes(a, modules, n_hh, strategy, seed)
    a, modules, perm = reorder_hh_prefix(a, modules, hh_nodes)

    actual_density = float(a.sum() / max(n * (n - 1), 1))
    indeg = a.sum(axis=1)
    outdeg = a.sum(axis=0)

    # Normalize by realized mean in-degree rather than target p*N.
    mean_in = float(max(indeg.mean(), 1.0))
    a_norm = a / mean_in

    meta = {
        "actual_density": actual_density,
        "mean_in_degree": float(indeg.mean()),
        "mean_out_degree": float(outdeg.mean()),
    }

    return torch.as_tensor(a_norm, dtype=torch.float32, device=device), meta
