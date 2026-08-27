from __future__ import annotations
import numpy as np
import torch


def _orient_undirected(u, rng):
    n = u.shape[0]
    a = np.zeros((n, n), dtype=np.float32)
    ii, jj = np.where(np.triu(u, 1) > 0)
    if len(ii):
        flip = rng.random(len(ii)) < 0.5
        a[ii[flip], jj[flip]] = 1.0
        a[jj[~flip], ii[~flip]] = 1.0
    return a


def make_er(n, p, rng):
    a = (rng.random((n, n)) < p).astype(np.float32)
    np.fill_diagonal(a, 0)
    modules = np.zeros(n, dtype=np.int32)
    return a, modules


def make_small_world(n, p, rng):
    k = max(2, int(round(p * (n - 1))))
    if k % 2:
        k += 1
    half = max(1, k // 2)
    u = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            u[i, j] = u[j, i] = 1

    beta = 0.15
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if rng.random() < beta:
                u[i, j] = u[j, i] = 0
                forbidden = set(np.flatnonzero(u[i]).tolist())
                forbidden.add(i)
                candidates = [x for x in range(n) if x not in forbidden]
                if candidates:
                    q = int(rng.choice(candidates))
                    u[i, q] = u[q, i] = 1
                else:
                    u[i, j] = u[j, i] = 1
    return _orient_undirected(u, rng), np.zeros(n, dtype=np.int32)


def make_scale_free(n, p, rng):
    target_mean_degree = max(2.0, p * (n - 1))
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
        probs /= probs.sum()
        chosen = rng.choice(new, size=min(m, new), replace=False, p=probs)
        for old in chosen:
            u[new, old] = u[old, new] = 1
        degree = u.sum(axis=0).astype(np.float64)

    return _orient_undirected(u, rng), np.zeros(n, dtype=np.int32)


def make_modular(n, p, rng):
    nmod = 4
    modules = np.arange(n, dtype=np.int32) % nmod
    rng.shuffle(modules)
    p_out = max(0.002, p * 0.2)
    frac_same = 1.0 / nmod
    p_in = (p - (1.0 - frac_same) * p_out) / frac_same
    p_in = float(np.clip(p_in, p, min(0.8, max(p, 4.0 * p))))
    same = modules[:, None] == modules[None, :]
    probs = np.where(same, p_in, p_out)
    a = (rng.random((n, n)) < probs).astype(np.float32)
    np.fill_diagonal(a, 0)
    return a, modules


def build_binary_topology(kind, n, p, seed):
    rng = np.random.default_rng(seed)
    if kind == "er":
        return make_er(n, p, rng)
    if kind == "small_world":
        return make_small_world(n, p, rng)
    if kind == "scale_free":
        return make_scale_free(n, p, rng)
    if kind == "modular":
        return make_modular(n, p, rng)
    raise ValueError(kind)


def node_feature_table(a, modules):
    indeg = a.sum(axis=1).astype(np.float64)
    outdeg = a.sum(axis=0).astype(np.float64)
    total = indeg + outdeg

    a64 = a.astype(np.float64, copy=False)
    a2 = a64 @ a64
    cycle3 = np.sum(a2 * a64.T, axis=1)

    bridge = np.zeros(len(a), dtype=np.float64)
    if np.unique(modules).size > 1:
        for i in range(len(a)):
            other = modules != modules[i]
            bridge[i] = a[i, other].sum() + a[other, i].sum()

    # power-iteration spectral centrality proxy
    v = np.ones(len(a), dtype=np.float64) / max(len(a), 1)
    sym = a64 + a64.T
    for _ in range(30):
        nv = sym @ v
        norm = np.linalg.norm(nv)
        if norm < 1e-12:
            break
        v = nv / norm
    spectral = np.abs(v)

    return {
        "in_degree": indeg,
        "out_degree": outdeg,
        "total_degree": total,
        "feedback_score": (indeg + 1.0) * (outdeg + 1.0),
        "cycle3": cycle3,
        "module_bridge": bridge,
        "spectral_centrality": spectral,
    }


def heuristic_select(a, modules, n_hh, strategy, seed):
    n = len(a)
    n_hh = int(np.clip(n_hh, 0, n))
    if n_hh == 0:
        return np.array([], dtype=np.int64)
    if n_hh == n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed + 99173)
    f = node_feature_table(a, modules)

    if strategy == "random":
        return rng.choice(n, size=n_hh, replace=False).astype(np.int64)
    if strategy == "high_degree":
        score = f["total_degree"]
    elif strategy == "feedback_hub":
        score = f["feedback_score"]
    elif strategy == "module_bridge":
        score = f["module_bridge"] + 0.05 * f["total_degree"]
    elif strategy == "cycle_proxy":
        score = f["cycle3"] + 0.01 * f["total_degree"]
    elif strategy == "spectral":
        score = f["spectral_centrality"]
    else:
        raise ValueError(strategy)

    score = score + 1e-9 * rng.random(n)
    return np.argsort(score)[::-1][:n_hh].astype(np.int64)


def reorder(a, modules, hh_nodes):
    n = len(a)
    s = set(int(x) for x in hh_nodes.tolist())
    rest = [i for i in range(n) if i not in s]
    perm = np.asarray(list(hh_nodes) + rest, dtype=np.int64)
    return a[np.ix_(perm, perm)], modules[perm], perm


def prepare_network(job, n, device):
    a, modules = build_binary_topology(job["topology"], n, float(job["connection_prob"]), int(job["seed"]))
    n_hh = int(job["n_hh"])
    placement = job["placement"]

    if placement == "none":
        hh_nodes = np.arange(n_hh, dtype=np.int64)
    else:
        hh_nodes = heuristic_select(a, modules, n_hh, placement, int(job["seed"]))

    features = node_feature_table(a, modules)
    selected_feature_means = {}
    if n_hh > 0:
        for k, arr in features.items():
            selected_feature_means[f"selected_{k}_mean"] = float(np.mean(arr[hh_nodes]))
    else:
        for k in features:
            selected_feature_means[f"selected_{k}_mean"] = 0.0

    a2, modules2, perm = reorder(a, modules, hh_nodes)

    indeg = a2.sum(axis=1)
    mean_in = float(max(indeg.mean(), 1.0))
    a2 = a2 / mean_in

    meta = {
        "actual_density": float((a2 > 0).sum() / max(n * (n - 1), 1)),
        "mean_in_degree": float((a2 > 0).sum(axis=1).mean()),
        **selected_feature_means,
    }

    return torch.as_tensor(a2, dtype=torch.float32, device=device), meta
