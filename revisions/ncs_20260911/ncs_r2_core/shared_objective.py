from __future__ import annotations

import numpy as np


def shared_objective(mid_scores, sparse_scores) -> float:
    mid = np.asarray(mid_scores, dtype=float)
    sparse = np.asarray(sparse_scores, dtype=float)
    if mid.size == 0 or sparse.size == 0:
        raise ValueError("both states require at least one noise replicate")
    return float(min(mid.mean(), sparse.mean()))


def wrong_per_noise_minimum(mid_scores, sparse_scores) -> float:
    """Documented anti-pattern, exposed only for a counterexample test."""
    mid = np.asarray(mid_scores, dtype=float)
    sparse = np.asarray(sparse_scores, dtype=float)
    if mid.shape != sparse.shape:
        raise ValueError("per-noise anti-pattern requires aligned arrays")
    return float(np.minimum(mid, sparse).mean())


def select_index(arm: str, mid_values, sparse_values) -> int:
    mid = np.asarray(mid_values, dtype=float)
    sparse = np.asarray(sparse_values, dtype=float)
    if mid.shape != sparse.shape or mid.ndim != 1:
        raise ValueError("candidate state arrays must be one-dimensional and aligned")
    if arm == "A_mid":
        objective = mid
    elif arm == "A_sparse":
        objective = sparse
    elif arm in ("A_shared", "shared_random"):
        objective = np.minimum(mid, sparse)
    else:
        raise ValueError(f"unknown selection arm: {arm}")
    return int(np.argmax(objective))

