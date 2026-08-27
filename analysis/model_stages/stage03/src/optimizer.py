from __future__ import annotations
import numpy as np

from src.topology import build_binary_topology, node_feature_table


def surrogate_rank_nodes(topology, n, p, seed):
    """
    Interpretable GPU-search precursor:
    construct a composite node score from normalized structural features.
    Later rounds can replace this with learned/evolutionary placement.
    """
    a, modules = build_binary_topology(topology, n, p, seed)
    f = node_feature_table(a, modules)

    names = [
        "total_degree",
        "feedback_score",
        "cycle3",
        "module_bridge",
        "spectral_centrality",
    ]

    Z = []
    for name in names:
        x = np.asarray(f[name], dtype=float)
        z = (x - x.mean()) / (x.std() + 1e-9)
        Z.append(z)

    # balanced initial composite; analysis will regress observed success back
    # onto selected feature means to learn which terms matter.
    score = (
        0.25 * Z[0]
        + 0.20 * Z[1]
        + 0.20 * Z[2]
        + 0.15 * Z[3]
        + 0.20 * Z[4]
    )
    return np.argsort(score)[::-1]
