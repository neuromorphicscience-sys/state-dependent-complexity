from __future__ import annotations
import itertools
import os
import pandas as pd
import numpy as np

def parameter_grid(cfg):
    for p_hh, g, p_conn, sigma in itertools.product(
        cfg["hh_fractions"],
        cfg["couplings"],
        cfg["connection_probs"],
        cfg["noise_sigmas"],
    ):
        yield {
            "hh_fraction": float(p_hh),
            "coupling": float(g),
            "connection_prob": float(p_conn),
            "noise_sigma": float(sigma),
        }

def detect_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["candidate_sparse_rhythm"] = (
        (out["hh_fraction"] <= 0.05) &
        (out["hh_fraction"] > 0) &
        (out["rhythm_score"] >= out["rhythm_score"].quantile(0.90))
    )

    # Aggregate before detecting non-monotonic HH optima
    group_cols = ["coupling", "connection_prob", "noise_sigma", "hh_fraction"]
    agg = out.groupby(group_cols, as_index=False)["rhythm_score"].mean()
    agg["candidate_local_optimum"] = False
    for key, sub in agg.groupby(["coupling", "connection_prob", "noise_sigma"]):
        sub = sub.sort_values("hh_fraction")
        idx = sub.index.to_list()
        vals = sub["rhythm_score"].to_numpy()
        fracs = sub["hh_fraction"].to_numpy()
        for i in range(1, len(sub)-1):
            if 0 < fracs[i] < 1 and vals[i] > vals[i-1] * 1.10 and vals[i] > vals[i+1] * 1.10:
                agg.loc[idx[i], "candidate_local_optimum"] = True

    # transition score: largest finite difference in mean rhythm score along HH fraction
    agg["transition_strength"] = 0.0
    for key, sub in agg.groupby(["coupling", "connection_prob", "noise_sigma"]):
        sub = sub.sort_values("hh_fraction")
        idx = sub.index.to_list()
        vals = sub["rhythm_score"].to_numpy()
        fracs = sub["hh_fraction"].to_numpy()
        if len(vals) > 1:
            dv = np.abs(np.diff(vals))
            dp = np.maximum(np.diff(fracs), 1e-9)
            strength = dv / dp
            for j, s in enumerate(strength):
                agg.loc[idx[j+1], "transition_strength"] = float(s)

    return agg.sort_values(
        ["candidate_local_optimum", "transition_strength", "rhythm_score"],
        ascending=[False, False, False]
    )

def save_incremental(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "runs.csv")
    df_new = pd.DataFrame(rows)
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["run_id"], keep="last")
    else:
        df = df_new
    df.to_csv(path, index=False)

    cand = detect_candidates(df)
    cand.to_csv(os.path.join(output_dir, "candidates.csv"), index=False)
    return df, cand
