from __future__ import annotations
import os
import numpy as np
import pandas as pd


def save_incremental(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "runs_stage1b.csv")
    df_new = pd.DataFrame(rows)

    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["run_id"], keep="last")
    else:
        df = df_new

    df.to_csv(path, index=False)
    return df


def aggregate_parameter_points(df):
    group_cols = [
        "window",
        "hh_fraction",
        "coupling",
        "connection_prob",
        "noise_sigma",
    ]
    agg = df.groupby(group_cols, as_index=False).agg(
        rhythm_score_mean=("rhythm_score", "mean"),
        rhythm_score_std=("rhythm_score", "std"),
        mean_rate_hz=("mean_rate_hz", "mean"),
        dominant_frequency_hz=("dominant_frequency_hz", "mean"),
        spectral_concentration=("spectral_concentration", "mean"),
        spectral_entropy=("spectral_entropy", "mean"),
        synchrony_proxy=("synchrony_proxy", "mean"),
        silent_fraction=("silent_fraction", "mean"),
        n=("run_id", "count"),
    )
    return agg


def attach_baselines_and_emergence(agg):
    """
    Baselines are defined within each (coupling, p_conn, noise) condition:
      LIF baseline = pHH=0
      HH baseline  = pHH=1
    Emergence gain compares mixed networks to linear interpolation
    between those two homogeneous endpoints.
    """
    df = agg.copy()
    keys = ["coupling", "connection_prob", "noise_sigma"]

    lif = (
        df[np.isclose(df["hh_fraction"], 0.0)]
        .groupby(keys, as_index=False)["rhythm_score_mean"]
        .mean()
        .rename(columns={"rhythm_score_mean": "lif_baseline_score"})
    )

    hh = (
        df[np.isclose(df["hh_fraction"], 1.0)]
        .groupby(keys, as_index=False)["rhythm_score_mean"]
        .mean()
        .rename(columns={"rhythm_score_mean": "hh_baseline_score"})
    )

    df = df.merge(lif, on=keys, how="left")
    df = df.merge(hh, on=keys, how="left")

    p = df["hh_fraction"].astype(float)
    linear_expectation = (
        (1.0 - p) * df["lif_baseline_score"]
        + p * df["hh_baseline_score"]
    )
    df["linear_mixture_expectation"] = linear_expectation
    df["emergence_gain"] = df["rhythm_score_mean"] - linear_expectation

    # Complexity efficiency: gain over all-LIF per HH fraction.
    # Only meaningful for mixed networks with pHH > 0.
    df["complexity_efficiency"] = np.where(
        p > 0,
        (df["rhythm_score_mean"] - df["lif_baseline_score"]) / p,
        np.nan,
    )

    return df


def detect_transitions(df):
    out = df.copy()
    out["rhythm_transition_strength"] = 0.0
    out["frequency_transition_strength"] = 0.0

    group_cols = ["coupling", "connection_prob", "noise_sigma"]

    for _, sub in out.groupby(group_cols):
        sub = sub.sort_values("hh_fraction")
        idx = sub.index.to_list()
        p = sub["hh_fraction"].to_numpy(dtype=float)
        r = sub["rhythm_score_mean"].to_numpy(dtype=float)
        f = sub["dominant_frequency_hz"].to_numpy(dtype=float)

        if len(sub) < 2:
            continue

        dp = np.maximum(np.diff(p), 1e-9)
        dr = np.abs(np.diff(r)) / dp
        dfreq = np.abs(np.diff(f)) / dp

        for j in range(len(dr)):
            out.loc[idx[j+1], "rhythm_transition_strength"] = float(dr[j])
            out.loc[idx[j+1], "frequency_transition_strength"] = float(dfreq[j])

    return out


def build_candidates(df):
    out = df.copy()

    out["candidate_sparse_efficiency"] = (
        (out["hh_fraction"] > 0)
        & (out["hh_fraction"] <= 0.05)
        & (out["complexity_efficiency"]
           >= out["complexity_efficiency"].quantile(0.90))
    )

    out["candidate_positive_emergence"] = (
        out["emergence_gain"]
        >= out["emergence_gain"].quantile(0.90)
    )

    out["candidate_frequency_transition"] = (
        out["frequency_transition_strength"]
        >= out["frequency_transition_strength"].quantile(0.95)
    )

    out["candidate_rhythm_transition"] = (
        out["rhythm_transition_strength"]
        >= out["rhythm_transition_strength"].quantile(0.95)
    )

    return out.sort_values(
        [
            "candidate_positive_emergence",
            "candidate_sparse_efficiency",
            "candidate_frequency_transition",
            "emergence_gain",
            "complexity_efficiency",
        ],
        ascending=[False, False, False, False, False],
    )
