from __future__ import annotations

from typing import List, Dict, Any
import torch


def _vtrap(x: torch.Tensor, y: float) -> torch.Tensor:
    """Numerically stable x / (exp(x/y) - 1)."""
    z = x / y
    return torch.where(
        torch.abs(z) < 1e-6,
        y * (1.0 - z / 2.0),
        x / torch.expm1(z),
    )


def hh_rates(v: torch.Tensor):
    """Classic Hodgkin-Huxley gating kinetics. v in mV."""
    alpha_m = 0.1 * _vtrap(-(v + 40.0), 10.0)
    beta_m = 4.0 * torch.exp(-(v + 65.0) / 18.0)

    alpha_h = 0.07 * torch.exp(-(v + 65.0) / 20.0)
    beta_h = 1.0 / (torch.exp(-(v + 35.0) / 10.0) + 1.0)

    alpha_n = 0.01 * _vtrap(-(v + 55.0), 10.0)
    beta_n = 0.125 * torch.exp(-(v + 65.0) / 80.0)

    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


@torch.no_grad()
def simulate_batch(cfg: Dict[str, Any], jobs: List[Dict[str, Any]]):
    """
    v0.2 mixed-parameter GPU super-batch.

    Each job may have different:
      - hh_fraction
      - coupling
      - connection_prob
      - noise_sigma
      - seed

    Performance changes vs v0.1:
      1. Mixed parameter points can coexist in one GPU batch.
      2. HH gating kinetics are computed only on the HH prefix actually needed.
      3. No per-step full-tensor last_v.clone().
      4. Random noise is generated in chunks.
      5. Batch-level parameter tensors avoid repeated Python-side branching.

    Scientific model remains the same as the Stage-1 v0.1 baseline:
      - directed ER networks
      - excitatory current-based coupling
      - classic HH neurons + LIF neurons
      - Gaussian noise
    """
    if not jobs:
        raise ValueError("simulate_batch received an empty jobs list")

    requested_device = cfg.get("device", "cuda")
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested_device)
    else:
        device = torch.device("cpu")

    dtype = torch.float32
    B = len(jobs)
    N = int(cfg["n_neurons"])
    dt = float(cfg["dt_ms"])
    steps = int(round(float(cfg["duration_ms"]) / dt))
    warmup_steps = int(round(float(cfg["warmup_ms"]) / dt))
    bin_steps = max(1, int(round(float(cfg["record_bin_ms"]) / dt)))
    record_bins = max(1, (steps - warmup_steps) // bin_steps)

    lif = cfg["lif"]
    hhcfg = cfg["hh"]
    syn = cfg["synapse"]

    # ------------------------------------------------------------------
    # Batch parameters
    # ------------------------------------------------------------------
    hh_fractions = torch.tensor(
        [float(j["hh_fraction"]) for j in jobs],
        device=device,
        dtype=dtype,
    )
    couplings = torch.tensor(
        [float(j["coupling"]) for j in jobs],
        device=device,
        dtype=dtype,
    )
    conn_probs = torch.tensor(
        [float(j["connection_prob"]) for j in jobs],
        device=device,
        dtype=dtype,
    )
    noise_sigmas = torch.tensor(
        [float(j["noise_sigma"]) for j in jobs],
        device=device,
        dtype=dtype,
    )

    hh_counts = torch.round(hh_fractions * N).to(torch.long)
    hh_counts = torch.clamp(hh_counts, 0, N)

    # HH neurons occupy a deterministic prefix [0:n_hh).
    # For an exchangeable ER topology this is statistically equivalent to
    # randomly assigning neuron labels, while enabling prefix-only HH kinetics.
    idx = torch.arange(N, device=device).unsqueeze(0)  # [1,N]
    is_hh = idx < hh_counts.unsqueeze(1)               # [B,N]
    is_lif = ~is_hh

    # Largest HH prefix required by any network in this super-batch.
    max_hh = int(hh_counts.max().item())

    # ------------------------------------------------------------------
    # Directed ER adjacency, independently seeded per realization
    # ------------------------------------------------------------------
    adjs = []
    for b, job in enumerate(jobs):
        g = torch.Generator(device=device)
        g.manual_seed(int(job["seed"]))

        p = float(job["connection_prob"])
        a = (
            torch.rand((N, N), generator=g, device=device, dtype=dtype) < p
        ).to(dtype)
        a.fill_diagonal_(0.0)

        # Normalize by expected in-degree, matching v0.1 semantics.
        a = a / max(p * N, 1.0)
        adjs.append(a)

    A = torch.stack(adjs, dim=0)  # [B,N,N]

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------
    v = torch.full((B, N), -65.0, device=device, dtype=dtype)
    ref = torch.zeros((B, N), device=device, dtype=dtype)
    s_syn = torch.zeros((B, N), device=device, dtype=dtype)

    # Only allocate HH gating variables for the largest HH prefix.
    if max_hh > 0:
        v_hh0 = v[:, :max_hh]
        am, bm, ah, bh, an, bn = hh_rates(v_hh0)
        m = am / (am + bm)
        h = ah / (ah + bh)
        n_gate = an / (an + bn)

        hh_active_prefix = (
            torch.arange(max_hh, device=device).unsqueeze(0)
            < hh_counts.unsqueeze(1)
        )
    else:
        m = h = n_gate = None
        hh_active_prefix = None

    pop_bins = torch.zeros((B, record_bins), device=device, dtype=dtype)
    spike_counts = torch.zeros((B, N), device=device, dtype=dtype)
    bin_accum = torch.zeros((B,), device=device, dtype=dtype)
    bin_idx = 0

    # HH constants
    C_m = 1.0
    gNa, ENa = 120.0, 50.0
    gK, EK = 36.0, -77.0
    gL, EL = 0.3, -54.387

    # RNG chunking
    noise_chunk_steps = max(1, int(cfg.get("noise_chunk_steps", 256)))
    noise_chunk = None
    noise_chunk_pos = noise_chunk_steps
    current_chunk_len = 0

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    for t in range(steps):
        # Synaptic population drive: [B,N,N] x [B,N,1] -> [B,N]
        syn_drive = torch.bmm(A, s_syn.unsqueeze(-1)).squeeze(-1)

        # Per-network coupling strength.
        I_syn = couplings.unsqueeze(1) * syn_drive * 100.0

        # Generate random noise in blocks rather than once per time step.
        if noise_chunk is None or noise_chunk_pos >= current_chunk_len:
            current_chunk_len = min(noise_chunk_steps, steps - t)
            noise_chunk = torch.randn(
                (current_chunk_len, B, N),
                device=device,
                dtype=dtype,
            )
            noise_chunk_pos = 0

        noise = noise_chunk[noise_chunk_pos] * noise_sigmas.unsqueeze(1)
        noise_chunk_pos += 1

        # --------------------------------------------------------------
        # LIF update
        # --------------------------------------------------------------
        lif_dv = (
            (float(lif["v_rest"]) - v)
            + float(lif["bias"]) * 20.0
            + I_syn
            + noise
        ) / float(lif["tau_m_ms"])

        can_update_lif = (ref <= 0.0) & is_lif
        v_lif_candidate = v + dt * lif_dv
        v = torch.where(can_update_lif, v_lif_candidate, v)

        ref = torch.clamp(ref - dt, min=0.0)

        lif_spk = is_lif & (v >= float(lif["v_th"])) & (ref <= 0.0)
        v = torch.where(
            lif_spk,
            torch.full_like(v, float(lif["v_reset"])),
            v,
        )
        ref = torch.where(
            lif_spk,
            torch.full_like(ref, float(lif["refractory_ms"])),
            ref,
        )

        # Start with LIF spikes.
        spk = lif_spk.clone()

        # --------------------------------------------------------------
        # HH update only on prefix [:max_hh]
        # --------------------------------------------------------------
        if max_hh > 0:
            v_hh_old = v[:, :max_hh]

            am, bm, ah, bh, an, bn = hh_rates(v_hh_old)

            dm = am * (1.0 - m) - bm * m
            dh = ah * (1.0 - h) - bh * h
            dn = an * (1.0 - n_gate) - bn * n_gate

            # Update only HH-active positions inside the prefix.
            m_new = m + dt * dm
            h_new = h + dt * dh
            n_new = n_gate + dt * dn

            m = torch.where(hh_active_prefix, m_new, m)
            h = torch.where(hh_active_prefix, h_new, h)
            n_gate = torch.where(hh_active_prefix, n_new, n_gate)

            INa = gNa * (m ** 3) * h * (v_hh_old - ENa)
            IK = gK * (n_gate ** 4) * (v_hh_old - EK)
            IL = gL * (v_hh_old - EL)

            Ihh = (
                float(hhcfg["bias_uA_cm2"])
                + I_syn[:, :max_hh]
                + noise[:, :max_hh]
            )

            hh_dv = (Ihh - INa - IK - IL) / C_m
            hh_new_v = v_hh_old + dt * hh_dv

            # Only genuine HH locations are allowed to update/spike.
            hh_spk_small = (
                hh_active_prefix
                & (v_hh_old < 0.0)
                & (hh_new_v >= 0.0)
            )

            v_hh_final = torch.where(
                hh_active_prefix,
                hh_new_v,
                v_hh_old,
            )
            v[:, :max_hh] = v_hh_final

            spk[:, :max_hh] = torch.where(
                hh_active_prefix,
                hh_spk_small,
                spk[:, :max_hh],
            )

        # --------------------------------------------------------------
        # Synaptic trace
        # --------------------------------------------------------------
        s_syn += dt * (-s_syn / float(syn["tau_ms"]))
        s_syn += spk.to(dtype)

        # --------------------------------------------------------------
        # Recording
        # --------------------------------------------------------------
        if t >= warmup_steps:
            spike_counts += spk.to(dtype)
            bin_accum += spk.to(dtype).sum(dim=1)

            rel_t = t - warmup_steps
            if (rel_t + 1) % bin_steps == 0 and bin_idx < record_bins:
                seconds = (bin_steps * dt) / 1000.0
                pop_bins[:, bin_idx] = bin_accum / N / seconds
                bin_accum.zero_()
                bin_idx += 1

    return {
        "pop_rate_hz": pop_bins[:, :bin_idx].cpu().numpy(),
        "spike_counts": spike_counts.cpu().numpy(),
        "hh_count": hh_counts.cpu().numpy(),
        "device": str(device),
    }
