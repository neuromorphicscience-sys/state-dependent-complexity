from __future__ import annotations
import math
import torch

def _vtrap(x, y):
    # numerically stable x/(exp(x/y)-1)
    z = x / y
    return torch.where(torch.abs(z) < 1e-6, y * (1.0 - z / 2.0), x / torch.expm1(z))

def hh_rates(v):
    # Classic HH kinetics, v in mV
    alpha_m = 0.1 * _vtrap(-(v + 40.0), 10.0)
    beta_m  = 4.0 * torch.exp(-(v + 65.0) / 18.0)
    alpha_h = 0.07 * torch.exp(-(v + 65.0) / 20.0)
    beta_h  = 1.0 / (torch.exp(-(v + 35.0) / 10.0) + 1.0)
    alpha_n = 0.01 * _vtrap(-(v + 55.0), 10.0)
    beta_n  = 0.125 * torch.exp(-(v + 65.0) / 80.0)
    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

@torch.no_grad()
def simulate_batch(cfg, hh_fraction, coupling, connection_prob, noise_sigma, seeds):
    """
    Simulate B independent directed random networks in parallel.
    Dense adjacency is intentionally used in v0.1 for simplicity and GPU throughput
    at N<=~512. Later stages should move to sparse/topology-specific kernels.
    """
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    B = len(seeds)
    N = int(cfg["n_neurons"])
    dt = float(cfg["dt_ms"])
    steps = int(round(cfg["duration_ms"] / dt))
    warmup_steps = int(round(cfg["warmup_ms"] / dt))
    bin_steps = max(1, int(round(cfg["record_bin_ms"] / dt)))
    record_bins = max(1, (steps - warmup_steps) // bin_steps)

    lif = cfg["lif"]
    hhcfg = cfg["hh"]
    syn = cfg["synapse"]

    # independent generators per batch item are emulated by deterministic seed offsets
    # and stacking independently generated tensors.
    adjs = []
    hh_masks = []
    for s in seeds:
        g = torch.Generator(device=device)
        g.manual_seed(int(s))
        a = (torch.rand((N, N), generator=g, device=device) < connection_prob).to(dtype)
        a.fill_diagonal_(0.0)
        # column j -> row i convention; normalize by expected in-degree
        a = a / max(connection_prob * N, 1.0)
        adjs.append(a)

        n_hh = int(round(hh_fraction * N))
        perm = torch.randperm(N, generator=g, device=device)
        mask = torch.zeros(N, device=device, dtype=torch.bool)
        if n_hh > 0:
            mask[perm[:n_hh]] = True
        hh_masks.append(mask)

    A = torch.stack(adjs, 0)                     # [B,N,N]
    is_hh = torch.stack(hh_masks, 0)             # [B,N]
    is_lif = ~is_hh

    # states
    v = torch.full((B, N), -65.0, device=device, dtype=dtype)
    am,bm,ah,bh,an,bn = hh_rates(v)
    m = am / (am + bm)
    h = ah / (ah + bh)
    n = an / (an + bn)

    ref = torch.zeros((B, N), device=device, dtype=dtype)
    s_syn = torch.zeros((B, N), device=device, dtype=dtype)

    pop_bins = torch.zeros((B, record_bins), device=device, dtype=dtype)
    spike_counts = torch.zeros((B, N), device=device, dtype=dtype)
    bin_accum = torch.zeros((B,), device=device, dtype=dtype)
    bin_idx = 0

    # HH constants
    C_m = 1.0
    gNa, ENa = 120.0, 50.0
    gK, EK = 36.0, -77.0
    gL, EL = 0.3, -54.387

    last_v = v.clone()

    for t in range(steps):
        # presynaptic synaptic state drives postsynaptic current
        syn_drive = torch.bmm(A, s_syn.unsqueeze(-1)).squeeze(-1)
        # positive current-based coupling for first discovery baseline
        I_syn = coupling * syn_drive * 100.0

        noise = noise_sigma * torch.randn_like(v)

        # ---- LIF ----
        lif_dv = (
            (lif["v_rest"] - v) + lif["bias"] * 20.0 + I_syn + noise
        ) / lif["tau_m_ms"]

        can_update = (ref <= 0.0) & is_lif
        v = torch.where(can_update, v + dt * lif_dv, v)
        ref = torch.clamp(ref - dt, min=0.0)

        lif_spk = is_lif & (v >= lif["v_th"]) & (ref <= 0.0)
        v = torch.where(lif_spk, torch.full_like(v, lif["v_reset"]), v)
        ref = torch.where(
            lif_spk,
            torch.full_like(ref, lif["refractory_ms"]),
            ref
        )

        # ---- HH ----
        am,bm,ah,bh,an,bn = hh_rates(v)
        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n
        m = torch.where(is_hh, m + dt * dm, m)
        h = torch.where(is_hh, h + dt * dh, h)
        n = torch.where(is_hh, n + dt * dn, n)

        INa = gNa * (m ** 3) * h * (v - ENa)
        IK  = gK * (n ** 4) * (v - EK)
        IL  = gL * (v - EL)
        Ihh = hhcfg["bias_uA_cm2"] + I_syn + noise
        hh_dv = (Ihh - INa - IK - IL) / C_m

        hh_new_v = v + dt * hh_dv
        v = torch.where(is_hh, hh_new_v, v)

        # upward 0 mV crossing = HH spike
        hh_spk = is_hh & (last_v < 0.0) & (v >= 0.0)
        spk = lif_spk | hh_spk

        # synaptic exponential trace
        s_syn += dt * (-s_syn / syn["tau_ms"])
        s_syn += spk.to(dtype)

        if t >= warmup_steps:
            spike_counts += spk.to(dtype)
            bin_accum += spk.to(dtype).sum(dim=1)

            rel_t = t - warmup_steps
            if (rel_t + 1) % bin_steps == 0 and bin_idx < record_bins:
                # convert bin spikes to per-neuron Hz
                seconds = (bin_steps * dt) / 1000.0
                pop_bins[:, bin_idx] = bin_accum / N / seconds
                bin_accum.zero_()
                bin_idx += 1

        last_v = v.clone()

    return {
        "pop_rate_hz": pop_bins[:, :bin_idx].cpu().numpy(),
        "spike_counts": spike_counts.cpu().numpy(),
        "hh_count": is_hh.sum(dim=1).cpu().numpy(),
        "device": str(device),
    }
