from __future__ import annotations
from typing import List, Dict, Any
import torch

from src.topology import prepare_network


def _vtrap(x: torch.Tensor, y: float):
    z = x / y
    return torch.where(torch.abs(z) < 1e-6, y * (1.0 - z / 2.0), x / torch.expm1(z))


def hh_rates(v):
    am = 0.1 * _vtrap(-(v + 40.0), 10.0)
    bm = 4.0 * torch.exp(-(v + 65.0) / 18.0)
    ah = 0.07 * torch.exp(-(v + 65.0) / 20.0)
    bh = 1.0 / (torch.exp(-(v + 35.0) / 10.0) + 1.0)
    an = 0.01 * _vtrap(-(v + 55.0), 10.0)
    bn = 0.125 * torch.exp(-(v + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def _lif_step(v, ref, is_lif, I_syn, noise, dt, lif):
    dv = (
        (float(lif["v_rest"]) - v)
        + float(lif["bias"]) * 20.0
        + I_syn + noise
    ) / float(lif["tau_m_ms"])

    can_update = (ref <= 0.0) & is_lif
    v = torch.where(can_update, v + dt * dv, v)
    ref = torch.clamp(ref - dt, min=0.0)

    spk = is_lif & (v >= float(lif["v_th"])) & (ref <= 0.0)
    v = torch.where(spk, torch.full_like(v, float(lif["v_reset"])), v)
    ref = torch.where(spk, torch.full_like(ref, float(lif["refractory_ms"])), ref)
    return v, ref, spk


@torch.no_grad()
def simulate_batch(cfg: Dict[str, Any], jobs: List[Dict[str, Any]]):
    device = torch.device(
        cfg.get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )
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

    adjs, metas = [], []
    for j in jobs:
        a, meta = prepare_network(j, N, device)
        adjs.append(a)
        metas.append(meta)
    A = torch.stack(adjs, dim=0)

    n_hh = torch.tensor([int(j["n_hh"]) for j in jobs], device=device, dtype=torch.long)
    couplings = torch.tensor([float(j["coupling"]) for j in jobs], device=device, dtype=dtype)
    noise_sigmas = torch.tensor([float(j["noise_sigma"]) for j in jobs], device=device, dtype=dtype)

    idx = torch.arange(N, device=device).unsqueeze(0)
    is_hh = idx < n_hh.unsqueeze(1)
    is_lif = ~is_hh
    max_hh = int(n_hh.max().item())

    v = torch.full((B, N), -65.0, device=device, dtype=dtype)
    ref = torch.zeros((B, N), device=device, dtype=dtype)
    s_syn = torch.zeros((B, N), device=device, dtype=dtype)

    if max_hh > 0:
        v0 = v[:, :max_hh]
        am,bm,ah,bh,an,bn = hh_rates(v0)
        m = am/(am+bm)
        h = ah/(ah+bh)
        ng = an/(an+bn)
        active_hh = torch.arange(max_hh, device=device).unsqueeze(0) < n_hh.unsqueeze(1)
    else:
        m=h=ng=active_hh=None

    pop_bins = torch.zeros((B, record_bins), device=device, dtype=dtype)
    spike_counts = torch.zeros((B, N), device=device, dtype=dtype)
    bin_accum = torch.zeros((B,), device=device, dtype=dtype)
    bin_idx = 0

    C_m = 1.0
    gNa, ENa = 120.0, 50.0
    gK, EK = 36.0, -77.0
    gL, EL = 0.3, -54.387

    chunk_steps = max(1, int(cfg.get("noise_chunk_steps", 256)))
    noise_chunk = None
    noise_pos = chunk_steps
    current_len = 0

    for t in range(steps):
        # all recurrent propagation remains on GPU
        syn_drive = torch.bmm(A, s_syn.unsqueeze(-1)).squeeze(-1)
        I_syn = couplings.unsqueeze(1) * syn_drive * 100.0

        if noise_chunk is None or noise_pos >= current_len:
            current_len = min(chunk_steps, steps - t)
            noise_chunk = torch.randn((current_len, B, N), device=device, dtype=dtype)
            noise_pos = 0
        noise = noise_chunk[noise_pos] * noise_sigmas.unsqueeze(1)
        noise_pos += 1

        v, ref, lif_spk = _lif_step(v, ref, is_lif, I_syn, noise, dt, lif)
        spk = lif_spk.clone()

        if max_hh > 0:
            vold = v[:, :max_hh]
            am,bm,ah,bh,an,bn = hh_rates(vold)
            m2 = m + dt*(am*(1.0-m)-bm*m)
            h2 = h + dt*(ah*(1.0-h)-bh*h)
            ng2 = ng + dt*(an*(1.0-ng)-bn*ng)

            m = torch.where(active_hh, m2, m)
            h = torch.where(active_hh, h2, h)
            ng = torch.where(active_hh, ng2, ng)

            INa = gNa*(m**3)*h*(vold-ENa)
            IK = gK*(ng**4)*(vold-EK)
            IL = gL*(vold-EL)
            Ihh = float(hhcfg["bias_uA_cm2"]) + I_syn[:, :max_hh] + noise[:, :max_hh]
            vnew = vold + dt*((Ihh-INa-IK-IL)/C_m)

            hh_spk = active_hh & (vold < 0.0) & (vnew >= 0.0)
            v[:, :max_hh] = torch.where(active_hh, vnew, vold)
            spk[:, :max_hh] = torch.where(active_hh, hh_spk, spk[:, :max_hh])

        s_syn += dt*(-s_syn/float(syn["tau_ms"]))
        s_syn += spk.to(dtype)

        if t >= warmup_steps:
            spike_counts += spk.to(dtype)
            bin_accum += spk.to(dtype).sum(dim=1)
            rel = t - warmup_steps
            if (rel + 1) % bin_steps == 0 and bin_idx < record_bins:
                seconds = (bin_steps * dt) / 1000.0
                pop_bins[:, bin_idx] = bin_accum / N / seconds
                bin_accum.zero_()
                bin_idx += 1

    return {
        "pop_rate_hz": pop_bins[:, :bin_idx].cpu().numpy(),
        "spike_counts": spike_counts.cpu().numpy(),
        "device": str(device),
        "network_meta": metas,
    }
