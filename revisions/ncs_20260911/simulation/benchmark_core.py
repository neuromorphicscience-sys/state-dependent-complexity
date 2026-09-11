"""Versioned driven HH/LIF benchmark; frozen source remains read-only."""
from __future__ import annotations
import datetime as dtm
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(__file__).resolve().parents[3] / 'analysis/model_stages/stage04a/formal_frozen/src'
# This public core does not launch a search. An external runner must explicitly
# provide its own deadline before using the optional resource guard.
DEADLINE = float(os.environ.get('NCS_RUN_DEADLINE_EPOCH', '0'))
DISK_LIMIT = 30_000_000_000

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

frozen = load_module('frozen_stage4_sim', FROZEN / 'stage4_gpu_simulator.py')
topology = load_module('frozen_stage4_top', FROZEN / 'stage4_topology.py')
CFG = dict(dt_ms=.05, duration_ms=400., warmup_ms=100., record_bin_ms=5.,
           noise_chunk_steps=256,
           lif=dict(v_rest=-65., v_reset=-65., v_th=-50., tau_m_ms=20., refractory_ms=2., bias=.75),
           hh=dict(bias_uA_cm2=9.5), synapse=dict(tau_ms=5., e_rev_mv=0.))
TASKS = ('delayed_recall', 'nonlinear_integration', 'pulse_order',
         'interval', 'evidence_accumulation', 'context_recall')

def seed(*parts):
    return int.from_bytes(hashlib.sha256(('NCS06-v1|'+'|'.join(map(str, parts))).encode()).digest()[:4], 'little')

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)

def guard(reserve_seconds=0):
    if time.time() + reserve_seconds >= DEADLINE:
        raise RuntimeError('Authorized deadline reached / insufficient reserve')
    size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if size > DISK_LIMIT - 500_000_000:
        raise RuntimeError(f'Disk budget guard: {size} bytes')
    return size

def graph(graph_id, n=256):
    a, modules = topology.build_topology('scale_free', n, .05, seed('graph', graph_id))
    return a, modules

def input_projection(graph_id, n=256, channels=4):
    rng = np.random.default_rng(seed('projection', graph_id))
    # Disjoint, fixed node groups; no dependence on HH membership or target label.
    order = rng.permutation(n)
    w = np.zeros((channels, n), np.float32)
    for c, nodes in enumerate(np.array_split(order, channels)):
        w[c, nodes] = 1.
    return w

def task_data(task, role, block, trials=64, amplitude=5., version='v1'):
    """Independent episodic input; targets never supplied to the reservoir.

    60 bins of 5 ms after 100 ms warm-up. Last 40 ms is always blank.
    Channels 0/1 sensory, channel 2 cue, channel 3 distractor.
    Fixed readout is the neuron-wise exponentially filtered spike state at end.
    """
    rng = np.random.default_rng(seed('data', version, task, role, block))
    x = np.zeros((trials, 60, 4), np.float32)
    labels = np.tile(np.array([-1., 1.]), (trials+1)//2)[:trials]
    rng.shuffle(labels)
    nuisance = rng.uniform(.85, 1.15, trials).astype(np.float32)
    if task == 'delayed_recall':
        y = rng.uniform(-1., 1., trials).astype(np.float32)
        x[:, 28:36, 0] = (y*nuisance)[:, None]
        x[:, 40:48, 3] = rng.uniform(-1., 1., (trials, 1))
    elif task == 'nonlinear_integration':
        a = rng.choice([-1., 1.], trials)
        b = labels*a
        x[:, 22:30, 0] = (a*nuisance)[:, None]
        x[:, 36:44, 1] = (b*nuisance)[:, None]
        y = labels
    elif task == 'pulse_order':
        y = labels
        for i, label in enumerate(labels):
            jitter = int(rng.integers(-2, 3))
            first, second = (0, 1) if label > 0 else (1, 0)
            if version in ('challenge_v2','challenge_v3'):
                x[i, 24+jitter:27+jitter, first] = nuisance[i]
                x[i, 28+jitter:31+jitter, second] = nuisance[i]
                # Identical late masking pulse in both channels, independent of label.
                x[i, 44:48, :2] = rng.uniform(.5,1.)
            else:
                x[i, 22+jitter:28+jitter, first] = nuisance[i]
                x[i, 40+jitter:46+jitter, second] = nuisance[i]
    elif task == 'interval':
        y = labels
        for i, label in enumerate(labels):
            end = int(rng.integers(42, 47))
            gap = (12 if version in ('challenge_v2','challenge_v3') else 18) if label > 0 else 8
            x[i, end-gap-4:end-gap, 0] = nuisance[i]
            x[i, end-4:end, 0] = nuisance[i]
    elif task == 'evidence_accumulation':
        y = labels
        # Marginal last sample is intentionally noisy; label is latent drift.
        x[:, 12:48, 0] = .20*labels[:, None] + rng.normal(0., .8, (trials, 36))
    elif task == 'context_recall':
        cue = rng.choice([-1., 1.], trials)
        other = rng.choice([-1., 1.], trials)
        a = np.where(cue > 0, labels, other)
        b = np.where(cue > 0, other, labels)
        x[:, 16:24, 2] = cue[:, None]
        x[:, 30:42, 0] = a[:, None]*nuisance[:, None]
        x[:, 30:42, 1] = b[:, None]*nuisance[:, None]
        if version in ('challenge_v2','challenge_v3'):
            x[:, 30:42, :2] *= .5
        y = labels
    else:
        raise ValueError(task)
    if version == 'challenge_v3':
        # Last, predeclared ceiling check: label-independent sensory corruption.
        # It is part of the observed input, not extra target-aware simulator noise.
        if task=='pulse_order':
            x[:,20:40,:2] *= .4
            x[:,20:48,:2] += rng.normal(0.,.20,(trials,28,2))
        elif task=='interval':
            x[:,20:48,0] += rng.normal(0.,.15,(trials,28))
        elif task=='context_recall':
            x[:,30:48,:2] += rng.normal(0.,.20,(trials,18,2))
            x[:,16:24,2] += rng.normal(0.,.20,(trials,8))
    if version not in ('v1','challenge_v2','challenge_v3'):
        raise ValueError('Unknown task version')
    assert not np.any(x[:, 52:])
    return x*float(amplitude), np.asarray(y, np.float32)

@torch.no_grad()
def simulate_driven(cfg, A, masks, coupling, noise_sigma, noise_seed,
                    inputs=None, projection=None, filter_tau_ms=30., check=None):
    """Batch [mask,trial,node]; common trial-specific noise across masks.

    Returns [mask,trial,node] final filtered spikes and aggregate diagnostics.
    No full trajectories. `inputs=None` uses frozen common-noise semantics with
    exactly one trial and must agree with frozen simulate_masks at same B.
    All arithmetic is float32; no AMP, changed dt, or compiled approximation.
    """
    device=A.device; dtype=torch.float32
    masks=masks.to(device=device,dtype=torch.bool)
    M,N=masks.shape
    T=1 if inputs is None else int(inputs.shape[0])
    masks=masks[:,None,:].expand(M,T,N).reshape(M*T,N)
    B=M*T
    dt=float(cfg['dt_ms']); steps=round(float(cfg['duration_ms'])/dt)
    warm=round(float(cfg['warmup_ms'])/dt)
    bin_steps=max(1,round(float(cfg['record_bin_ms'])/dt)); bins=(steps-warm)//bin_steps
    lif=cfg['lif']; hhcfg=cfg['hh']; syn=cfg['synapse']
    v=torch.full((B,N),-65.,device=device,dtype=dtype)
    ref=torch.zeros_like(v); s=torch.zeros_like(v)
    am,bm,ah,bh,an,bn=frozen.hh_rates(v)
    m=am/(am+bm); h=ah/(ah+bh); ng=an/(an+bn)
    pop=torch.zeros((B,bins),device=device,dtype=dtype)
    counts=torch.zeros_like(v); acc=torch.zeros(B,device=device,dtype=dtype)
    filtered=torch.zeros_like(v)
    gen=torch.Generator(device=device); gen.manual_seed(int(noise_seed))
    chunk_steps=max(1,int(cfg.get('noise_chunk_steps',256)))
    noise_chunk=None; noise_pos=chunk_steps; current_len=0; bidx=0
    AT=A.T.contiguous(); drive=None
    if inputs is not None:
        inputs=torch.as_tensor(inputs,device=device,dtype=dtype)
        projection=torch.as_tensor(projection,device=device,dtype=dtype)
        assert inputs.shape[1] == bins and projection.shape == (inputs.shape[2],N)
        drive=inputs@projection
    for t in range(steps):
        if check is not None and t % 1000 == 0:
            check()
        syn_drive=s@AT
        I_syn=float(coupling)*syn_drive*100.
        if drive is not None and t>=warm:
            ext=drive[:,(t-warm)//bin_steps,:]
            I_syn=I_syn+ext[None,:,:].expand(M,T,N).reshape(B,N)
        if noise_chunk is None or noise_pos>=current_len:
            current_len=min(chunk_steps,steps-t)
            noise_chunk=torch.randn((current_len,T,N),device=device,dtype=dtype,generator=gen)
            noise_pos=0
        noise=noise_chunk[noise_pos]*float(noise_sigma)
        noise=noise[None,:,:].expand(M,T,N).reshape(B,N)
        noise_pos+=1
        lifmask=~masks
        dv_lif=((float(lif['v_rest'])-v)+float(lif['bias'])*20.+I_syn+noise)/float(lif['tau_m_ms'])
        can=(ref<=0.)&lifmask
        v=torch.where(can,v+dt*dv_lif,v)
        ref=torch.clamp(ref-dt,min=0.)
        lif_spk=lifmask&(v>=float(lif['v_th']))&(ref<=0.)
        v=torch.where(lif_spk,torch.full_like(v,float(lif['v_reset'])),v)
        ref=torch.where(lif_spk,torch.full_like(ref,float(lif['refractory_ms'])),ref)
        am,bm,ah,bh,an,bn=frozen.hh_rates(v)
        m2=m+dt*(am*(1-m)-bm*m)
        h2=h+dt*(ah*(1-h)-bh*h)
        n2=ng+dt*(an*(1-ng)-bn*ng)
        m=torch.where(masks,m2,m); h=torch.where(masks,h2,h); ng=torch.where(masks,n2,ng)
        INa=120.*(m**3)*h*(v-50.)
        IK=36.*(ng**4)*(v-(-77.))
        IL=.3*(v-(-54.387))
        vnew=v+dt*(float(hhcfg['bias_uA_cm2'])+I_syn+noise-INa-IK-IL)
        hh_spk=masks&(v<0.)&(vnew>=0.)
        v=torch.where(masks,vnew,v)
        spk=lif_spk|hh_spk
        s+=dt*(-s/float(syn['tau_ms'])); s+=spk.float()
        filtered+=dt*(-filtered/float(filter_tau_ms)); filtered+=spk.float()
        if t>=warm:
            sf=spk.float(); counts+=sf; acc+=sf.sum(1)
            if (t-warm+1)%bin_steps==0 and bidx<bins:
                sec=(bin_steps*dt)/1000.
                pop[:,bidx]=acc/N/sec
                acc.zero_(); bidx+=1
    if not torch.isfinite(v).all() or not torch.isfinite(filtered).all():
        raise FloatingPointError('Non-finite reservoir state; record failed attempt, do not drop trial')
    metrics=frozen.gpu_metrics(pop[:,:bidx],counts,(float(cfg['duration_ms'])-float(cfg['warmup_ms']))/1000.,float(cfg['record_bin_ms']))
    return filtered.reshape(M,T,N).cpu().numpy(), {k:x.reshape(M,T).cpu().numpy() for k,x in metrics.items()}

def score(y, pred, task):
    if task == 'delayed_recall':
        return float(1.-np.mean((y-pred)**2)/max(np.var(y),1e-12))
    return float(np.mean((pred>=0.)==(y>=0.)))

def fit_readout(x, y, alpha=10.):
    """Training-only z-scoring, fixed ridge regularization, unpenalized intercept."""
    from scipy.linalg import solve
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
    mean=x.mean(0); scale=x.std(0); scale=np.where(scale>1e-8,scale,1.)
    z=(x-mean)/scale; ym=y.mean(); yc=y-ym
    if len(x)<x.shape[1]:
        beta=z.T@solve(z@z.T+alpha*np.eye(len(x)),yc,assume_a='pos')
    else:
        beta=solve(z.T@z+alpha*np.eye(x.shape[1]),z.T@yc,assume_a='pos')
    return dict(mean=mean,scale=scale,beta=beta,intercept=ym)

def predict(x, model):
    return ((x-model['mean'])/model['scale'])@model['beta']+model['intercept']
