from __future__ import annotations
import math
import torch

# IMPORTANT: equations are intentionally copied from the frozen v1 simulator.
# The only change is execution layout: independent noise seeds are batched on GPU.

def _vtrap(x,y):
    z=x/y
    return torch.where(torch.abs(z)<1e-6, y*(1.0-z/2.0), x/torch.expm1(z))

def hh_rates(v):
    am=0.1*_vtrap(-(v+40.0),10.0)
    bm=4.0*torch.exp(-(v+65.0)/18.0)
    ah=0.07*torch.exp(-(v+65.0)/20.0)
    bh=1.0/(torch.exp(-(v+35.0)/10.0)+1.0)
    an=0.01*_vtrap(-(v+55.0),10.0)
    bn=0.125*torch.exp(-(v+65.0)/80.0)
    return am,bm,ah,bh,an,bn

def gpu_metrics(pop_rate,spike_counts,duration_s,bin_ms):
    x=pop_rate
    mean=x.mean(1); std=x.std(1,unbiased=False); sync=std/(mean+1e-8)
    xc=x-x.mean(1,keepdim=True)
    spec=(torch.fft.rfft(xc,dim=1).abs()**2)
    freq=torch.fft.rfftfreq(x.shape[1],d=float(bin_ms)/1000.0).to(x.device)
    valid=(freq>=0.5)&(freq<=100.0); pv=spec[:,valid]; fv=freq[valid]
    total=pv.sum(1)+1e-12; k=pv.argmax(1); dom=fv[k]
    conc=pv.gather(1,k[:,None]).squeeze(1)/total
    pn=pv/total[:,None]
    entropy=-(pn*torch.log(pn+1e-12)).sum(1)/math.log(max(pv.shape[1],2))
    rates=spike_counts/max(duration_s,1e-9); silent=(rates<0.1).float().mean(1)
    rhythm=torch.tanh(mean/2.0)*conc*(1.0-torch.clamp(entropy,max=1.0))*torch.tanh(sync)*(1.0-silent)
    return {"rhythm_score":rhythm,"mean_rate_hz":mean,"dominant_frequency_hz":dom,
            "spectral_concentration":conc,"spectral_entropy":entropy,
            "synchrony_proxy":sync,"silent_fraction":silent}

@torch.no_grad()
def simulate_masks_seed_batch(cfg,A,masks,coupling,noise_sigma,noise_seeds,common_noise=True):
    """
    Exact-science turbo layout for the frozen Stage4 simulator.

    A: [N,N], masks: [B,N], noise_seeds: iterable length S.
    Returns each metric with shape [S,B]. Each seed owns an independent
    torch.Generator initialized exactly as the legacy scalar-seed simulator.
    For common_noise=True, the same noise trace is shared by all B masks for
    a given seed, preserving the original common-noise design.
    """
    device=A.device; dtype=torch.float32
    masks=masks.to(device=device,dtype=torch.bool)
    seeds=[int(x) for x in noise_seeds]
    if not seeds: raise ValueError("noise_seeds must be non-empty")
    S=len(seeds); B,N=masks.shape; Q=S*B
    masks_q=masks.unsqueeze(0).expand(S,B,N).reshape(Q,N)

    dt=float(cfg["dt_ms"]); steps=int(round(float(cfg["duration_ms"])/dt))
    warm=int(round(float(cfg["warmup_ms"])/dt)); bin_steps=max(1,int(round(float(cfg["record_bin_ms"])/dt)))
    bins=max(1,(steps-warm)//bin_steps)
    lif=cfg["lif"]; hhcfg=cfg["hh"]; syn=cfg["synapse"]

    v=torch.full((Q,N),-65.0,device=device,dtype=dtype)
    ref=torch.zeros((Q,N),device=device,dtype=dtype); s=torch.zeros((Q,N),device=device,dtype=dtype)
    am,bm,ah,bh,an,bn=hh_rates(v); m=am/(am+bm); h=ah/(ah+bh); ng=an/(an+bn)
    pop=torch.zeros((Q,bins),device=device,dtype=dtype); counts=torch.zeros((Q,N),device=device,dtype=dtype)
    acc=torch.zeros(Q,device=device,dtype=dtype); bidx=0

    # Separate generators preserve each legacy seed stream exactly.
    gens=[]
    for seed in seeds:
        g=torch.Generator(device=device); g.manual_seed(seed); gens.append(g)
    chunk_steps=max(1,int(cfg.get("noise_chunk_steps",256)))
    noise_chunk=None; noise_pos=chunk_steps; current_len=0

    gNa,ENa=120.0,50.0; gK,EK=36.0,-77.0; gL,EL=0.3,-54.387
    AT=A.T.contiguous()
    for t in range(steps):
        syn_drive=s@AT; I_syn=float(coupling)*syn_drive*100.0
        if noise_chunk is None or noise_pos>=current_len:
            current_len=min(chunk_steps,steps-t)
            if common_noise:
                # [T,S,1,N]; each call shape matches legacy [T,1,N].
                chunks=[torch.randn((current_len,1,N),device=device,dtype=dtype,generator=g) for g in gens]
                noise_chunk=torch.stack(chunks,dim=1)
            else:
                # Rare fallback: independent noise by seed and mask.
                chunks=[torch.randn((current_len,B,N),device=device,dtype=dtype,generator=g) for g in gens]
                noise_chunk=torch.stack(chunks,dim=1)
            noise_pos=0
        z=noise_chunk[noise_pos]
        if common_noise: z=z.expand(S,B,N)
        noise=z.reshape(Q,N)*float(noise_sigma); noise_pos+=1

        lifmask=~masks_q
        dv_lif=((float(lif["v_rest"])-v)+float(lif["bias"])*20.0+I_syn+noise)/float(lif["tau_m_ms"])
        can=(ref<=0.0)&lifmask; v=torch.where(can,v+dt*dv_lif,v)
        ref=torch.clamp(ref-dt,min=0.0)
        lif_spk=lifmask&(v>=float(lif["v_th"]))&(ref<=0.0)
        v=torch.where(lif_spk,torch.full_like(v,float(lif["v_reset"])),v)
        ref=torch.where(lif_spk,torch.full_like(ref,float(lif["refractory_ms"])),ref)

        am,bm,ah,bh,an,bn=hh_rates(v)
        m2=m+dt*(am*(1-m)-bm*m); h2=h+dt*(ah*(1-h)-bh*h); n2=ng+dt*(an*(1-ng)-bn*ng)
        m=torch.where(masks_q,m2,m); h=torch.where(masks_q,h2,h); ng=torch.where(masks_q,n2,ng)
        INa=gNa*(m**3)*h*(v-ENa); IK=gK*(ng**4)*(v-EK); IL=gL*(v-EL)
        vnew=v+dt*(float(hhcfg["bias_uA_cm2"])+I_syn+noise-INa-IK-IL)
        hh_spk=masks_q&(v<0.0)&(vnew>=0.0); v=torch.where(masks_q,vnew,v)
        spk=lif_spk|hh_spk; s += dt*(-s/float(syn["tau_ms"])); s += spk.float()
        if t>=warm:
            sf=spk.float(); counts+=sf; acc+=sf.sum(1); rel=t-warm
            if (rel+1)%bin_steps==0 and bidx<bins:
                sec=(bin_steps*dt)/1000.0; pop[:,bidx]=acc/N/sec; acc.zero_(); bidx+=1

    met=gpu_metrics(pop[:,:bidx],counts,(float(cfg["duration_ms"])-float(cfg["warmup_ms"]))/1000.0,float(cfg["record_bin_ms"]))
    return {k:v.reshape(S,B) for k,v in met.items()}
