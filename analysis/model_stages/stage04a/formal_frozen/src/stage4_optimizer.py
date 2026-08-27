from __future__ import annotations
import torch
from src.stage4_gpu_simulator import simulate_masks

def random_fixed_k(pop,n,k,device,generator):
    if k<=0:
        return torch.zeros((pop,n),dtype=torch.bool,device=device)
    if k>=n:
        return torch.ones((pop,n),dtype=torch.bool,device=device)
    r=torch.rand((pop,n),device=device,generator=generator)
    idx=torch.topk(r,k=k,dim=1).indices
    m=torch.zeros((pop,n),dtype=torch.bool,device=device)
    m.scatter_(1,idx,True)
    return m

def mutate_swap(parents,n_swaps,generator):
    child=parents.clone()
    B,N=child.shape
    rows=torch.arange(B,device=child.device)
    for _ in range(int(n_swaps)):
        rr=torch.rand((B,N),device=child.device,generator=generator)
        rem=rr.masked_fill(~child,-1.0).argmax(1)
        aa=torch.rand((B,N),device=child.device,generator=generator)
        add=aa.masked_fill(child,-1.0).argmax(1)
        child[rows,rem]=False
        child[rows,add]=True
    return child

def evolve(cfg,A,k,coupling,noise_sigma,search_seed,initial_masks=None,resume=None,checkpoint_cb=None):
    device=A.device; n=A.shape[0]
    P=int(cfg["optimizer"]["population"])
    E=int(cfg["optimizer"]["elite"])
    G=int(cfg["optimizer"]["generations"])
    inject=int(cfg["optimizer"]["random_injection"])
    swaps=int(cfg["optimizer"]["swap_mutations"])

    gen=torch.Generator(device=device); gen.manual_seed(int(search_seed))

    if resume is not None:
        pop=resume["population"].to(device=device,dtype=torch.bool)
        start_gen=int(resume["generation"])+1
        history=list(resume.get("history",[]))
        best_mask=resume["best_mask"].to(device=device,dtype=torch.bool)
        best_score=float(resume["best_score"])
    else:
        pop=random_fixed_k(P,n,k,device,gen)
        if initial_masks:
            for i,m in enumerate(initial_masks[:min(len(initial_masks),P)]):
                pop[i]=torch.as_tensor(m,dtype=torch.bool,device=device)
        start_gen=0; history=[]; best_mask=pop[0].clone(); best_score=-1e30

    last_metrics=None
    for gidx in range(start_gen,G):
        # Same stochastic drive for every placement in this generation.
        # Noise seed changes across generations to reduce overfitting to one trace.
        noise_seed=int(search_seed)*1009+gidx*7919+17
        metrics=simulate_masks(cfg,A,pop,coupling,noise_sigma,noise_seed,common_noise=True)
        scores=metrics["rhythm_score"]
        vals,idx=torch.topk(scores,k=min(E,P))
        elites=pop[idx].clone()
        if float(vals[0])>best_score:
            best_score=float(vals[0].item()); best_mask=elites[0].clone()

        history.append({
            "generation":gidx,
            "best":float(vals[0].item()),
            "mean":float(scores.mean().item()),
            "p90":float(torch.quantile(scores,0.9).item()),
        })

        keep=E
        offspring=P-keep-inject
        parent_idx=torch.randint(0,E,(offspring,),device=device,generator=gen)
        kids=mutate_swap(elites[parent_idx],swaps,gen) if offspring>0 else elites[:0]
        rnd=random_fixed_k(inject,n,k,device,gen) if inject>0 else elites[:0]
        pop=torch.cat([elites,kids,rnd],dim=0)
        last_metrics=metrics

        if checkpoint_cb is not None:
            checkpoint_cb({
                "generation":gidx,
                "population":pop.detach().cpu(),
                "best_mask":best_mask.detach().cpu(),
                "best_score":best_score,
                "history":history,
            })

    return best_mask,best_score,history,pop
