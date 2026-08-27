from __future__ import annotations
import numpy as np
import torch

def _orient_undirected(u, rng):
    n = u.shape[0]
    a = np.zeros((n,n), dtype=np.float32)
    ii,jj = np.where(np.triu(u,1)>0)
    if len(ii):
        flip = rng.random(len(ii)) < 0.5
        a[ii[flip],jj[flip]] = 1.0
        a[jj[~flip],ii[~flip]] = 1.0
    return a

def make_er(n,p,rng):
    a=(rng.random((n,n))<p).astype(np.float32)
    np.fill_diagonal(a,0)
    return a, np.zeros(n,dtype=np.int32)

def make_small_world(n,p,rng):
    # Correct Stage-3 confound:
    # orienting an undirected graph halves mean in-degree.
    # Build undirected mean degree ~2*p*(n-1), so directed density after orientation ~p.
    k=max(2,int(round(2.0*p*(n-1))))
    if k%2: k+=1
    k=min(k, n-1 if (n-1)%2==0 else n-2)
    half=max(1,k//2)
    u=np.zeros((n,n),dtype=np.uint8)
    for i in range(n):
        for d in range(1,half+1):
            j=(i+d)%n
            u[i,j]=u[j,i]=1
    beta=0.15
    for i in range(n):
        for d in range(1,half+1):
            j=(i+d)%n
            if rng.random()<beta and u[i,j]:
                u[i,j]=u[j,i]=0
                forbidden=set(np.flatnonzero(u[i]).tolist()); forbidden.add(i)
                cand=np.asarray([x for x in range(n) if x not in forbidden], dtype=np.int64)
                if len(cand):
                    q=int(rng.choice(cand)); u[i,q]=u[q,i]=1
                else:
                    u[i,j]=u[j,i]=1
    return _orient_undirected(u,rng), np.zeros(n,dtype=np.int32)

def make_scale_free(n,p,rng):
    # BA-like undirected graph has mean degree ~2m.
    # After one-way orientation mean in-degree ~m, so choose m~p*(n-1).
    m=max(1,int(round(p*(n-1))))
    m=min(m,max(1,n//5))
    seed_n=min(n,max(m+1,3))
    u=np.zeros((n,n),dtype=np.uint8)
    for i in range(seed_n):
        for j in range(i+1,seed_n):
            u[i,j]=u[j,i]=1
    degree=u.sum(0).astype(np.float64)
    for new in range(seed_n,n):
        probs=degree[:new]+1.0; probs/=probs.sum()
        chosen=rng.choice(new,size=min(m,new),replace=False,p=probs)
        for old in chosen:
            u[new,old]=u[old,new]=1
        degree=u.sum(0).astype(np.float64)
    return _orient_undirected(u,rng), np.zeros(n,dtype=np.int32)

def make_modular(n,p,rng):
    nmod=4
    modules=np.arange(n,dtype=np.int32)%nmod
    rng.shuffle(modules)
    # directed density matched in expectation
    p_out=max(0.001,0.20*p)
    same_fraction=(n/nmod-1)/(n-1)
    p_in=(p-(1-same_fraction)*p_out)/max(same_fraction,1e-9)
    p_in=float(np.clip(p_in,p,min(0.95,max(p,6*p))))
    same=modules[:,None]==modules[None,:]
    probs=np.where(same,p_in,p_out)
    a=(rng.random((n,n))<probs).astype(np.float32)
    np.fill_diagonal(a,0)
    return a,modules

def build_topology(kind,n,p,seed):
    rng=np.random.default_rng(int(seed))
    if kind=="er": return make_er(n,p,rng)
    if kind=="small_world": return make_small_world(n,p,rng)
    if kind=="scale_free": return make_scale_free(n,p,rng)
    if kind=="modular": return make_modular(n,p,rng)
    raise ValueError(kind)

def node_features(a,modules):
    a64=a.astype(np.float64,copy=False)
    indeg=a64.sum(1); outdeg=a64.sum(0); total=indeg+outdeg
    a2=a64@a64
    cycle3=np.sum(a2*a64.T,axis=1)
    bridge=np.zeros(len(a),dtype=np.float64)
    if np.unique(modules).size>1:
        for i in range(len(a)):
            other=modules!=modules[i]
            bridge[i]=a64[i,other].sum()+a64[other,i].sum()
    sym=a64+a64.T
    v=np.ones(len(a),dtype=np.float64)/max(len(a),1)
    for _ in range(40):
        nv=sym@v; norm=np.linalg.norm(nv)
        if norm<1e-12: break
        v=nv/norm
    spectral=np.abs(v)
    return {
        "in_degree":indeg,
        "out_degree":outdeg,
        "total_degree":total,
        "feedback_score":(indeg+1)*(outdeg+1),
        "cycle3":cycle3,
        "module_bridge":bridge,
        "spectral_centrality":spectral,
    }

def graph_descriptors(a):
    indeg=a.sum(1); outdeg=a.sum(0); deg=indeg+outdeg
    density=float(a.sum()/max(len(a)*(len(a)-1),1))
    return {
        "graph_density":density,
        "degree_cv":float(np.std(deg)/(np.mean(deg)+1e-9)),
        "max_degree_ratio":float(np.max(deg)/(np.mean(deg)+1e-9)),
        "inout_corr":float(np.corrcoef(indeg,outdeg)[0,1]) if np.std(indeg)>0 and np.std(outdeg)>0 else 0.0,
    }

def normalize_for_dynamics(a):
    mean_in=max(float(a.sum(1).mean()),1.0)
    return (a/mean_in).astype(np.float32), mean_in

def heuristic_mask(a,modules,k,strategy,seed=0):
    n=len(a); k=int(k)
    mask=np.zeros(n,dtype=bool)
    if k<=0: return mask
    if k>=n: mask[:]=True; return mask
    rng=np.random.default_rng(int(seed)+173)
    f=node_features(a,modules)
    if strategy=="random":
        idx=rng.choice(n,size=k,replace=False)
    else:
        if strategy=="high_degree": score=f["total_degree"]
        elif strategy=="feedback_hub": score=f["feedback_score"]
        elif strategy=="module_bridge": score=f["module_bridge"]+0.03*f["total_degree"]
        elif strategy=="cycle_proxy": score=f["cycle3"]+0.01*f["total_degree"]
        elif strategy=="spectral": score=f["spectral_centrality"]
        else: raise ValueError(strategy)
        score=np.asarray(score)+1e-12*rng.random(n)
        idx=np.argsort(score)[::-1][:k]
    mask[idx]=True
    return mask

def set_descriptors(a,modules,mask):
    f=node_features(a,modules)
    idx=np.flatnonzero(mask)
    n=len(a)
    if len(idx)==0:
        return {k:0.0 for k in [
            "sel_total_degree","sel_feedback","sel_cycle3","sel_bridge","sel_spectral",
            "coverage1","coverage2","out_neighbor_redundancy","selected_dispersion"
        ]}
    vals={
        "sel_total_degree":float(np.mean(f["total_degree"][idx])),
        "sel_feedback":float(np.mean(f["feedback_score"][idx])),
        "sel_cycle3":float(np.mean(f["cycle3"][idx])),
        "sel_bridge":float(np.mean(f["module_bridge"][idx])),
        "sel_spectral":float(np.mean(f["spectral_centrality"][idx])),
    }
    out_neighbors=(a[:,idx].sum(1)>0) # columns are sources in our convention
    coverage1=float(np.mean(out_neighbors | mask))
    reach=(a@mask.astype(np.float32)>0)
    reach2=(a@reach.astype(np.float32)>0)
    vals["coverage1"]=coverage1
    vals["coverage2"]=float(np.mean(mask | reach | reach2))
    neigh=[]
    for i in idx:
        neigh.append(set(np.flatnonzero(a[:,i]>0).tolist()))
    js=[]
    for ii in range(len(neigh)):
        for jj in range(ii+1,len(neigh)):
            u=neigh[ii]|neigh[jj]
            js.append(len(neigh[ii]&neigh[jj])/max(len(u),1))
    vals["out_neighbor_redundancy"]=float(np.mean(js)) if js else 0.0
    # undirected shortest-path dispersion among selected nodes
    und=(a+a.T)>0
    dists=[]
    for s in idx:
        dist=np.full(n,-1,dtype=np.int32); dist[s]=0; q=[int(s)]
        for v in q:
            for w in np.flatnonzero(und[v]):
                if dist[w]<0:
                    dist[w]=dist[v]+1; q.append(int(w))
        for t in idx:
            if t>s and dist[t]>=0: dists.append(int(dist[t]))
    vals["selected_dispersion"]=float(np.mean(dists)) if dists else 0.0
    return vals

def to_gpu_adjacency(a,device):
    an,_=normalize_for_dynamics(a)
    return torch.as_tensor(an,dtype=torch.float32,device=device)
