import json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from frozen_stage4.src.stage4_topology import build_topology,to_gpu_adjacency
from frozen_stage4.src.stage4_gpu_simulator import simulate_masks
from frozen_stage4.src.stage4_gpu_simulator_turbo import simulate_masks_seed_batch

def main():
    cfg=json.loads((ROOT/'frozen_stage4'/'stage4a.json').read_text())
    cfg=dict(cfg);cfg['duration_ms']=30.0;cfg['warmup_ms']=5.0;cfg['record_bin_ms']=1.0;cfg['noise_chunk_steps']=64
    N=24;A_np,_=build_topology('scale_free',N,.05,123);A=to_gpu_adjacency(A_np,torch.device('cpu'))
    masks=np.zeros((3,N),bool);masks[0,:4]=1;masks[1,4:8]=1;masks[2,[1,7,13,19]]=1;T=torch.as_tensor(masks)
    seeds=[1001,1002]
    old=[]
    for s in seeds: old.append(simulate_masks(cfg,A,T,.035,1.0,s,common_noise=True))
    new=simulate_masks_seed_batch(cfg,A,T,.035,1.0,seeds,common_noise=True)
    worst=0.0
    for key in new:
        ref=torch.stack([x[key] for x in old],0); d=float(torch.max(torch.abs(ref-new[key])).item());worst=max(worst,d)
        if not torch.allclose(ref,new[key],rtol=2e-5,atol=2e-6): raise AssertionError(f'{key} mismatch max_abs={d}')
    print('STAGE4_TURBO_PARITY_OK max_abs_diff=',worst)
if __name__=='__main__': main()
