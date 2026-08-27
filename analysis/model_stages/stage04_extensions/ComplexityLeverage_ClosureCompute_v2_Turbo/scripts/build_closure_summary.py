from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import write_json

def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8')) if Path(p).exists() else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',default='/data/coding/NeuralScience');ap.add_argument('--output-root',default=None);args=ap.parse_args()
    root=Path(args.project_root).resolve();base=Path(args.output_root) if args.output_root else root/'results_complexity_leverage_closure_v1'
    ct=readj(base/'stage4_cross_state_transfer'/'analysis'/'cross_state_key_metrics.json');dg=readj(base/'stage4_functional_degeneracy'/'analysis'/'degeneracy_key_metrics.json');al=readj(base/'allen_lag_aware_refit'/'analysis'/'lagaware_key_metrics.json')
    out={'stage4_cross_state_transfer':ct,'stage4_functional_degeneracy':dg,'allen_lag_aware_refit':al}
    write_json(base/'closure_summary.json',out)
    md=['# Complexity–leverage closure compute summary','']
    if ct:
        md += ['## Stage 4 cross-state transfer',f"- Pure-state crossover interaction: {ct.get('mean_crossover_interaction')}",f"- 95% bootstrap CI: [{ct.get('bootstrap95_low')}, {ct.get('bootstrap95_high')}]",f"- exact anchor sign-flip p: {ct.get('exact_anchor_signflip_p_greater')}",'']
    if dg:
        md += ['## Functional degeneracy',f"- mean D_0.02: {dg.get('mean_D_epsilon')}",f"- mean sampled near-optimal count: {dg.get('mean_near_optimal_count')}",f"- k8 > k64 sign-flip p: {dg.get('k8_gt_k64_exact_signflip_p')}",'']
    if al:
        md += ['## Allen lag-aware refit',f"- median held-out R2: {al.get('median_test_r2')}",f"- positive R2 fraction: {al.get('positive_r2_fraction')}",f"- median delta vs persistence: {al.get('median_delta_vs_persistence')}",f"- raw / residual state-pair rho: {al.get('median_raw_state_pair_rho')} / {al.get('median_residual_state_pair_rho')}",'']
    (base/'closure_summary.md').write_text('\n'.join(md),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
