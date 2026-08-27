from pathlib import Path
import sys,json,re
import numpy as np,pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import read_yaml,ensure_dir,write_json

CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
OUT=ensure_dir(Path(CFG["output_root"])/"stage5a_recovery")
perf=OUT/"glif_performance_atlas_specimen_level.csv"
if not perf.exists():
    raise SystemExit("No recovered GLIF performance atlas; epsilon sensitivity NOT RUN.")
df=pd.read_csv(perf)
cost={1:0,2:1,3:1,4:2,5:3}
epsilons=[0.01,0.02,0.05,0.10]
rows=[]
for _,r in df.iterrows():
    vals={i:pd.to_numeric(pd.Series([r.get(f"GLIF{i}_EVR")]),errors="coerce").iloc[0] for i in range(1,6)}
    valid={i:v for i,v in vals.items() if np.isfinite(v)}
    if not valid: continue
    best=max(valid.values())
    rec={"specimen_id":r["specimen_id"],"best_EVR":best}
    for eps in epsilons:
        eligible=[cost[i] for i,v in valid.items() if v>=best-eps]
        rec[f"C_req_eps_{eps:.2f}"]=min(eligible) if eligible else np.nan
    rows.append(rec)
out=pd.DataFrame(rows)
out.to_csv(OUT/"creq_epsilon_sensitivity.csv",index=False)
summary={}
for eps in epsilons:
    c=f"C_req_eps_{eps:.2f}"
    summary[c]=out[c].value_counts(dropna=False).sort_index().to_dict()
write_json(OUT/"epsilon_sensitivity_summary.json",summary)
print(out.head())
print(json.dumps(summary,indent=2,default=str))
