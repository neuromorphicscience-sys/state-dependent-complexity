from pathlib import Path
import sys,json
import pandas as pd, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import *
CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
O=Path(CFG["output_root"]);R=ensure_dir(O/"reports")

def readj(p):
    try:return json.loads(Path(p).read_text(encoding="utf-8"))
    except:return {}

def stat_csv(p):
    p=Path(p)
    if not p.exists():return {"exists":False}
    d=pd.read_csv(p)
    return {"exists":True,"rows":len(d),"columns":list(d.columns)}

stein=stat_csv(O/"steinmetz"/"leverage_long.csv")
steinF=stat_csv(O/"steinmetz"/"role_flexibility.csv")
allen=stat_csv(O/"allen_vbo"/"leverage_long.csv")
allenF=stat_csv(O/"allen_vbo"/"role_flexibility.csv")
ctrl=readj(O/"allen_vbo"/"control_statistics.json")
bridge=readj(O/"bridge"/"bridge_status.json")
rec=readj(O/"stage5a_recovery"/"recovery_status.json")

lines=[
"# Stage 5B baseline report","",
"## Completion","",
f"- Steinmetz leverage table: {stein.get('rows','missing')} rows.",
f"- Steinmetz flexibility table: {steinF.get('rows','missing')} rows.",
f"- Allen leverage table: {allen.get('rows','missing')} rows.",
f"- Allen same-cell flexibility table: {allenF.get('rows','missing')} rows.",
f"- Stage 5A GLIF recovery: `{rec.get('status','not run')}`.",
f"- Cross-dataset bridge: `{bridge.get('status','not run')}`.","",
"## Allen same-cell control","",
f"- matched cells ≥2 states: {ctrl.get('matched_cells_2plus_states','NA')}",
f"- matched cells ≥3 states: {ctrl.get('matched_cells_3plus_states','NA')}",
f"- flexibility vs mean activity: {ctrl.get('flexibility_vs_mean_activity',{})}","",
"## Interpretation guardrail","",
"This baseline establishes whether state-resolved leverage and same-cell role flexibility are measurable. "
"It does not by itself prove that intrinsic cellular complexity causes flexibility. "
"The Stage 5A ↔ Stage 5B bridge remains class-conditioned and exploratory until class overlap, "
"uncertainty, and null/permutation controls are adequate.","",
"## Next decision","",
"1. If Steinmetz shows broad state re-ranking and Allen shows reproducible same-cell cross-state re-ranking, "
"advance to hierarchical mixed-effects/permutation inference and robustness analyses.",
"2. If flexibility is almost fully explained by mean activity, revise the leverage definition before paper-level claims.",
"3. If the class bridge has fewer than four well-supported overlapping classes, do not report a correlation; "
"use it only to generate the next biological prediction."
]
(R/"STAGE5B_BASELINE_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))
