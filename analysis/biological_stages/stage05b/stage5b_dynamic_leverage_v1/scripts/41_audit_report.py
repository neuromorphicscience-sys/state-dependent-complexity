from pathlib import Path
import sys,json,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import *
CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
O=Path(CFG["output_root"]);R=ensure_dir(O/"reports")
def load(p):
    try:return pd.read_csv(p)
    except:return pd.DataFrame()
s=load(O/"steinmetz"/"inventory.csv");a=load(O/"allen_vbo"/"inventory.csv")
lines=["# Stage 5B audit summary","",
       f"- Steinmetz NWB inventory: {len(s)} files; failures: {(s.get('status',pd.Series(dtype=str))!='ok').sum() if len(s) else 'NA'}.",
       f"- Allen VBO NWB inventory: {len(a)} files; failures: {(a.get('status',pd.Series(dtype=str))!='ok').sum() if len(a) else 'NA'}.",
       "",
       "Proceed to baseline only if both inventories can be opened and Allen has a usable RoiResponseSeries with cell specimen IDs.",
       "If a schema fails, inspect `schema_by_file.json`; do not guess a field name."]
(R/"AUDIT_SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))
