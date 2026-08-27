from pathlib import Path
import sys,json,traceback
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import *
from stage5b.nwb_utils import *

CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
ROOT=Path(CFG["allen_root"]);OUT=ensure_dir(Path(CFG["output_root"])/"allen_vbo")
inv_path=ROOT/"download_status"/"cohort_inventory.json"
cohort={}
if inv_path.exists():
    j=json.loads(inv_path.read_text(encoding="utf-8"))
    cohort={int(x["experiment_id"]):x for x in j.get("items",[])}
rows=[];schemas={}
for p in find_nwb(ROOT/"raw"):
    try:eid=int(p.stem.split("_")[-1])
    except:eid=-1
    rec={"file":p.name,"path":str(p),"bytes":p.stat().st_size,"experiment_id":eid,"status":"ok"}
    if eid in cohort: rec.update({k:cohort[eid].get(k) for k in ("container_id","cre_line","state")})
    try:
        with open_nwb(p) as nwb:
            s=nwb_schema_summary(nwb);schemas[p.name]=s
            cands=find_roi_response_series(nwb)
            rec["roi_series_count"]=len(cands)
            rec["preferred_roi_series"]=cands[0][1] if cands else ""
            if cands:
                X,ts,cids,tab=roi_series_matrix(cands[0][2])
                rec["timepoints"]=X.shape[0];rec["cells"]=X.shape[1]
                rec["matched_id_nonmissing"]=int(pd.Series(cids).notna().sum())
    except Exception as e:
        rec["status"]="failed";rec["error"]=repr(e);schemas[p.name]={"error":traceback.format_exc()}
    rows.append(rec)
pd.DataFrame(rows).to_csv(OUT/"inventory.csv",index=False)
write_json(OUT/"schema_by_file.json",schemas)
print("Allen inventory complete:",len(rows))
