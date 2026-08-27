from pathlib import Path
import sys,json,traceback
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import read_yaml,find_nwb,ensure_dir,write_json,normalise_trial_table
from stage5b.nwb_utils import open_nwb,nwb_schema_summary,table_df

CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
root=Path(CFG["steinmetz_root"]); out=ensure_dir(Path(CFG["output_root"])/"steinmetz")
rows=[]; schemas={}
for i,p in enumerate(find_nwb(root),1):
    print(f"[{i}] {p.name}",flush=True)
    rec={"path":str(p),"file":p.name,"bytes":p.stat().st_size,"status":"ok"}
    try:
        with open_nwb(p) as nwb:
            s=nwb_schema_summary(nwb); schemas[p.name]=s
            rec.update({"identifier":s["identifier"],"units":s["units_rows"],
                        "trials":s["trials_rows"],
                        "processing":";".join(s["processing"]),
                        "intervals":";".join(s["intervals"])})
            t=normalise_trial_table(table_df(getattr(nwb,"trials",None)))
            rec["normalised_trials"]=len(t)
            rec["trial_columns"]=";".join(map(str,t.columns))
    except Exception as e:
        rec["status"]="failed";rec["error"]=repr(e);schemas[p.name]={"error":traceback.format_exc()}
    rows.append(rec)
pd.DataFrame(rows).to_csv(out/"inventory.csv",index=False)
write_json(out/"schema_by_file.json",schemas)
print("Steinmetz inventory complete:",len(rows))
