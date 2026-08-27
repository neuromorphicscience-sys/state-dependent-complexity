from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import read_yaml,find_nwb,ensure_dir,write_json
CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
out=ensure_dir(Path(CFG["output_root"])/"audit")
rep={}
for key in ("steinmetz_root","allen_root"):
    p=Path(CFG[key])
    rep[key]={"path":str(p),"exists":p.exists(),
              "nwb_count":len(find_nwb(p)) if p.exists() else 0}
rep["python"]=sys.version
write_json(out/"environment_check.json",rep)
print(json.dumps(rep,indent=2))
if rep["steinmetz_root"]["nwb_count"]<1 or rep["allen_root"]["nwb_count"]<1:
    raise SystemExit("DATA ROOT CHECK FAILED")
print("ENVIRONMENT/DATA ROOT CHECK PASS")
