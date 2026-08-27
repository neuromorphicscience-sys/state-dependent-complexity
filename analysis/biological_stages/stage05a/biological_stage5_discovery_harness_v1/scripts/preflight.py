#!/usr/bin/env python3
import argparse, importlib, json, os, platform, subprocess, sys
from pathlib import Path
from common import load_config, ensure_dir, write_json, now

REQ=["numpy","h5py","torch","scipy","matplotlib"]

def count(root, pattern):
    p=Path(root)
    return len(list(p.rglob(pattern))) if p.exists() else 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a=ap.parse_args()
    c=load_config(a.config)
    out=ensure_dir(Path(c["output_root"])/"00_preflight")

    imports={}
    for name in REQ:
        try:
            mod=importlib.import_module(name)
            imports[name]={"ok":True,"version":getattr(mod,"__version__","")}
        except Exception as e:
            imports[name]={"ok":False,"error":f"{type(e).__name__}: {e}"}

    gpu={}
    try:
        import torch
        gpu={
            "cuda_available":torch.cuda.is_available(),
            "torch_version":torch.__version__,
            "cuda_runtime":torch.version.cuda,
            "device_count":torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            gpu["device_name"]=torch.cuda.get_device_name(0)
            gpu["memory_gib"]=torch.cuda.get_device_properties(0).total_memory/1024**3
            cap=torch.cuda.get_device_capability(0)
            gpu["capability"]=f"{cap[0]}.{cap[1]}"
    except Exception as e:
        gpu={"error":str(e)}

    allen=Path(c["allen_root"])
    whole=Path(c["digital_whole_root"])
    pfc=Path(c["digital_pfc_root"])
    paths={
        "allen_root_exists":allen.exists(),
        "allen_nwb_count":count(allen/"ephys_nwb"/"raw","*.nwb"),
        "digital_whole_swc_count":count(whole/"extracted_bsdc","*.swc"),
        "pfc2022_swc_count":count(pfc/"pfc_2022"/"extracted_by_archive","*.swc"),
        "pfc2023_swc_count":count(pfc/"pfc_2023"/"extracted","*.swc"),
        "pfc2022_subtypeinfo_exists":(pfc/"pfc_2022"/"metadata"/"SubtypeInfo.txt").exists(),
    }
    report={
        "created_utc":now(),"python":sys.version,"platform":platform.platform(),
        "imports":imports,"gpu":gpu,"paths":paths
    }
    write_json(out/"preflight.json",report)
    print(json.dumps(report,indent=2))
    missing=[k for k,v in imports.items() if not v["ok"]]
    if missing:
        print("MISSING PYTHON PACKAGES:",",".join(missing),file=sys.stderr)
        raise SystemExit(10)
    if paths["allen_nwb_count"]<1200:
        print("WARNING: Allen NWB count <1200. Temporal lane will be incomplete.",file=sys.stderr)
    if paths["digital_whole_swc_count"]==0 or paths["pfc2022_swc_count"]==0:
        print("WARNING: one or more Digital Brain SWC roots are empty.",file=sys.stderr)

if __name__=="__main__": main()
