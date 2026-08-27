#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from common import cfg,write_json,now,read_csv
def count_rows(p):
    try:return sum(1 for _ in open(p,encoding="utf-8-sig"))-1
    except:return -1
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    c=cfg(a.config); out=Path(c["output_root"])/"00_preflight";out.mkdir(parents=True,exist_ok=True)
    ar=Path(c["allen_root"]); vr=Path(c["v11_result_root"]); pr=Path(c["digital_pfc_root"]); wr=Path(c["digital_whole_root"])
    ephys=ar/"analysis/ephys_axis_sweep_inventory_v1/ephys_features/ephys_features_specimen_level_clean.csv"
    tax=ar/"analysis/ephys_axis_sweep_inventory_v1/ephys_features/feature_taxonomy.csv"
    glif=ar/"analysis/glif_performance_atlas_v1/tables/glif_performance_atlas_specimen_level.csv"
    ab=vr/"30_temporal_context/AB/context_performance.csv"; ba=vr/"30_temporal_context/BA/context_performance.csv"
    subtype=pr/"pfc_2022/metadata/SubtypeInfo.txt"
    rep={
      "created_utc":now(),
      "ephys_table":str(ephys),"ephys_exists":ephys.exists(),
      "taxonomy_exists":tax.exists(),"glif_exists":glif.exists(),
      "AB_rows":count_rows(ab),"BA_rows":count_rows(ba),
      "whole_swc":len(list((wr/"extracted_bsdc").rglob("*.swc"))) if (wr/"extracted_bsdc").exists() else 0,
      "pfc2022_raw_swc":len(list((pr/"pfc_2022/extracted_by_archive").rglob("*.swc"))) if (pr/"pfc_2022/extracted_by_archive").exists() else 0,
      "pfc2022_aligned_swc":len([p for p in (pr/"pfc_2022/extracted_by_archive").rglob("*.swc") if "swc_allen_space" in [x.lower() for x in p.parts]]) if (pr/"pfc_2022/extracted_by_archive").exists() else 0,
      "subtypeinfo_exists":subtype.exists(),
      "pfc2023_swc":len(list((pr/"pfc_2023/extracted").rglob("*.swc"))) if (pr/"pfc_2023/extracted").exists() else 0
    }
    write_json(out/"preflight_v12.json",rep);print(json.dumps(rep,indent=2))
    req=["ephys_exists","taxonomy_exists","glif_exists","subtypeinfo_exists"]
    if any(not rep[k] for k in req):raise SystemExit("PREFLIGHT FAILED: missing required Allen/PFC metadata.")
    if rep["AB_rows"]!=7278 or rep["BA_rows"]!=7278:raise SystemExit("PREFLIGHT FAILED: v1.1 temporal AB/BA must each contain 7278 rows.")
    if rep["whole_swc"]!=12264:raise SystemExit(f"PREFLIGHT FAILED: expected 12264 whole-cortex SWCs, got {rep['whole_swc']}.")
    if rep["pfc2022_aligned_swc"]!=6357:raise SystemExit(f"PREFLIGHT FAILED: expected exactly 6357 canonical swc_allen_space PFC2022 files, got {rep['pfc2022_aligned_swc']}.")
    print("PREFLIGHT V1.2 PASS")
if __name__=="__main__":main()
