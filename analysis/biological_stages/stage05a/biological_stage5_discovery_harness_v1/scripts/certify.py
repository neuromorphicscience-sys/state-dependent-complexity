#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from common import sha256_file, write_json, now
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);a=ap.parse_args()
    root=Path(a.root);rows=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and "certificate_manifest" not in p.name:
            try:rows.append({"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
            except:pass
    write_json(root/"certificate_manifest.json",{"created_utc":now(),"files":rows})
    print("certificate files:",len(rows))
if __name__=="__main__":main()
