#!/usr/bin/env python3
import argparse
from pathlib import Path
from common import sha256_file,write_json,now
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);a=ap.parse_args();root=Path(a.root)
    files=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name!="certificate_manifest_v12.json":
            files.append({"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    write_json(root/"certificate_manifest_v12.json",{"created_utc":now(),"files":files})
    print("CERTIFIED",len(files),"files")
if __name__=="__main__":main()
