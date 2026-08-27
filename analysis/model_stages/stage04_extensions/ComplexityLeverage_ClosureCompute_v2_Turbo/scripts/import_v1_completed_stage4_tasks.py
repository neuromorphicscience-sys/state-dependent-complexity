from __future__ import annotations
import argparse,json,shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description='Import only scientifically complete v1 Stage4 transfer tasks into v2 Turbo output.')
    ap.add_argument('--v1-root',required=True);ap.add_argument('--v2-root',required=True)
    ap.add_argument('--noise-seed-start',type=int,default=7601);ap.add_argument('--noise-seed-count',type=int,default=24)
    a=ap.parse_args();src=Path(a.v1_root);dst=Path(a.v2_root);expected=list(range(a.noise_seed_start,a.noise_seed_start+a.noise_seed_count))
    imported=skipped=invalid=0
    if not src.exists(): print(json.dumps({'status':'NO_V1_ROOT','path':str(src)}));return
    for done in src.rglob('done.json'):
        td=done.parent; rel=td.relative_to(src); summ=td/'mask_summary.csv'; seedm=td/'seed_metrics.csv'
        try: d=json.loads(done.read_text(encoding='utf-8'))
        except Exception: invalid+=1;continue
        if d.get('status')!='COMPLETE' or d.get('noise_seeds')!=expected or not summ.exists() or not seedm.exists(): invalid+=1;continue
        out=dst/rel
        if (out/'done.json').exists() and (out/'mask_summary.csv').exists() and (out/'seed_metrics.csv').exists(): skipped+=1;continue
        out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(summ,out/'mask_summary.csv');shutil.copy2(seedm,out/'seed_metrics.csv')
        d['imported_from_v1']=str(td);d['engine']='legacy_v1_imported_science_equivalent'
        (out/'done.json').write_text(json.dumps(d,indent=2),encoding='utf-8');imported+=1
    print(json.dumps({'status':'COMPLETE','imported':imported,'already_present':skipped,'invalid_or_partial':invalid,'expected_noise_seeds':expected},indent=2))
if __name__=='__main__':main()
