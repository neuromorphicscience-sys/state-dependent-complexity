#!/usr/bin/env python3
import argparse,csv,json,re,hashlib,os
from collections import defaultdict,Counter
from pathlib import Path

def loadcfg(p):return json.load(open(p))
def parse_subtype(path):
    rows=[]
    with open(path,encoding='utf-8',errors='replace') as f:
        for line in f:
            m=re.match(r'^(\d+)\s+"([^"]+)"\s+(\d+)',line.strip())
            if m:rows.append(m.groups())
    return rows

def sha(p):
    h=hashlib.sha1()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def classify_path(rel):
    s=str(rel).lower()
    tags=[]
    for t in ['axon','dend','ccf','registered','raw','soma','original']:
        if t in s:tags.append(t)
    return '+'.join(tags) if tags else 'unlabeled'

def match_cell(stem,cells):
    nums=re.findall(r'\d+',stem)
    candidates=[stem]+nums+[x.zfill(3) for x in nums]
    for c in candidates:
        if c in cells:return c
    return ''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args()
    c=loadcfg(a.config);root=Path(c['digital_pfc_root'])
    out=Path(c['output_root'])/'11_pfc_reconciliation';out.mkdir(parents=True,exist_ok=True)
    info=root/'pfc_2022/metadata/SubtypeInfo.txt'
    rows=parse_subtype(info)
    byarch=defaultdict(set); subtype={}
    for arch,cell,sub in rows:byarch[arch].add(cell);subtype[(arch,cell)]=int(sub)
    expected=set(subtype)
    report={'expected_neuron_ids':len(expected),'subtype_rows':len(rows),'datasets':{}}
    all_inventory=[]
    for label,base in [
        ('pfc2022',root/'pfc_2022/extracted_by_archive'),
        ('pfc2023',root/'pfc_2023/extracted')]:
        files=sorted(base.rglob('*.swc')); groups=defaultdict(list);unmatched=[]
        for p in files:
            rel=p.relative_to(base); arch=''
            for part in rel.parts:
                if part in byarch:arch=part;break
            cell=match_cell(p.stem,byarch.get(arch,set())) if arch else ''
            rec={'dataset':label,'path':str(p),'relative_path':str(rel),'archive_id':arch,
                 'cell_id':cell,'path_class':classify_path(rel),'bytes':p.stat().st_size}
            if arch and cell: groups[(arch,cell)].append(rec)
            else: unmatched.append(rec)
            all_inventory.append(rec)
        mult=Counter(len(v) for v in groups.values())
        # Hash only paired/multiple groups; useful to distinguish exact duplicates vs components.
        exact_dupe_groups=0;distinct_multi_groups=0
        for key,v in groups.items():
            if len(v)>1:
                hs=[sha(x['path']) for x in v]
                for x,h in zip(v,hs):x['sha1']=h
                if len(set(hs))<len(hs):exact_dupe_groups+=1
                if len(set(hs))>1:distinct_multi_groups+=1
        report['datasets'][label]={
            'raw_swc_files':len(files),'matched_biological_ids':len(groups),
            'unmatched_files':len(unmatched),'files_per_biological_id':dict(sorted(mult.items())),
            'expected_ids_recovered':len(set(groups)&expected),
            'missing_expected_ids':len(expected-set(groups)),
            'exact_duplicate_multi_groups':exact_dupe_groups,
            'distinct_multi_groups':distinct_multi_groups,
            'path_classes':dict(Counter(x['path_class'] for x in all_inventory if x['dataset']==label))
        }
    fields=sorted({k for r in all_inventory for k in r})
    with (out/'pfc_swc_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_inventory)
    (out/'pfc_reconciliation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    # PFC2022 must resolve exactly to the 6357 biological IDs before subtype statistics.
    d=report['datasets']['pfc2022']
    if d['expected_ids_recovered']!=len(expected):
        raise SystemExit('PFC2022 reconciliation did not recover all 6357 SubtypeInfo biological IDs.')
    print('PFC2022 biological-ID reconciliation PASS')
if __name__=='__main__':main()
