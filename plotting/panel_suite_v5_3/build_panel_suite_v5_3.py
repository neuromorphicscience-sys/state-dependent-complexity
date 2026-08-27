from __future__ import annotations

import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'upstream'
RUNTIME = HERE / 'runtime'

STYLE_FILES = ['plot_style_complexity.py','plot_style_v3.py','plot_style_v4.py','plot_style_v5.py','nature_style_v53.py']

@dataclass
class Job:
    key: str
    label: str
    script: str
    args: list[str]
    expected: list[str]


def sha256_file(path: Path, chunk=1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def parse_args():
    ap=argparse.ArgumentParser(description='Neural Science Panel Suite V5.3: complete panel regeneration with semantic colors and 统一重绘：参考用户偏好的实心点/克制配色/热图样式，并保持物理坐标轴长度一致。')
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--stages', nargs='*', default=None, help='Optional job keys. Omit to run the complete suite.')
    ap.add_argument('--fail-fast', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--keep-style-installed', action='store_true')
    ap.add_argument('--skip-heavy-synphys-recompute', action='store_true', help='Use existing SynPhys integrated tables when available; narrative/atlas still run.')
    ap.add_argument('--openscope-strict', action='store_true')
    ap.add_argument('--inventory-bio-data', action='store_true', help='Inventory every file under <root>/bio data. Recommended for final archival audit.')
    return ap.parse_args()


def jobs(root: Path, dpi: int, skip_synphys_integrated: bool, openscope_strict: bool):
    j=[
      Job('stage1','Stage1 baseline direct redraw','neural_plot_pipeline_v2_4.py',['--root','{root}','--plot-root','{plot}','--dpi',str(dpi),'stage1'],['{plot}/Stage1/figures','{plot}/Stage1/source_data']),
      Job('stage1b','Stage1B transition direct redraw','neural_plot_pipeline_v2_4.py',['--root','{root}','--plot-root','{plot}','--dpi',str(dpi),'stage1b'],['{plot}/Stage1B/figures','{plot}/Stage1B/source_data']),
      Job('stage2','Stage2 allocation direct redraw','neural_plot_pipeline_v2_4.py',['--root','{root}','--plot-root','{plot}','--dpi',str(dpi),'stage2'],['{plot}/Stage2/figures','{plot}/Stage2/source_data']),
      Job('stage3','Stage3 scientific atlas direct redraw','neural_plot_pipeline_v2_4.py',['--root','{root}','--plot-root','{plot}','--dpi',str(dpi),'stage3'],['{plot}/Stage3/figures','{plot}/Stage3/source_data']),
      Job('stage4a_analysis','Stage4A manuscript analysis','analyze_stage4a_manuscript_v1_1.py',['--root','{root}'],['{plot}/Stage4A/analysis/stage4a_canonical_methods.csv']),
      Job('stage4a_figures','Stage4A final publication panels','plot_stage4a_final.py',['--root','{root}','--dpi',str(dpi)],['{plot}/Stage4A/figure/panels']),
      Job('stage4_integrated','Stage4 integrated discovery/replication figures','analyze_stage4_integrated_v2_figures.py',['--root','{root}','--dpi',str(dpi)],['{plot}/Stage4_integrated_full/analysis','{plot}/Stage4_integrated_full/figures_main']),
      Job('stage5a','Stage5A robustness closure','stage5a_robustness_closure_v1.py',['--root','{root}'],['{plot}/Stage5A_robustness_v1/analysis','{plot}/Stage5A_robustness_v1/figures']),
      Job('stage5a_epsilon','Stage5A epsilon-sensitivity patch','stage5a_epsilon_sensitivity_patch_v1.py',['--root','{root}','--dpi',str(dpi)],['{plot}/Stage5A_robustness_v1/epsilon_sensitivity_v1/analysis','{plot}/Stage5A_robustness_v1/epsilon_sensitivity_v1/figures']),
      Job('stage5b','Stage5B leverage closure','stage5br_leverage_closure_v1.py',['--root','{root}','--dpi',str(dpi)],['{plot}/Stage5BR_leverage_closure_v1/analysis','{plot}/Stage5BR_leverage_closure_v1/figures']),
      Job('closure_extensions','Stage4 CT/DG + Allen lag-aware closure','analyze_closure_extensions_v1.py',['--root','{root}','--dpi',str(dpi)],['{plot}/ClosureExtensions_v1/analysis','{plot}/ClosureExtensions_v1/figures_stage4_transfer','{plot}/ClosureExtensions_v1/figures_stage4_degeneracy','{plot}/ClosureExtensions_v1/figures_allen_lagaware']),
    ]
    if not skip_synphys_integrated:
        j.append(Job('synphys_integrated','Stage5C SynPhys integrated analysis','stage5c_synphys_integrated_v1.py',['--root','{root}','--dpi',str(dpi)],['{plot}/Stage5C_synphys_integrated_v1/analysis','{plot}/Stage5C_synphys_integrated_v1/tables']))
    j += [
      Job('synphys_narrative','Stage5C SynPhys narrative closure','stage5c_synphys_narrative_closure_v2.py',['--root','{root}','--dpi',str(dpi)],['{plot}/Stage5C_synphys_narrative_closure_v2/analysis','{plot}/Stage5C_synphys_narrative_closure_v2/figures']),
      Job('synphys_atlas','Stage5C SynPhys full figure atlas','stage5c_synphys_figure_atlas_v1.py',['--root','{root}','--dpi',str(dpi)],['{plot}/Stage5C_SynPhys_FigureAtlas_v1/figures']),
      Job('openscope','OpenScope frozen final manuscript atlas','plot_openscope_final.py',['--root','{root}','--dpi',str(dpi)] + (['--strict'] if openscope_strict else []),['{plot}/OpenScope_Manuscript_FigureAtlas_Final/MAIN_candidates','{plot}/OpenScope_Manuscript_FigureAtlas_Final/SI_support','{plot}/OpenScope_Manuscript_FigureAtlas_Final/plot_data']),
    ]
    return j


def install_styles(plot: Path, suite: Path):
    backup=suite/'style_backup'; backup.mkdir(parents=True,exist_ok=True)
    state={}
    for name in STYLE_FILES:
        dst=plot/name; src=RUNTIME/name
        state[name]=dst.exists()
        if dst.exists(): shutil.copy2(dst, backup/name)
        shutil.copy2(src,dst)
    (backup/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    return backup


def restore_styles(plot: Path, backup: Path):
    try: state=json.loads((backup/'state.json').read_text(encoding='utf-8'))
    except Exception: state={}
    for name in STYLE_FILES:
        dst=plot/name; bak=backup/name
        if state.get(name) and bak.exists(): shutil.copy2(bak,dst)
        elif not state.get(name) and dst.exists(): dst.unlink()


def run_job(job: Job, root: Path, plot: Path, log_dir: Path, dpi: int, dry=False):
    script=UPSTREAM/job.script
    fmt={'root':str(root),'plot':str(plot)}
    cmd=[sys.executable,str(script)] + [a.format(**fmt) for a in job.args]
    log=log_dir/f'{job.key}.log'
    env=os.environ.copy()
    env['MPLBACKEND']='Agg'
    env['NS_V53_AXIS_NORMALIZE']='1'
    env['NS_V53_WRITE_AXIS_AUDIT']='1'
    env['NS_V53_STRIP_TITLES']='1'
    env['NATURE_NORMALIZE_AXES']='0'
    env['NATURE_STRIP_TITLES']='1'
    env['NATURE_STRIP_PANEL_LETTERS']='1'
    env['NATURE_OUTPUT_DPI']=str(dpi)
    env['PYTHONPATH']=os.pathsep.join([str(RUNTIME),str(plot),env.get('PYTHONPATH','')])
    t0=time.time()
    if dry:
        log.write_text('DRY RUN\n'+' '.join(cmd),encoding='utf-8')
        return 0,0.0,cmd,log
    with log.open('w',encoding='utf-8') as f:
        f.write('COMMAND\n'+' '.join(cmd)+'\n\n'); f.flush()
        p=subprocess.run(cmd,cwd=str(plot),env=env,stdout=f,stderr=subprocess.STDOUT)
    return p.returncode,time.time()-t0,cmd,log


def classify_file(rel: str, suffix: str):
    s=rel.lower().replace('\\','/')
    if '/source_data/' in s: return 'source_data'
    if '/analysis/' in s or '/tables/' in s: return 'analysis_result'
    if '/metadata/' in s: return 'metadata'
    if '/figures' in s or '/main_candidates/' in s or '/si_support/' in s or suffix in {'.png','.pdf','.svg','.eps','.tif','.tiff'}: return 'figure'
    if '/logs/' in s: return 'log'
    if suffix in {'.tar','.gz','.zip','.sqlite','.db','.npy','.npz','.parquet','.pkl','.pickle'}: return 'data_or_archive'
    return 'other'


def scan_files(base: Path, root: Path, tag: str, hash_small=True):
    rows=[]
    if not base.exists(): return rows
    for p in base.rglob('*'):
        if not p.is_file(): continue
        try: rel=str(p.relative_to(root)).replace('\\','/')
        except Exception: rel=str(p)
        size=p.stat().st_size; suffix=p.suffix.lower()
        sha=''
        if hash_small and size <= 256*1024*1024:
            try: sha=sha256_file(p)
            except Exception: sha=''
        rows.append({'scope':tag,'relative_path':rel,'category':classify_file(rel,suffix),'suffix':suffix,'size_bytes':size,'sha256':sha})
    return rows


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def aggregate_axis_audits(root: Path, suite: Path):
    rows=[]
    for p in (root/'plot').rglob('*.axis.json'):
        if 'Panel_Suite_V5_3' in p.parts: continue
        try: d=json.loads(p.read_text(encoding='utf-8'))
        except Exception: continue
        norm=d.get('normalization') or {}
        target_w=norm.get('target_axis_width_mm'); target_h=norm.get('target_axis_height_mm')
        for a in d.get('axes',[]):
            aw=a.get('axis_width_mm'); ah=a.get('axis_height_mm')
            status='INFO'
            if target_w and target_h and aw is not None and ah is not None:
                ok=abs(float(aw)-float(target_w))<=0.6 and abs(float(ah)-float(target_h))<=0.6
                status='PASS' if ok else 'REVIEW'
            rows.append({'axis_json':str(p.relative_to(root)).replace('\\','/'),'source_file':d.get('file',''),'family':norm.get('family',''),'target_width_mm':target_w,'target_height_mm':target_h,'axis_index':a.get('axis_index'),'axis_width_mm':aw,'axis_height_mm':ah,'status':status})
    write_csv(suite/'physical_axis_audit.csv',rows,['axis_json','source_file','family','target_width_mm','target_height_mm','axis_index','axis_width_mm','axis_height_mm','status'])
    return rows


def input_inventory(root: Path, include_bio: bool):
    rows=[]
    candidates=[root/'results']
    # root-level scientific archives
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in {'.gz','.zip','.tar','.npz','.npy','.csv','.json'}:
            candidates.append(p)
    if include_bio: candidates.append(root/'bio data')
    else:
        for p in [
            root/'bio data'/'stage5_v11_review_bundle.tar.gz', root/'bio data'/'stage5_v12_results.tar.gz',
            root/'bio data'/'stage5_v11_hotfix.zip', root/'bio data'/'stage5_v12_biological_atlas.zip',
            root/'bio data'/'glif_neuronal_models_metadata_raw.json', root/'bio data'/'Stage5B_complete_20260820.tar.gz',
            root/'bio data'/'stage5b_dynamic_leverage_v1.zip', root/'bio data'/'allen_synap'/'synphys_r2.1_full.sqlite',
        ]:
            if p.exists(): candidates.append(p)
        for d in [root/'plot'/'OpenScope_Illusion_Analysis_v2_3',root/'plot'/'OpenScope_FinalClosure_v1']:
            if d.exists(): candidates.append(d)
    seen=set()
    for c in candidates:
        if not c.exists(): continue
        files=[c] if c.is_file() else list(c.rglob('*'))
        for p in files:
            if not p.is_file(): continue
            rp=str(p.resolve())
            if rp in seen: continue
            seen.add(rp)
            try: rel=str(p.relative_to(root)).replace('\\','/')
            except Exception: rel=str(p)
            rows.append({'relative_path':rel,'suffix':p.suffix.lower(),'size_bytes':p.stat().st_size})
    return rows


def main():
    args=parse_args(); root=args.root.resolve()
    if not root.exists(): raise SystemExit(f'Project root not found: {root}')
    plot=root/'plot'; plot.mkdir(parents=True,exist_ok=True)
    suite=plot/'Panel_Suite_V5_3'; suite.mkdir(parents=True,exist_ok=True)
    logs=suite/'run_logs'; logs.mkdir(exist_ok=True)

    all_jobs=jobs(root,args.dpi,args.skip_heavy_synphys_recompute,args.openscope_strict)
    if args.stages:
        wanted=set(args.stages); known={j.key for j in all_jobs}
        bad=sorted(wanted-known)
        if bad: raise SystemExit(f'Unknown job keys: {bad}')
        all_jobs=[j for j in all_jobs if j.key in wanted]

    backup=install_styles(plot,suite)
    results=[]
    try:
        for i,j in enumerate(all_jobs,1):
            print(f'[{i}/{len(all_jobs)}] {j.key}: {j.label}')
            rc,dt,cmd,log=run_job(j,root,plot,logs,args.dpi,args.dry_run)
            exp=[x.format(root=str(root),plot=str(plot)) for x in j.expected]
            exist=[Path(x).exists() for x in exp]
            status='DRY_RUN' if args.dry_run else ('PASS' if rc==0 and all(exist) else ('PARTIAL' if rc==0 else 'FAIL'))
            results.append({'job':j.key,'label':j.label,'status':status,'return_code':rc,'elapsed_s':round(dt,1),'expected_outputs':' | '.join(exp),'expected_present':' | '.join(map(str,exist)),'log':str(log)})
            if rc!=0 and args.fail_fast and not args.dry_run: break
    finally:
        if not args.keep_style_installed: restore_styles(plot,backup)

    manifest_path=suite/'run_manifest.csv'
    if args.stages and manifest_path.exists():
        with manifest_path.open('r',newline='',encoding='utf-8-sig') as f:
            previous=list(csv.DictReader(f))
        merged={r['job']:r for r in previous}
        merged.update({r['job']:r for r in results})
        preferred=[j.key for j in jobs(root,args.dpi,args.skip_heavy_synphys_recompute,args.openscope_strict)]
        results=[merged[k] for k in preferred if k in merged] + [r for k,r in merged.items() if k not in preferred]
    write_csv(manifest_path,results,['job','label','status','return_code','elapsed_s','expected_outputs','expected_present','log'])

    # Exhaustive output inventory: no deduplication, no dropping.
    output_roots=[
      'Stage1','Stage1B','Stage2','Stage3','Stage4A','Stage4_integrated_full','Stage5A_robustness_v1',
      'Stage5BR_leverage_closure_v1','ClosureExtensions_v1','Stage5C_synphys_integrated_v1',
      'Stage5C_synphys_narrative_closure_v2','Stage5C_SynPhys_FigureAtlas_v1','OpenScope_Manuscript_FigureAtlas_Final',
      'OpenScope_Illusion_Analysis_v2_3','OpenScope_FinalClosure_v1'
    ]
    inv=[]
    for name in output_roots: inv += scan_files(plot/name,root,name)
    write_csv(suite/'complete_output_inventory.csv',inv,['scope','relative_path','category','suffix','size_bytes','sha256'])
    write_csv(suite/'source_data_inventory.csv',[r for r in inv if r['category']=='source_data'],['scope','relative_path','category','suffix','size_bytes','sha256'])
    write_csv(suite/'analysis_result_inventory.csv',[r for r in inv if r['category']=='analysis_result'],['scope','relative_path','category','suffix','size_bytes','sha256'])
    write_csv(suite/'figure_inventory.csv',[r for r in inv if r['category']=='figure'],['scope','relative_path','category','suffix','size_bytes','sha256'])
    inp=input_inventory(root,args.inventory_bio_data)
    write_csv(suite/'input_dataset_inventory.csv',inp,['relative_path','suffix','size_bytes'])
    axis_rows=aggregate_axis_audits(root,suite)

    summary={
      'version':'5.3','root':str(root),'jobs_total':len(results),'jobs_pass':sum(r['status']=='PASS' for r in results),
      'jobs_fail':sum(r['status']=='FAIL' for r in results),'jobs_partial':sum(r['status']=='PARTIAL' for r in results),
      'output_files_cataloged':len(inv),'source_data_files':sum(r['category']=='source_data' for r in inv),
      'analysis_result_files':sum(r['category']=='analysis_result' for r in inv),'figure_files':sum(r['category']=='figure' for r in inv),
      'input_dataset_files_cataloged':len(inp),'axis_audit_rows':len(axis_rows),
      'color_contract':str(HERE/'config'/'COLOR_CONTRACT_V5_3.json'),
      'axis_contract':str(HERE/'config'/'PHYSICAL_AXIS_CONTRACT_V5_3.csv'),
    }
    (suite/'V5_3_RUN_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nV5.3 complete')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if any(r['status']=='FAIL' for r in results): raise SystemExit(2)

if __name__=='__main__': main()
