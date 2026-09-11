"""CPU-only processed-data reproduction of the revision's principal contrasts.

This verifies published numerical estimands from graph-level and per-mask score
tables. It does not claim to rerun simulations, raw recordings or model fitting.
"""
from pathlib import Path
import argparse,csv,hashlib,json,platform,sys
import numpy as np

IDS={
 'tasks':'sd_c16cc414a8f01ec8f726','task_statistics':'sd_2fe45a875165fe9d1f47',
 'task_protocol':'sd_eab3d59c11bcbc7fbc32','crossover':'sd_768480c9a5c0bc62aa22',
 'shared_effects':'sd_8807141fbfd50009bf91','shared_random':'sd_b8f457b5267e3545f831',
 'shared_statistics':'sd_f4f3a571f4abde360ac3','stress':'sd_0f2d2f7e53593c8c9573',
 'stress_statistics':'sd_62025e1a4762177f5bc6'}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def seed(*parts):
    return int.from_bytes(hashlib.sha256(('NCS06-v1|'+'|'.join(map(str,parts))).encode()).digest()[:4],'little')
def exact(values,one_sided=False,tolerance=1e-12):
    values=np.asarray(values,float)
    if values.ndim!=1 or not 1<=len(values)<=20 or not np.isfinite(values).all():raise ValueError('Finite independent-unit vector of length 1-20 required')
    signs=2*((np.arange(2**len(values))[:,None]>>np.arange(len(values)))&1)-1
    null=(signs*values).mean(1);observed=values.mean()
    return float(np.mean(null>=observed-tolerance) if one_sided else np.mean(np.abs(null)>=abs(observed)-tolerance))
def holm(values):
    p=np.asarray(values,float);order=np.argsort(p);out=np.empty(len(p));running=0.
    for rank,index in enumerate(order):running=max(running,min(1.,(len(p)-rank)*p[index]));out[index]=running
    return out
def close(actual,expected):np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

class Data:
    def __init__(self,root):
        self.root=Path(root);self.index=json.loads((self.root/'DATASET_INDEX.json').read_text())
        self.records={r['id']:r for r in self.index['datasets']};self.used={}
    def load(self,key):
        record=self.records[IDS[key]];path=self.root/record['file']
        assert digest(path)==record['sha256'],f'Checksum mismatch: {key}'
        self.used[key]={'id':record['id'],'sha256':record['sha256']}
        if path.suffix=='.json':return json.loads(path.read_text())
        with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    def verify_all(self):
        manifest=json.loads((self.root/'SHA256SUMS.json').read_text())
        actual={str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
        assert set(manifest)==actual,'The revision file inventory does not match its checksum manifest.'
        for name,expected in manifest.items():
            path=(self.root/name).resolve()
            assert path.is_relative_to(self.root.resolve()),'Manifest path escapes the data revision.'
            assert digest(path)==expected,name
        return len(manifest)

def verify_tasks(data):
    protocol=data.load('task_protocol');statistics=data.load('task_statistics');rows=data.load('tasks')
    graphs,tasks,states=protocol['graph_ids'],protocol['tasks'],protocol['couplings']
    assert len(graphs)==12 and len(tasks)==6 and len(states)==2
    table={}
    for row in rows:
        key=(row['graph'],row['task'],float(row['coupling']),row['method']);assert key not in table
        value=float(row['score']);value=value if row['task']=='delayed_recall' else 2*value-1
        close(value,float(row['normalized_score']));table[key]=value
    assert len(rows)==statistics['verified_prediction_score_pairs']==3024
    rng=np.random.default_rng(seed('analysis-bootstrap-v1'));checks=[]
    def verify(values,result,label):
        v=np.asarray(values);close(v,result['values']);close(v.mean(),result['mean'])
        p=exact(v);assert p==result['exact_two_sided_sign_flip_p']
        assert len(v)==result['n_graphs']==12
        assert int((v>0).sum())==result['positive_graphs'] and int((v==0).sum())==result['zero_graphs']
        ci=np.percentile(v[rng.integers(0,len(v),size=(20000,len(v)))].mean(1),[2.5,97.5])
        close(ci,result['ci95_percentile_bootstrap'])
        checks.append({'endpoint':label,'mean':float(v.mean()),'n_graphs':12,'ci95':ci.tolist(),'exact_two_sided_p':p,'passed':True})
    for contrast,prefix,count in [('population_vs_budget_random','random_search',1),('population_vs_unselected_random','unselected_random_',8),('population_vs_group_matched','group_matched_random_',4)]:
        delta=np.empty((12,6))
        for gi,g in enumerate(graphs):
            for ti,t in enumerate(tasks):
                worst=lambda m:min(table[g,t,s,m] for s in states)
                methods=sorted({key[3] for key in table if key[:2]==(g,t) and key[3].startswith(prefix)});assert len(methods)==count
                delta[gi,ti]=worst('population_search')-np.mean([worst(m) for m in methods])
        result=statistics['results'][contrast];raw=[]
        for ti,t in enumerate(tasks):
            verify(delta[:,ti],result['tasks'][t],contrast+'/'+t);raw.append(result['tasks'][t]['exact_two_sided_sign_flip_p'])
        for ti,t in enumerate(tasks):close(holm(raw)[ti],result['tasks'][t]['holm_six_task_p'])
        verify(delta.mean(1),result['equal_weight_six_task_graph_aggregate'],contrast+'/six_task_aggregate')
    interaction=[np.mean([(table[g,t,states[1],'population_search']-table[g,t,states[1],'random_search'])-(table[g,t,states[0],'population_search']-table[g,t,states[0],'random_search']) for t in tasks]) for g in graphs]
    verify(interaction,statistics['results']['secondary_state_interaction']['equal_weight_six_task_graph_aggregate'],'secondary_state_interaction/six_task_aggregate')
    assert len(checks)==22
    return {'score_rows':3024,'endpoint_checks':checks,'holm_families':3,'bootstrap_resamples_per_endpoint':20000,'mean_of_per_mask_minimum_verified':True}

def verify_shared(data):
    rows=data.load('shared_effects');random=data.load('shared_random');reference=data.load('shared_statistics')
    assert len(rows)==len(random)==12 and len({r['graph_seed'] for r in rows})==12
    checks=[]
    for name,table in [(name,rows) for name in ['D_g','d_mid','d_sparse']]+[(name,random) for name in ['gain_mid','gain_sparse','gain_maximin']]:
        v=np.array([float(r[name]) for r in table]);ref=reference[name]
        close(v.mean(),ref['mean']);assert exact(v)==ref['exact_signflip_p'];assert int((v>0).sum())==ref['positive_graphs']
        # The frozen R4 analysis restarts this same RNG for each endpoint.
        rng=np.random.default_rng(7992675377723887068)
        ci=np.quantile(v[rng.integers(0,12,size=(10000,12))].mean(1),[.025,.975])
        close(ci,[ref['ci95_low'],ref['ci95_high']])
        checks.append({'endpoint':name,'mean':float(v.mean()),'ci95':ci.tolist(),'exact_two_sided_p':exact(v),'passed':True})
    for r in rows:
        shared=min(float(r['V_shared_mid']),float(r['V_shared_sparse']))
        specialists=min(float(r['V_specialized_mid']),float(r['V_specialized_sparse']))
        close(shared,float(r['J_shared']));close(specialists,float(r['J_specialized_policy']));close(specialists-shared,float(r['D_g']))
    upper={}
    for state in ['mid','sparse']:
        v=np.array([float(r['d_'+state]) for r in rows]);upper[state]=float(v.mean()+1.959963984540054*v.std(ddof=1)/np.sqrt(12))
        close(upper[state],reference['sufficiency'][state+'_ucb'])
    assert all(v>.02 for v in upper.values()) and reference['sufficiency']['sufficient'] is False
    for labels,key in [(['d_mid','d_sparse'],'holm_p_two_state_family'),(['gain_mid','gain_sparse','gain_maximin'],'holm_p_exploratory_three_control_family')]:
        adjusted=holm([reference[n]['exact_signflip_p'] for n in labels])
        for n,p in zip(labels,adjusted):close(p,reference[n][key])
    return {'n_reused_graphs':12,'endpoint_checks':checks,'normal_approximation_upper_bounds':upper,'sufficiency_established':False,'bootstrap_resamples_per_endpoint':10000,'bootstrap_seed':7992675377723887068,'holm_families_verified':2}

def verify_stress(data):
    rows=data.load('stress');ref=data.load('stress_statistics');v=np.array([float(r['grid_mean_gain']) for r in rows])
    assert len(v)==12 and len({r['graph_seed'] for r in rows})==12
    assert all(int(r['n_conditions'])==15 for r in rows)
    expected=ref['primary_secondary_protocol_contrast'];close(v.mean(),expected['mean']);assert exact(v)==expected['exact_two_sided_signflip_p']
    rng=np.random.default_rng(ref['graph_bootstrap_seed']);ci=np.percentile(v[rng.integers(0,12,(ref['bootstrap_resamples'],12))].mean(1),[2.5,97.5])
    close(ci,[expected['ci95_low'],expected['ci95_high']])
    return {'n_reused_graphs':12,'conditions_per_graph':15,'mean':float(v.mean()),'ci95':ci.tolist(),'exact_two_sided_p':exact(v),'passed':True}

def verify_crossover(data):
    rows=data.load('crossover');assert len(rows)==12 and len({r['graph_seed'] for r in rows})==12
    out={}
    for field,p_expected,positive in [('primary',.0029296875,10),('matched',.000244140625,12)]:
        v=np.array([float(r[field]) for r in rows]);assert exact(v,True)==p_expected and int((v>0).sum())==positive
        out[field]={'mean':float(v.mean()),'exact_one_sided_p':p_expected,'positive_graphs':positive,'n_graphs':12}
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output path; previous reports are preserved.')
    data=Data(args.data_root);count=data.verify_all()
    report={'passed':True,'scope':'processed-data inference reproduction, not raw-data or simulation rerun','verified_manifest_files':count,
            'tasks':verify_tasks(data),'shared':verify_shared(data),'local_stress':verify_stress(data),'crossover':verify_crossover(data),
            'used_inputs':data.used,'python':platform.python_version(),'numpy':np.__version__,'script_sha256':digest(Path(__file__)),
            'limitations':['No raw biological model fitting is rerun here.','No search or new scientific simulation is run here.','This command audits the listed quantitative endpoints, not every descriptive biological panel.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'passed':True,'task_endpoints':22,'shared_endpoints':6,'stress_endpoints':1,'crossover_endpoints':2,'output':str(args.output)},indent=2))
if __name__=='__main__':main()
