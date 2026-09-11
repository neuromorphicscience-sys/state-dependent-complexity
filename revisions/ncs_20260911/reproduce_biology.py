"""Recompute eight biological result groups from the published processed tables.

No raw-data fitting, state estimation or permutation model refitting occurs here.
The fixed-repeat null and the repeat-averaged effect remain separate estimands.
"""
from pathlib import Path
import argparse,csv,json,platform
import numpy as np
from reproduce_revision import Data,close,digest,exact

BIO_IDS={
 'primary_pairs':'sd_eee33052633db55c847e','primary_delta':'sd_d220e46a29901760a7ac',
 'primary_formal':'sd_66abcb2658a4c26e2ba4','decoder_null':'sd_92e3c3dedd1577c5b460',
 'same_image':'sd_b44b2edf819359aed2a3','same_image_formal':'sd_23b5385fd32c0fb107a3',
 'split_population':'sd_95644ce1ca8780567e7e','landscape_null':'sd_ac4f913658cf71fdf6da',
 'jaccard_null':'sd_0b6095f94b16d070ced8','glif_null':'sd_b593a20cfbcae9fd0b6a',
 'glif_predictions':'sd_676665dd72e834e30e7c','glif_costs':'sd_43b567a5360385353e1e'}

def load(data,key):
    rec=data.records[BIO_IDS[key]];p=data.root/rec['file']
    assert digest(p)==rec['sha256'],key
    data.used[key]={'id':rec['id'],'sha256':rec['sha256']}
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def vector(values):
    x=np.asarray(values,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty independent-unit vector required')
    return x

def bootstrap(x,seed,n=50000):
    x=vector(x);rng=np.random.default_rng(seed)
    return np.quantile(x[rng.integers(0,len(x),(n,len(x)))].mean(1),[.025,.975])

def ranks(x):
    """Average ranks for ties, preserving original observation order."""
    x=vector(x);order=np.argsort(x,kind='stable');r=np.empty(len(x));start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        r[order[start:end]]=(start+1+end)/2;start=end
    return r

def permutation(null,observed):
    x=vector(null)
    if not np.isfinite(observed):raise ValueError('Finite observed statistic required')
    k=int((x>=observed).sum())
    return {'n_permutations':len(x),'n_null_ge_observed':k,'one_sided_plus_one_p':(k+1)/(len(x)+1),'observed':float(observed)}

def unique(rows,key,n):
    d={r[key]:r for r in rows}
    assert len(d)==len(rows)==n,('Duplicate or missing independent units',key)
    return d

def check_formal(values,reference,seed):
    x=vector(values);ref=reference
    assert len(x)==int(ref['n_mouse_sessions'])==12
    close(x.mean(),float(ref['mean_effect']))
    close(np.median(x),float(ref['median_effect']))
    close(x.std(ddof=1),float(ref['sd_effect']))
    assert int((x>0).sum())==int(ref['n_positive'])
    assert int((x<0).sum())==int(ref['n_negative'])
    p=exact(x,True,tolerance=1e-15)
    close(p,float(ref['exact_signflip_one_sided_p']))
    close(exact(x,tolerance=1e-15),float(ref['exact_signflip_two_sided_p']))
    ci=bootstrap(x,seed);close(ci,[float(ref['bootstrap_ci_low']),float(ref['bootstrap_ci_high'])])
    return {'n_mice':12,'mean':float(x.mean()),'positive_mice':int((x>0).sum()),'exact_one_sided_p':p,
            'ci95':ci.tolist(),'bootstrap_resamples':50000,'bootstrap_seed':seed,'passed':True}

def verify(data):
    checks={}
    pairs=unique(load(data,'primary_pairs'),'subject',12)
    delta=unique(load(data,'primary_delta'),'subject',12)
    assert pairs.keys()==delta.keys()
    for mouse,r in pairs.items():close(float(r['decoder_auc_home'])-float(r['decoder_auc_cross']),float(delta[mouse]['decoder_auc_crossover']))
    values=[float(r['decoder_auc_crossover']) for r in delta.values()]
    ref=load(data,'primary_formal');assert len(ref)==1
    checks['primary_decoder_transfer']=check_formal(values,ref[0],20260821+1000)
    same=unique(load(data,'same_image'),'subject',12);assert same.keys()==pairs.keys()
    values=[float(r['same_image_state_auc_excess_mean']) for r in same.values()]
    ref=load(data,'same_image_formal');assert len(ref)==1
    close(np.mean(values)+.5,float(ref[0]['mean_raw']));assert float(ref[0]['null'])==.5
    checks['same_image_state_imprint']=check_formal(values,ref[0],20260821+1006)
    split=unique(load(data,'split_population'),'session',12)
    subjects={session.split('_')[0].removeprefix('sub-') for session in split}
    assert subjects==pairs.keys() and len(subjects)==12
    for r in split.values():
        assert int(r['repeats'])==24
        close(float(r['decoder_auc_home'])-float(r['decoder_auc_cross']),float(r['decoder_auc_crossover']))
    x=np.array([float(r['decoder_auc_crossover']) for r in split.values()])
    p=exact(x,True,tolerance=1e-15);assert p==.00048828125 and (x>0).sum()==11
    checks['disjoint_population_transfer']={'mean':float(x.mean()),'n_reused_mice':12,'positive_mice':11,'exact_one_sided_p':p,'passed':True}
    for key,expected in [('decoder_null',4),('landscape_null',1168),('jaccard_null',808)]:
        rows=load(data,key);observed={float(r['observed']) for r in rows};assert len(observed)==1
        result=permutation([float(r['null']) for r in rows],observed.pop())
        assert result['n_permutations']==2000 and result['n_null_ge_observed']==expected
        result.update(passed=True,estimand='fixed_repeat_0_group_mean_not_repeat_averaged_effect')
        checks[key]=result
    assert not np.isclose(checks['decoder_null']['observed'],checks['primary_decoder_transfer']['mean'],rtol=0,atol=1e-6)
    assert checks['landscape_null']['one_sided_plus_one_p']>.05 and checks['jaccard_null']['one_sided_plus_one_p']>.05
    costs=unique(load(data,'glif_costs'),'specimen_id',400)
    predictions=unique(load(data,'glif_predictions'),'specimen_id',400)
    assert costs.keys()==predictions.keys()
    for ident,r in costs.items():
        ev=[float(r[f'ev_glif{i}']) for i in range(1,6)]
        levels=np.array([ev[0],max(ev[1],ev[2]),ev[3],ev[4]])
        close(levels,[float(r[f'ev_cost{i}']) for i in range(4)])
        close(max(levels),float(r['ev_best']))
        for eps in [.0,.005,.01,.015,.02,.025,.03,.04,.05]:
            key='creq_eps_'+f'{eps:.3f}'.replace('.','p')
            expected=int(np.flatnonzero(levels>=max(levels)-eps-1e-15)[0])
            assert int(r[key])==expected,(ident,key)
        assert float(predictions[ident]['C_req'])==float(r['creq_eps_0p020'])
    counts=np.bincount([int(r['creq_eps_0p020']) for r in costs.values()],minlength=4)
    np.testing.assert_array_equal(counts,[105,144,85,66])
    checks['glif_mechanism_cost']={'n_cells':400,'baseline_epsilon':.02,'counts_cost0_to3':counts.tolist(),'tolerances_checked':9,'GLIF2_GLIF3_share_cost1':True,'passed':True}
    y=np.array([float(r['C_req']) for r in predictions.values()]);pred=np.array([float(r['pred_nested_rerun']) for r in predictions.values()])
    rho=float(np.corrcoef(ranks(y),ranks(pred))[0,1]);close(rho,.2527190709987035)
    null=load(data,'glif_null');assert {int(r['permutation']) for r in null}==set(range(1,301)) and len(null)==300
    result=permutation([float(r['rho']) for r in null],rho);assert result['n_null_ge_observed']==0
    result.update(n_cells=400,MAE=float(np.abs(y-pred).mean()),naive_median_MAE=float(np.abs(y-np.median(y)).mean()),passed=True)
    close(result['MAE'],.8378933723085638)
    assert result['MAE']>result['naive_median_MAE']
    checks['glif_nested_rank_and_permutation']=result
    assert len(checks)==8
    return checks

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new report path; old outputs are preserved.')
    data=Data(args.data_root);count=data.verify_all();checks=verify(data)
    report={'passed':True,'scope':'Eight processed biological result groups; not raw-data model refitting',
            'verified_manifest_files':count,'checks':checks,'inputs':data.used,'python':platform.python_version(),'numpy':np.__version__,
            'script_sha256':digest(Path(__file__)),'limitations':['Permutation statistics are recalculated from archived null draws; no models are refitted.',
            'This does not validate raw-data version identities, every descriptive biological panel, or causal mechanism reallocation.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'passed':True,'biological_result_groups':len(checks),'output':str(args.output)}))
if __name__=='__main__':main()
