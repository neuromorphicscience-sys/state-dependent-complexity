"""Frozen task orchestration, exposed for synthetic control-flow audit.

Scientific methods are unchanged from the archived run_formal.py. The original
workstation constructor/CLI are deliberately not reproduced here. Instantiation
fails closed. See WORKFLOW_AUDIT.md before reusing this audit-only class.
"""
from __future__ import annotations
from benchmark_core import *

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def mask_id(mask):
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()[:20]

def atomic_npz(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays)
    os.replace(temp,path)

class Experiment:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'Audit-only controller: no portable production launcher or QA gate '
            'is supplied. Use the documented synthetic CPU test subclass only.')

    def log(self, **record):
        record.update(utc=dtm.datetime.now(dtm.timezone.utc).isoformat(),pid=os.getpid())
        with (self.out/'events.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        atomic_json(self.out/'progress.json',record)
        print(json.dumps(record),flush=True)

    def features(self, taskdir, graph_id, task, A, projection, masks, state, role, block):
        ids=[mask_id(m) for m in masks]
        metadata=dict(protocol=self.hash,graph=graph_id,task=task,ids=ids,state=state,role=role,block=block)
        key=digest(metadata)
        path=taskdir/'feature_cache'/f'{key}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if str(z['metadata_hash'])!=key: raise RuntimeError('Cache identity mismatch')
                return z['features'],z['labels']
        guard(600)
        mb=self.p['mask_batch']; tb=self.p['trial_batch']
        if len(masks)>mb: raise ValueError('Mask batch overflow')
        padded=np.concatenate([masks,np.repeat(masks[:1],mb-len(masks),axis=0)],axis=0)
        x,y=task_data(task,role,block,trials=tb,amplitude=self.p['amplitude'],version=self.p['task_version'])
        start=time.perf_counter()
        feats,metrics=simulate_driven(self.p['cfg'],A,torch.as_tensor(padded,device='cuda'),state,self.p['noise_sigma'],
            seed('noise',graph_id,task,role,block),x,projection,filter_tau_ms=self.p['filter_tau_ms'],
            check=lambda: (_ for _ in ()).throw(RuntimeError('deadline reserve reached')) if time.time()>DEADLINE-600 else None)
        feats=feats[:len(masks)]
        atomic_npz(path,features=feats,labels=y,metadata_hash=np.asarray(key),
            **{'diagnostic_'+k:v[:len(masks)] for k,v in metrics.items()})
        self.log(event='simulation_block_complete',graph=graph_id,task=task,state=state,role=role,block=block,
            actual_masks=len(masks),padded_masks=mb,trials=tb,seconds=time.perf_counter()-start,
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),cache=path.name)
        return feats,y

    def evaluate_new(self, taskdir, graph_id, task, A, projection, records, new):
        """Train on search_train only, score on search_validation only."""
        for start in range(0,len(new),self.p['mask_batch']):
            group=new[start:start+self.p['mask_batch']]
            masks=np.stack([r['mask'] for r in group])
            for state in self.p['couplings']:
                train_parts=[self.features(taskdir,graph_id,task,A,projection,masks,state,'search_train',b)
                             for b in range(self.p['train_blocks'])]
                val_parts=[self.features(taskdir,graph_id,task,A,projection,masks,state,'search_validation',b)
                           for b in range(self.p['search_blocks'])]
                train=np.concatenate([x for x,y in train_parts],axis=1); ytrain=np.concatenate([y for x,y in train_parts])
                val=np.concatenate([x for x,y in val_parts],axis=1); yval=np.concatenate([y for x,y in val_parts])
                for j,record in enumerate(group):
                    model=fit_readout(train[j],ytrain,alpha=self.p['ridge_alpha'])
                    record['models'][str(state)]=model
                    record['search_scores'][str(state)]=score(yval,predict(val[j],model),task)
            for record in group:
                record['search_objective']=min(record['search_scores'].values())
                records[record['id']]=record

    def candidate(self, mask, origin):
        if int(mask.sum())!=self.p['hh_budget']: raise ValueError('HH budget mismatch')
        return dict(id=mask_id(mask),mask=np.asarray(mask,bool),origin=origin,models={},search_scores={})

    def run_task(self, graph_id, task):
        taskdir=self.out/str(graph_id)/task; taskdir.mkdir(parents=True,exist_ok=True)
        if (taskdir/'complete.json').exists():
            done=json.loads((taskdir/'complete.json').read_text())
            if done['protocol_hash']!=self.hash: raise RuntimeError('Completed artifact protocol mismatch')
            self.log(event='resume_completed_task',graph=graph_id,task=task); return
        guard(1200)
        a,modules=graph(graph_id,self.p['n_neurons']); A=topology.to_gpu_adjacency(a,'cuda')
        projection=input_projection(graph_id,self.p['n_neurons'])
        graphpath=self.out/str(graph_id)/'graph.npz'
        if not graphpath.exists(): atomic_npz(graphpath,adjacency=a,modules=modules,projection=projection,graph_seed=seed('graph',graph_id))
        rng=np.random.default_rng(seed('optimizer',graph_id,task)); k=self.p['hh_budget']; n=len(a)
        records={}; seen=set()
        def random_mask():
            m=np.zeros(n,bool); m[rng.choice(n,k,replace=False)]=True; return m
        def unique(factory):
            for _ in range(10000):
                m=factory(); ident=mask_id(m)
                if ident not in seen:
                    seen.add(ident); return m
            raise RuntimeError('Unable to produce a unique mask')
        initial=[self.candidate(unique(random_mask),'shared_initial') for _ in range(self.p['initial_candidates'])]
        initial_ids=[r['id'] for r in initial]
        pools={method:list(initial_ids) for method in ('random_search','local_search','population_search')}
        baselines=[]
        for name in ('high_degree','spectral'):
            m=topology.heuristic_mask(a,modules,k,name,seed('static',graph_id,task,name)); seen.add(mask_id(m))
            baselines.append(self.candidate(m,name))
        # Strong fixed input-access comparator: rank squared current exposure,
        # estimated from training inputs only, then degree as deterministic tie-break.
        xtrain=np.concatenate([task_data(task,'search_train',b,trials=self.p['trial_batch'],amplitude=self.p['amplitude'],version=self.p['task_version'])[0]
                               for b in range(self.p['train_blocks'])])
        exposure=(np.mean(xtrain.astype(np.float64)**2,axis=(0,1))[:,None]*(projection**2)).sum(0)
        order=np.lexsort((np.arange(n),a.sum(0)+a.sum(1),exposure))[::-1]
        m=np.zeros(n,bool); m[order[:k]]=True; seen.add(mask_id(m)); baselines.append(self.candidate(m,'input_energy_degree'))
        # Equal per-input-group HH quota; not selected by outcomes.
        m=np.zeros(n,bool)
        qrng=np.random.default_rng(seed('quota',graph_id,task))
        for ch in range(4): m[qrng.choice(np.flatnonzero(projection[ch]),k//4,replace=False)]=True
        seen.add(mask_id(m)); baselines.append(self.candidate(m,'input_quota_random'))
        self.evaluate_new(taskdir,graph_id,task,A,projection,records,initial+baselines)
        history=[]
        def mutate(parent, swaps=2):
            m=parent.copy(); m[rng.choice(np.flatnonzero(m),swaps,replace=False)]=False
            # additions drawn from the original complement; exactly `swaps` replacements
            m[rng.choice(np.flatnonzero(~parent),swaps,replace=False)]=True; return m
        for generation in range(self.p['generations']):
            new=[]
            for method in pools:
                ranking=sorted(pools[method],key=lambda q:(-records[q]['search_objective'],q))
                elite=ranking[:min(self.p['elite_count'],len(ranking))]
                for j in range(self.p['offspring_per_generation']):
                    if method=='random_search': factory=random_mask
                    elif method=='local_search':
                        parent=records[ranking[0]]['mask']; factory=lambda p=parent:mutate(p,2)
                    else:
                        p1=records[elite[int(rng.integers(len(elite)))]]['mask']
                        p2=records[elite[int(rng.integers(len(elite)))]]['mask']
                        def factory(p1=p1,p2=p2):
                            common=np.flatnonzero(p1&p2); union=np.flatnonzero(p1^p2)
                            child=np.zeros(n,bool); child[common]=True
                            child[rng.choice(union,k-len(common),replace=False)]=True
                            return mutate(child,2)
                    candidate=self.candidate(unique(factory),f'{method}_generation_{generation}')
                    pools[method].append(candidate['id']); new.append(candidate)
                history.append(dict(generation=generation,method=method,parents_ranked_before_generation=ranking))
            self.evaluate_new(taskdir,graph_id,task,A,projection,records,new)
            self.log(event='search_generation_complete',graph=graph_id,task=task,generation=generation,unique_candidates=len(records))
        def public(r):
            return {key:r[key] for key in ('id','origin','search_scores','search_objective')}
        atomic_json(taskdir/'search_candidates.json',dict(protocol_hash=self.hash,pools=pools,initial_ids=initial_ids,
            baselines={r['origin']:r['id'] for r in baselines},records=[public(r) for r in records.values()],history=history,
            unique_budget_per_method=self.p['initial_candidates']+self.p['generations']*self.p['offspring_per_generation']))
        atomic_npz(taskdir/'all_search_masks.npz',ids=np.asarray(list(records)),masks=np.stack([r['mask'] for r in records.values()]))
        shortlist={method:sorted(ids,key=lambda q:(-records[q]['search_objective'],q))[:self.p['shortlist_size']] for method,ids in pools.items()}
        specialists={str(s):sorted(pools['population_search'],key=lambda q:(-records[q]['search_scores'][str(s)],q))[:self.p['shortlist_size']]
                     for s in self.p['couplings']}
        selection_ids=sorted(set(sum(shortlist.values(),[])+sum(specialists.values(),[])))
        selection_scores={ident:{} for ident in selection_ids}
        for offset in range(0,len(selection_ids),self.p['mask_batch']):
            ids=selection_ids[offset:offset+self.p['mask_batch']]; masks=np.stack([records[q]['mask'] for q in ids])
            for state in self.p['couplings']:
                parts=[self.features(taskdir,graph_id,task,A,projection,masks,state,'independent_selection',b)
                       for b in range(self.p['selection_blocks'])]
                feat=np.concatenate([x for x,y in parts],axis=1); y=np.concatenate([y for x,y in parts])
                for j,q in enumerate(ids): selection_scores[q][str(state)]=score(y,predict(feat[j],records[q]['models'][str(state)]),task)
        chosen={method:sorted(ids,key=lambda q:(-min(selection_scores[q].values()),q))[0] for method,ids in shortlist.items()}
        chosen.update({f'specialist_pool_{s}':sorted(ids,key=lambda q:(-selection_scores[q][s],q))[0] for s,ids in specialists.items()})
        chosen.update({r['origin']:r['id'] for r in baselines})
        chosen.update({f'unselected_random_{i:02d}':q for i,q in enumerate(initial_ids)})
        # Post-selection mechanistic control: preserve the selected HH count in
        # every input group, randomize identities within groups, never optimize.
        # This cannot influence the already fixed algorithm choices.
        parent=records[chosen['population_search']]['mask']
        control_rng=np.random.default_rng(seed('group_matched_control',graph_id,task))
        controls=[]; group_counts=[]
        groups=[np.flatnonzero(projection[ch]) for ch in range(4)]
        group_counts=[int(parent[g].sum()) for g in groups]
        for ci in range(self.p.get('matched_controls',4)):
            m=np.zeros(n,bool)
            for g,count in zip(groups,group_counts):
                m[control_rng.choice(g,count,replace=False)]=True
            r=self.candidate(m,f'group_matched_random_{ci:02d}')
            controls.append(r); chosen[r['origin']]=r['id']
        if controls:
            masks=np.stack([r['mask'] for r in controls])
            for state in self.p['couplings']:
                parts=[self.features(taskdir,graph_id,task,A,projection,masks,state,'search_train',b)
                       for b in range(self.p['train_blocks'])]
                features=np.concatenate([x for x,y in parts],axis=1); y=np.concatenate([y for x,y in parts])
                for j,r in enumerate(controls): r['models'][str(state)]=fit_readout(features[j],y,alpha=self.p['ridge_alpha'])
            for r in controls: records[r['id']]=r
        atomic_json(taskdir/'selection_frozen.json',dict(protocol_hash=self.hash,shortlists=shortlist,specialist_shortlists=specialists,
            independent_selection_scores=selection_scores,chosen=chosen,group_matched_hh_counts=group_counts,
            final_data_consumed=False))
        final_ids=sorted(set(chosen.values())); final_masks=np.stack([records[q]['mask'] for q in final_ids])
        weights={}
        for si,state in enumerate(self.p['couplings']):
            for field in ('mean','scale','beta','intercept'):
                weights[f'{field}_state{si}']=np.asarray([records[q]['models'][str(state)][field] for q in final_ids])
        atomic_npz(taskdir/'frozen_masks_readouts.npz',ids=np.asarray(final_ids),masks=final_masks,**weights)
        final_scores={}; cross_scores={}; predictions={}
        for si,state in enumerate(self.p['couplings']):
            all_pred=[]; cross_pred=[]; test_y=None
            other_state=self.p['couplings'][1-si]
            for offset in range(0,len(final_ids),self.p['mask_batch']):
                ids=final_ids[offset:offset+self.p['mask_batch']]; masks=np.stack([records[q]['mask'] for q in ids])
                parts=[self.features(taskdir,graph_id,task,A,projection,masks,state,'final_test',b)
                       for b in range(self.p['test_blocks'])]
                feat=np.concatenate([x for x,y in parts],axis=1); test_y=np.concatenate([y for x,y in parts])
                all_pred.extend([predict(feat[j],records[q]['models'][str(state)]) for j,q in enumerate(ids)])
                cross_pred.extend([predict(feat[j],records[q]['models'][str(other_state)]) for j,q in enumerate(ids)])
            pred=np.stack(all_pred)
            predictions[f'pred_state{si}']=pred; predictions[f'y_state{si}']=test_y
            predictions[f'crossreadout_pred_state{si}']=np.stack(cross_pred)
            final_scores[str(state)]={name:score(test_y,pred[final_ids.index(ident)],task) for name,ident in chosen.items()}
            cross_scores[str(state)]={name:score(test_y,cross_pred[final_ids.index(ident)],task) for name,ident in chosen.items()}
        atomic_npz(taskdir/'final_predictions.npz',ids=np.asarray(final_ids),**predictions)
        # Transparent task controls with explicit access to complete input history.
        train_parts=[task_data(task,'search_train',b,trials=self.p['trial_batch'],amplitude=self.p['amplitude'],version=self.p['task_version'])
                     for b in range(self.p['train_blocks'])]
        test_parts=[task_data(task,'final_test',b,trials=self.p['trial_batch'],amplitude=self.p['amplitude'],version=self.p['task_version'])
                    for b in range(self.p['test_blocks'])]
        tx=np.concatenate([x for x,y in train_parts]); ty=np.concatenate([y for x,y in train_parts])
        vx=np.concatenate([x for x,y in test_parts]); vy=np.concatenate([y for x,y in test_parts])
        input_model=fit_readout(tx.reshape(len(tx),-1),ty,alpha=self.p['ridge_alpha'])
        input_pred=predict(vx.reshape(len(vx),-1),input_model)
        constant=np.full_like(vy,ty.mean())
        atomic_npz(taskdir/'input_history_controls.npz',y=vy,linear_input_history_prediction=input_pred,constant_prediction=constant,
            **input_model)
        atomic_json(taskdir/'final_scores.json',dict(protocol_hash=self.hash,graph=graph_id,task=task,chosen=chosen,scores=final_scores,
            secondary_readout_trained_in_other_state_scores=cross_scores,
            input_history_controls=dict(linear_full_input_history=score(vy,input_pred,task),constant=score(vy,constant,task)),
            n_final_trials=self.p['trial_batch']*self.p['test_blocks'],replication_unit='graph'))
        atomic_json(taskdir/'complete.json',dict(protocol_hash=self.hash,utc=dtm.datetime.now(dtm.timezone.utc).isoformat()))
        self.log(event='graph_task_complete',graph=graph_id,task=task)
