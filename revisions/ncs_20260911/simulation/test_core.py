import hashlib
import json
import time
import unittest
import numpy as np
import torch
from benchmark_core import *

class CoreTests(unittest.TestCase):
    def test_frozen_final_task_version(self):
        for task in TASKS:
            x,y=task_data(task,'final_test',0,version='challenge_v3')
            xx,yy=task_data(task,'final_test',0,version='challenge_v3')
            np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy)
            self.assertFalse(x[:,52:].any())
            self.assertTrue(np.isfinite(x).all())
            self.assertEqual(x.shape,(64,60,4))
            other,_=task_data(task,'selection',0,version='challenge_v3')
            self.assertFalse(np.array_equal(x,other))

    def test_roles_and_tasks(self):
        identities=set()
        for task in TASKS:
            for role in ('development_train','search_train','search_validation','selection','final_test'):
                ident=seed('data','v1',task,role,0)
                self.assertNotIn(ident,identities); identities.add(ident)
                x,y=task_data(task,role,0)
                self.assertEqual(x.shape,(64,60,4))
                self.assertTrue(np.isfinite(x).all())
                self.assertTrue(np.isfinite(y).all())
                self.assertFalse(x[:,52:].any())
                xx,yy=task_data(task,role,0)
                np.testing.assert_array_equal(x,xx); np.testing.assert_array_equal(y,yy)
                if task!='delayed_recall':
                    self.assertEqual(int((y>0).sum()),32)

    def test_generators(self):
        for task in TASKS:
            x,y=task_data(task,'unit',0,amplitude=1.)
            if task=='delayed_recall':
                np.testing.assert_array_equal(np.sign(x[:,28,0]),np.sign(y))
            elif task=='nonlinear_integration':
                np.testing.assert_array_equal(np.sign(x[:,22,0]*x[:,36,1]),y)
            elif task=='context_recall':
                selected=np.where(x[:,16,2]>0,x[:,30,0],x[:,30,1])
                np.testing.assert_array_equal(np.sign(selected),y)
            elif task=='pulse_order':
                order=np.argmax(x[:,:,1]>0,axis=1)-np.argmax(x[:,:,0]>0,axis=1)
                np.testing.assert_array_equal(np.sign(order),y)
                np.testing.assert_allclose(x[:,:,0].sum(1),x[:,:,1].sum(1))
            elif task=='interval':
                spans=[]
                for trial in x:
                    idx=np.flatnonzero(trial[:,0]); spans.append(idx[-1]-idx[0])
                np.testing.assert_array_equal(np.where(np.array(spans)>16,1.,-1.),y)

    def test_readout(self):
        rng=np.random.default_rng(52)
        x=rng.normal(size=(100,8)); y=x[:,0]*2-x[:,1]+.2
        model=fit_readout(x,y,alpha=.001)
        self.assertGreater(score(y,predict(x,model),'delayed_recall'),.999)
        np.testing.assert_allclose(model['mean'],x.mean(0))

    @unittest.skipUnless(torch.cuda.is_available(), 'Optional CUDA numerical parity check')
    def test_frozen_parity_cuda(self):
        a,_=graph('unit-parity',n=32)
        A=topology.to_gpu_adjacency(a,'cuda')
        masks=np.zeros((4,32),bool)
        masks[1,:4]=True; masks[2,::2]=True; masks[3,:]=True
        masks=torch.as_tensor(masks,device='cuda')
        cfg=dict(CFG); cfg['duration_ms']=400.
        expected=frozen.simulate_masks(cfg,A,masks,.035,1.,314159,True)
        feat,actual=simulate_driven(cfg,A,masks,.035,1.,314159)
        _,zero=simulate_driven(cfg,A,masks,.035,1.,314159,np.zeros((1,60,4),np.float32),np.zeros((4,32),np.float32))
        differences={}
        for k,val in expected.items():
            ref=val.cpu().numpy()
            differences[k]=float(np.max(np.abs(ref-actual[k][:,0])))
            np.testing.assert_array_equal(ref,actual[k][:,0],err_msg=k)
            np.testing.assert_array_equal(ref,zero[k][:,0],err_msg=k+' zero drive')
        self.assertTrue(np.isfinite(feat).all())
        print(json.dumps(dict(passed=True,differences=differences,
                    n=32,masks=4,dt_ms=.05,duration_ms=400,source_sha256=hashlib.sha256((FROZEN/'stage4_gpu_simulator.py').read_bytes()).hexdigest(),
                    limitation='Exact same-batch-shape zero-input parity; not a claim of invariance under batch reshaping.')))

if __name__=='__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    unittest.main(verbosity=2)
