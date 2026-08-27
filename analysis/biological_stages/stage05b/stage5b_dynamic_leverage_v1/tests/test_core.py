import numpy as np,pandas as pd,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import rank01,dynamics_leverage,normalise_trial_table

def test_rank():
    r=rank01([1,2,3]); assert np.allclose(r,[1/6,3/6,5/6])

def test_trial_pivot():
    d=pd.DataFrame({"type":["visual_stimulus_time","response_time","feedback_time","included"],
                    "data":[np.array([1,2]),np.array([1.5,2.5]),np.array([1.8,2.8]),np.array([1,1])]})
    x=normalise_trial_table(d); assert len(x)==2 and "response_time" in x

def test_dyn():
    rng=np.random.default_rng(0)
    X=np.abs(rng.normal(size=(500,12))).astype("float32")
    m={"a":np.arange(500)<250,"b":np.arange(500)>=250}
    d,diag=dynamics_leverage(X,m,min_state_bins=30)
    assert len(d)>0 and set(d.state)=={"a","b"}

if __name__=="__main__":
    test_rank();test_trial_pivot();test_dyn();print("TESTS PASS")
