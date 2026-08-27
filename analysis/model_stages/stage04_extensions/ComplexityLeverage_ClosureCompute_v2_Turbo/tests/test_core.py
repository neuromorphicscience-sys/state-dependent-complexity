import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import jaccard_distance, exact_signflip_p
from allen_lag_aware_refit import fit_select_lag

def main():
    assert abs(jaccard_distance({1,2},{2,3})-(2/3))<1e-12
    assert exact_signflip_p([1,1,1],"greater") <= .25
    rng=np.random.default_rng(1);T=700;n=8;X=np.zeros((T,n),np.float32)
    for t in range(1,T): X[t]=.8*X[t-1]+.25*rng.normal(size=n)
    X-=X.min(0)
    fit,diag=fit_select_lag(X,.25,[.25,.5,1,2],[.1,1,10],6,3)
    assert fit is not None and diag['status']=='ok' and diag['chosen_lag_seconds'] in [.25,.5,1,2]
    print('SELF_TEST_OK',diag['chosen_lag_seconds'],diag['test_r2'])
if __name__=='__main__': main()
