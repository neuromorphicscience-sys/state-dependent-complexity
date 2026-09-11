"""Unit tests for the public inferential definitions; no research simulation."""
import unittest
import numpy as np
from reproduce_revision import exact,holm,seed
from ncs_r2_core.shared_objective import shared_objective,wrong_per_noise_minimum,select_index
from ncs_r2_core.search_adapter import run_search_interface
from ncs_r2_core.stats import graph_equal_weight,require_graph_level,sufficiency_label

class Definitions(unittest.TestCase):
    def test_one_sided_resolution(self):self.assertEqual(exact(np.ones(12),True),1/4096)
    def test_two_sided_resolution(self):self.assertEqual(exact(np.ones(12)),2/4096)
    def test_null(self):self.assertEqual(exact(np.zeros(12)),1.)
    def test_sign_symmetry(self):self.assertEqual(exact([1.,-2.,3.,-4.]),exact([-1.,2.,-3.,4.]))
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):exact([1.,np.nan])
    def test_holm_original_order(self):np.testing.assert_allclose(holm([.02,.001,.5]),[.04,.003,.5])
    def test_seed_roles(self):self.assertNotEqual(seed('data','final_test'),seed('data','search_train'))
    def test_minimum_before_mask_mean(self):
        scores=np.array([[1.,0.],[0.,1.]])
        self.assertEqual(scores.min(axis=1).mean(),0.)
        self.assertEqual(scores.mean(axis=0).min(),.5)
    def test_noise_mean_before_state_minimum(self):
        self.assertEqual(shared_objective([1.,0.],[0.,1.]),.5)
        self.assertEqual(wrong_per_noise_minimum([1.,0.],[0.,1.]),0.)
    def test_state_specific_selection(self):
        self.assertEqual(select_index('A_mid',[.9,.6],[.1,.6]),0)
        self.assertEqual(select_index('A_shared',[.9,.6],[.1,.6]),1)
    def test_graph_equal_weight(self):
        values,mean=graph_equal_weight({'a':[0.]*100,'b':[1.]})
        self.assertEqual(mean,.5)
        with self.assertRaises(ValueError):require_graph_level(values,['a','a'])
    def test_unresolved_not_equivalent(self):
        result=sufficiency_label([0.,.2],[-.1,.1])
        self.assertFalse(result['sufficient'])
        self.assertEqual(result['wording'],'sufficiency_not_established')
    def test_interface_budget_and_fixed_cardinality(self):
        def score(masks,state,generation):
            np.testing.assert_array_equal(masks.sum(1),3)
            return masks[:,:4].sum(1)/3
        common=dict(n=16,k=3,population=8,generations=3,elite=2,random_injection=2,optimizer_rng_seed=27,score_callback=score)
        for arm,evus in [('A_mid',24),('A_shared',48),('shared_random',48)]:
            result=run_search_interface(arm=arm,**common)
            self.assertEqual(result.candidate_evaluations,24)
            self.assertEqual(result.state_evus,evus)
            self.assertEqual(len(result.best_mask),3)
            self.assertEqual(result.evolution_used,arm!='shared_random')
if __name__=='__main__':unittest.main()
