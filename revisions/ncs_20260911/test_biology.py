"""Small correctness tests for processed biological inference helpers."""
import unittest
import numpy as np
from reproduce_biology import ranks,permutation,bootstrap,unique

class BiologicalInferenceTests(unittest.TestCase):
    def test_rank_ties_and_original_order(self):
        np.testing.assert_array_equal(ranks([3,1,1,2]),[4,1.5,1.5,3])
    def test_plus_one_resolution(self):
        self.assertEqual(permutation([0,0,0],1)['one_sided_plus_one_p'],.25)
    def test_null_ties_count_as_exceedance(self):
        self.assertEqual(permutation([1,0,1],1)['n_null_ge_observed'],2)
    def test_reject_duplicate_mice(self):
        with self.assertRaises(AssertionError):unique([{'subject':'a'},{'subject':'a'}],'subject',2)
    def test_bootstrap_preserves_constant(self):
        np.testing.assert_array_equal(bootstrap([.1,.1,.1],72,n=100),[np.mean([.1,.1,.1])]*2)
    def test_seed_reproducibility(self):
        np.testing.assert_array_equal(bootstrap([1,2,4,8],71,n=100),bootstrap([1,2,4,8],71,n=100))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):permutation([0,np.nan],1)
        with self.assertRaises(ValueError):ranks([0,np.inf])
    def test_negative_ranking_not_reordered(self):
        np.testing.assert_array_equal(ranks([-1,-3,-2]),[3,1,2])
if __name__=='__main__':unittest.main()
