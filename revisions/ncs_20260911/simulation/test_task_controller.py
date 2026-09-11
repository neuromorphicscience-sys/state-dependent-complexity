"""Synthetic CPU control-flow tests; never simulate scientific reservoir data."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import task_controller as controller


class SyntheticExperiment(controller.Experiment):
    """Test-only reservoir substitute. Synthetic labels deliberately drive features.

    These are not valid experimental observations and must not enter the paper.
    Real task inputs, optimizer, readout fitting and orchestration are retained.
    """
    def __init__(self, out, perturb_final=False):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.hash = 'SYNTHETIC_WORKFLOW_AUDIT_ONLY'
        self.calls = []
        self.perturb_final = perturb_final
        self.p = dict(n_neurons=32, hh_budget=4, couplings=[.035, .05],
                      initial_candidates=8, generations=3,
                      offspring_per_generation=8, elite_count=4, mask_batch=24,
                      train_blocks=2, search_blocks=1, selection_blocks=1,
                      test_blocks=2, shortlist_size=3, amplitude=5.,
                      task_version='challenge_v3', ridge_alpha=10.,
                      trial_batch=32, matched_controls=4)

    def log(self, **record):
        pass

    def features(self, taskdir, graph_id, task, A, projection, masks,
                 state, role, block):
        self.calls.append((role, block, state))
        if role == 'final_test':
            assert (taskdir / 'selection_frozen.json').exists()
            assert (taskdir / 'frozen_masks_readouts.npz').exists()
            frozen = json.loads((taskdir / 'selection_frozen.json').read_text())
            assert frozen['final_data_consumed'] is False
        _, y = controller.task_data(task, role, block,
                                    trials=self.p['trial_batch'],
                                    version=self.p['task_version'])
        rng = np.random.default_rng(controller.seed('synthetic_features', role, block, state))
        base = rng.normal(size=(self.p['trial_batch'], 8))
        result = []
        for mask in masks:
            values = base.copy()
            ids = np.flatnonzero(mask)
            strength = .25 + ((ids.sum() * (7 if state == .035 else 11)) % 23) / 12.
            values[:, 0] += y * strength
            result.append(values)
        values = np.stack(result)
        if self.perturb_final and role == 'final_test':
            y = -y
            values = values[:, :, ::-1] * 2.7 + .4
        return values, y


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ncs-synthetic-controller-')
        cls.runs = []
        for i, change in enumerate((False, True)):
            run = SyntheticExperiment(Path(cls.temp.name) / str(i), change)
            with patch.object(controller, 'guard', return_value=0), \
                 patch.object(controller.topology, 'to_gpu_adjacency', side_effect=lambda a, device: a), \
                 patch.object(controller, 'simulate_driven', side_effect=AssertionError('GPU simulation forbidden')):
                run.run_task('synthetic-graph', 'nonlinear_integration')
            td = run.out / 'synthetic-graph' / 'nonlinear_integration'
            cls.runs.append((run, td))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def read(self, filename, index=0):
        return json.loads((self.runs[index][1] / filename).read_text())

    def test_default_constructor_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'Audit-only controller'):
            controller.Experiment({})

    def test_original_scientific_method_integrity(self):
        root = Path(controller.__file__).parent
        record = json.loads((root / 'TASK_CONTROLLER_PROVENANCE.json').read_text())
        self.assertEqual(hashlib.sha256(Path(controller.__file__).read_bytes()).hexdigest(),
                         record['published_sha256'])
        tree = ast.parse(Path(controller.__file__).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        for nodes, hashes in ((cls.body, record['unchanged_method_AST_sha256']),
                               (tree.body, record['unchanged_helper_AST_sha256'])):
            for node in nodes:
                if isinstance(node, ast.FunctionDef) and node.name in hashes:
                    actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                    self.assertEqual(actual, hashes[node.name])
        self.assertFalse(record['production_rerun_validated'])

    def test_final_data_cannot_change_search_or_selection(self):
        self.assertEqual(self.read('search_candidates.json'), self.read('search_candidates.json', 1))
        self.assertEqual(self.read('selection_frozen.json'), self.read('selection_frozen.json', 1))
        self.assertNotEqual(self.read('final_scores.json')['scores'],
                            self.read('final_scores.json', 1)['scores'])

    def test_final_data_cannot_change_frozen_readouts(self):
        with np.load(self.runs[0][1] / 'frozen_masks_readouts.npz') as left, \
             np.load(self.runs[1][1] / 'frozen_masks_readouts.npz') as right:
            self.assertEqual(left.files, right.files)
            for key in left.files:
                np.testing.assert_array_equal(left[key], right[key])

    def test_roles_and_freeze_precede_final_access(self):
        for run, _ in self.runs:
            roles = [r[0] for r in run.calls]
            self.assertLess(roles.index('search_train'), roles.index('search_validation'))
            self.assertLess(roles.index('search_validation'), roles.index('independent_selection'))
            self.assertLess(roles.index('independent_selection'), roles.index('final_test'))
            self.assertTrue(all(r == 'final_test' for r in roles[roles.index('final_test'):]))

    def test_equal_search_budgets_and_shared_initial_pool(self):
        search = self.read('search_candidates.json')
        initial = search['initial_ids']
        self.assertEqual(len(initial), 8)
        self.assertEqual(search['unique_budget_per_method'], 32)
        for ids in search['pools'].values():
            self.assertEqual(ids[:8], initial)
            self.assertEqual(len(ids), 32)
            self.assertEqual(len(set(ids)), 32)
        children = [q for ids in search['pools'].values() for q in ids[8:]]
        self.assertEqual(len(children), len(set(children)))

    def test_every_generated_mask_has_fixed_cardinality(self):
        td = self.runs[0][1]
        for filename in ('all_search_masks.npz', 'frozen_masks_readouts.npz'):
            with np.load(td / filename) as arrays:
                self.assertTrue(np.all(arrays['masks'].sum(axis=1) == 4))
        with self.assertRaisesRegex(ValueError, 'HH budget mismatch'):
            self.runs[0][0].candidate(np.ones(32, dtype=bool), 'invalid_test_only')

    def test_maximin_and_independent_selection_rule(self):
        search = self.read('search_candidates.json')
        chosen = self.read('selection_frozen.json')
        records = {r['id']: r for r in search['records']}
        for r in records.values():
            self.assertEqual(r['search_objective'], min(r['search_scores'].values()))
        for name, ids in search['pools'].items():
            shortlist = sorted(ids, key=lambda q: (-records[q]['search_objective'], q))[:3]
            self.assertEqual(shortlist, chosen['shortlists'][name])
            expected = sorted(shortlist, key=lambda q: (
                -min(chosen['independent_selection_scores'][q].values()), q))[0]
            self.assertEqual(chosen['chosen'][name], expected)
        for state, ids in chosen['specialist_shortlists'].items():
            expected = sorted(ids, key=lambda q: (
                -chosen['independent_selection_scores'][q][state], q))[0]
            self.assertEqual(chosen['chosen'][f'specialist_pool_{state}'], expected)

    def test_matched_controls_preserve_selected_input_group_counts(self):
        chosen = self.read('selection_frozen.json')
        projection = controller.input_projection('synthetic-graph', 32)
        with np.load(self.runs[0][1] / 'frozen_masks_readouts.npz') as z:
            masks = dict(zip(z['ids'].tolist(), z['masks']))
        parent = masks[chosen['chosen']['population_search']]
        counts = [int(parent[np.flatnonzero(g)].sum()) for g in projection]
        self.assertEqual(counts, chosen['group_matched_hh_counts'])
        controls = [q for name, q in chosen['chosen'].items()
                    if name.startswith('group_matched_random_')]
        self.assertEqual(len(controls), 4)
        for q in controls:
            self.assertEqual([int(masks[q][np.flatnonzero(g)].sum()) for g in projection], counts)

    def test_completed_resume_checks_protocol_identity(self):
        run, td = self.runs[0]
        before = len(run.calls)
        run.run_task('synthetic-graph', 'nonlinear_integration')
        self.assertEqual(len(run.calls), before)
        with patch.object(run, 'hash', 'different_protocol'):
            with self.assertRaisesRegex(RuntimeError, 'protocol mismatch'):
                run.run_task('synthetic-graph', 'nonlinear_integration')

    def test_cache_rejects_incorrect_metadata_without_simulation(self):
        with tempfile.TemporaryDirectory(prefix='ncs-cache-identity-') as tmp:
            run = SyntheticExperiment(Path(tmp) / 'run')
            masks = np.zeros((1, 32), dtype=bool)
            masks[:, :4] = True
            metadata = dict(protocol=run.hash, graph='g', task='interval',
                            ids=[controller.mask_id(masks[0])], state=.035,
                            role='final_test', block=0)
            cache = Path(tmp) / 'feature_cache'
            cache.mkdir()
            np.savez(cache / (controller.digest(metadata) + '.npz'),
                     metadata_hash=np.asarray('wrong'), features=np.zeros((1, 32, 8)), labels=np.zeros(32))
            with patch.object(controller, 'simulate_driven', side_effect=AssertionError('forbidden')):
                with self.assertRaisesRegex(RuntimeError, 'Cache identity mismatch'):
                    controller.Experiment.features(run, Path(tmp), 'g', 'interval',
                                                   None, None, masks, .035, 'final_test', 0)

    def test_frozen_noise_registry_has_no_collisions(self):
        identities = [controller.seed('noise', f'formal-{g:02d}', task, role, block)
                      for g in range(1, 13) for task in controller.TASKS
                      for role, count in [('search_train', 2), ('search_validation', 1),
                                          ('independent_selection', 1), ('final_test', 4)]
                      for block in range(count)]
        self.assertEqual(len(identities), 576)
        self.assertEqual(len(set(identities)), 576)


if __name__ == '__main__':
    unittest.main(verbosity=2)
