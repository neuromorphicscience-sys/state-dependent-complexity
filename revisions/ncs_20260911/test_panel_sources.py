"""Synthetic failure tests for metadata validation, not scientific evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from verify_panel_sources import safe_file, validate_bindings


def fixture():
    index = dict(datasets=[dict(id=str(i), file=f'tables/{i}.csv', sha256=str(i))
                          for i in range(296)], unique_datasets=296,
                 display_items=dict(main=6, extended_data=8, supplementary=6))
    rows = [dict(figure=f'F{i % 20}', panel=str(i), quantitative_microplots=2 if i < 44 else 1,
                 datasets=[dict(dataset_id=str(i), sha256=str(i), public_file=f'tables/{i}.csv')])
            for i in range(202)]
    breakdown = {f'F{i}': dict(panels=sum(r['figure'] == f'F{i}' for r in rows),
                              microplots=sum(r['quantitative_microplots'] for r in rows
                                             if r['figure'] == f'F{i}')) for i in range(20)}
    full = dict(crosswalks=['test.json'], quantitative_panels=202, quantitative_microplots=246,
                figure_count=20, breakdown=breakdown)
    return index, full, {'test.json': dict(records=rows)}


class PanelSourceTests(unittest.TestCase):
    def setUp(self):
        self.index, self.full, self.walks = fixture()
        self.rows = self.walks['test.json']['records']

    def check(self):
        return validate_bindings(self.index, self.full, self.walks)

    def test_complete(self):
        self.assertEqual(self.check()['quantitative_panels'], 202)

    def test_duplicate_panel(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate figure/panel'): self.check()

    def test_missing_dataset(self):
        self.rows[0]['datasets'][0]['dataset_id'] = 'absent'
        with self.assertRaisesRegex(ValueError, 'Unresolved dataset'): self.check()

    def test_wrong_hash(self):
        self.rows[0]['datasets'][0]['sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.check()

    def test_wrong_alias(self):
        self.rows[0]['datasets'][0]['public_file'] = 'tables/wrong.csv'
        with self.assertRaisesRegex(ValueError, 'path mismatch'): self.check()

    def test_wrong_microplot_total(self):
        self.rows[0]['quantitative_microplots'] += 1
        with self.assertRaisesRegex(ValueError, 'Microplot total'): self.check()

    def test_duplicate_dataset(self):
        self.index['datasets'].append(copy.deepcopy(self.index['datasets'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate dataset'): self.check()

    def test_unsafe_path(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ['../secret', '/absolute', 'C:/absolute', 'a\\b']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_file(Path(folder), name)

    def test_unknown_figure(self):
        self.rows[0]['figure'] = 'unlisted'
        with self.assertRaisesRegex(ValueError, 'Figure inventory'): self.check()


if __name__ == '__main__': unittest.main()
