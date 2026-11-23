import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from util import dump, read, verify, checkpoint, rows
from math_core import label_panel, original_states, pooled, top_counts, count_scores, rc_flag
from prepare import aggregate
from train import summarize, capture_support
import submit


class PanelTests(unittest.TestCase):
    def test_original_states_restores_each_foxp3_parent_individually(self):
        data = [dict(training_class='Foxp3_high_TIL', gse267584_reference_state='TIL Tex-term'),
                dict(training_class='Foxp3_high_TIL', gse267584_reference_state='TIL Tex-prolif'),
                dict(training_class='Tex-prog', gse267584_reference_state='Tex-prog')]
        self.assertEqual(original_states(data), ['TIL Tex-term', 'TIL Tex-prolif', 'Tex-prog'])
        with self.assertRaises(RuntimeError):
            original_states([dict(gse267584_reference_state='Foxp3_high_TIL')])

    def test_atac_target_kept_and_small_background_strings_not_truncated(self):
        target = np.array([True, True, False, False, False, False])
        names, codes = label_panel(['A', 'B', 'C', 'C', 'D', 'E'], minimum=2, target=target)
        self.assertIn('OTHER_SMALL_ATAC', names)
        self.assertEqual((np.array(names)[codes] == 'TARGET_C1').sum(), 2)
        self.assertEqual((np.array(names)[codes] == 'OTHER_SMALL_ATAC').sum(), 2)
        self.assertEqual(len(codes), 6)

    def test_weighted_union_is_fragment_exposure_weighted_not_mean(self):
        rates = np.array([[2., 10., 30.], [3., 8., 20.]])
        exposure = np.array([1., 3., 20.])
        self.assertTrue(np.allclose(pooled(rates, exposure, [0, 1]), [8., 6.75]))

    def test_raw_aggregate_capture_and_pooled_counts_conserve_exact_cells(self):
        x = sparse.csr_matrix(np.array([[1, 2, 3, 4], [5, 0, 8, 1], [0, 2, 0, 9]], np.int32))
        pos = np.array([3, 0, 2])
        codes = np.array([0, 1, 0])
        exposure = np.array([20., 10., 30.])
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            cells, denoms = aggregate(x, pos, codes, exposure, 2, folder, np.array(['B', 'A', 'A']))
            self.assertTrue(np.array_equal(cells, [2, 1]))
            self.assertTrue(np.array_equal(denoms, [50., 10.]))
            observed = np.load(folder / 'counts.npy')
            expected = np.c_[np.asarray(x[:, [3, 2]].sum(axis=1)).ravel(), x[:, 0].toarray().ravel()]
            self.assertTrue(np.array_equal(observed, expected))
            cap = np.load(folder / 'capture_counts.npy')
            metadata = read(folder / 'CAPTURE_GROUPS.json')
            for j in range(2):
                self.assertTrue(np.array_equal(cap[:, [i for i, r in enumerate(metadata) if r['group'] == j]].sum(axis=1), observed[:, j]))

    def test_selectivity_uses_strongest_background_and_measured_counts(self):
        meta = dict(names=['A', 'B', 'C'], cells=[200, 200, 200], exposure=[1, 1, 1], panel='custom', checkpoint_targets=['A'])
        a = np.array([[10., 1., 2.], [10., 1., 8.], [10., 1., 2.]])
        n = np.array([[20, 2, 2], [20, 2, 2], [1, 2, 2]])
        r = list(count_scores(a, a, n, meta))
        self.assertTrue(np.array_equal(r[0][2], [True, False, False]))

    def test_topics_retain_overlapping_memberships_but_test_exclusive_ranking(self):
        y = np.array([[1., 0.], [1., 1.], [0., 1.], [0., 0.]])
        p = np.array([[.9, .1], [.8, .8], [.1, .9], [.1, .1]])
        meta = dict(task='topic', names=['Topic1', 'Topic2'], checkpoint_targets=['Topic1', 'Topic2'])
        rows_, scores, summary = summarize(y, p, None, meta)
        self.assertEqual(rows_[0]['positives'], 1)
        self.assertEqual(rows_[0]['membership_AP'], 1.)
        self.assertEqual(summary['macro_selectivity_AP'], 1.)
        self.assertTrue(np.array_equal(scores['Topic1'][1], [True, False, False, False]))
        none = summarize(np.zeros_like(y), p, None, meta)[2]
        self.assertIsNone(none['macro_selectivity_AP'])

    def test_ties_and_recall_arithmetic(self):
        r = top_counts(np.array([True, False, True, False]), np.ones(4), 2)
        self.assertEqual(r['hits'], 1.)
        self.assertEqual(r['precision'], .5)
        self.assertEqual(r['recall'], .5)

    def test_ambiguous_backgrounds_have_a_separate_conservative_sensitivity(self):
        meta = dict(names=['A', 'B', 'CD8', 'not classified'], cells=[200, 200, 5, 5], exposure=[1]*4,
                    panel='custom', checkpoint_targets=['A'])
        rates = np.array([[10., 1., 30., 40.], [1., 10., 1., 1.]])
        counts = np.full_like(rates, 20)
        primary = list(count_scores(rates, rates, counts, meta))[0]
        sensitivity = list(count_scores(rates, rates, counts, dict(meta, include_ambiguous_backgrounds=True)))[0]
        self.assertTrue(primary[2][0])
        self.assertFalse(sensitivity[2][0])

    def test_zero_target_capture_with_background_signal_counts_against_consistency(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            info = [dict(group=0, capture='A', cells=50, exposure=100), dict(group=1, capture='A', cells=50, exposure=100),
                    dict(group=0, capture='B', cells=50, exposure=100), dict(group=1, capture='B', cells=50, exposure=100)]
            dump(folder / 'CAPTURE_GROUPS.json', info)
            np.save(folder / 'capture_counts.npy', np.array([[30, 5, 0, 50], [2, 1, 0, 0]], np.int64))
            a, b = capture_support(folder, np.array([0, 1]), [0])
            self.assertEqual(a['informative_captures'], 2)
            self.assertEqual(a['supporting_captures'], 1)
            self.assertEqual(a['support_fraction'], .5)
            self.assertEqual(b['informative_captures'], 0)

    def test_receipts_detect_accidental_changes(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / 'a').write_text('data')
            checkpoint(folder, ['a'])
            verify(folder)
            (folder / 'a').write_text('different')
            with self.assertRaises(RuntimeError):
                verify(folder)

    def test_gpu_array_single_concurrency_and_duplicate_protection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.dict(os.environ, dict(SBATCH_GRES='gpu:old:2')):
                with patch.object(submit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '12345;betty\n', '')) as call:
                    resources = ['--partition=dgx-b200', '--gpus=1', '--array=0,1,2,3%1']
                    self.assertEqual(submit.job(root, 'train', resources, 'afterok:12344'), '12345')
                    self.assertEqual(submit.job(root, 'train', resources), '12345')
                    self.assertEqual(call.call_count, 1)
                    args = call.call_args.args[0]
                    self.assertIn('--array=0,1,2,3%1', args)
                    self.assertIn('--dependency=afterok:12344', args)
                    self.assertNotIn('SBATCH_GRES', call.call_args.kwargs['env'])

    def test_ambiguous_submission_and_active_array_prevent_duplicate_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(submit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'controller unavailable')) as call:
                with self.assertRaises(RuntimeError):
                    submit.job(root, 'train', [])
                with self.assertRaises(RuntimeError):
                    submit.job(root, 'train', [])
                self.assertEqual(call.call_count, 1)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'jobs').mkdir()
            (root / 'jobs/train_JOB_ID.txt').write_text('12345\n')
            with patch.object(submit, 'terminal', return_value=['COMPLETED', 'PENDING']), patch.object(submit.subprocess, 'run') as call:
                self.assertEqual(submit.job(root, 'train', [], resume=True), '12345')
                call.assert_not_called()

    def test_accounting_retains_pending_array_range(self):
        result = subprocess.CompletedProcess([], 0, '12345_0|COMPLETED|\n12345_[1-3]|PENDING|\n', '')
        with patch.object(submit.subprocess, 'run', return_value=result):
            self.assertEqual(submit.terminal('12345'), ['COMPLETED', 'PENDING'])

    def test_metadata_and_native_guard_structure(self):
        root = Path(__file__).parent
        metadata = rows(root / 'CELL_METADATA.tsv.gz')
        self.assertEqual(len(metadata), 111347)
        self.assertEqual(len({m['Cell_ID'] for m in metadata}), 111347)
        self.assertEqual(sum(m['ATAC_coarse'] == 'C1' for m in metadata), 4440)
        self.assertEqual(len({m['ImmGenT'] for m in metadata}), 22)
        text = (root / 'model.py').read_text()
        self.assertIn('torch.rsqrt(torch.clamp(', text)
        self.assertNotIn('nan_to_num', text)
        self.assertIn('torch.autograd.grad', text)
        training = (root / 'train.py').read_text()
        self.assertIn('error_if_nonfinite=True', training)
        self.assertIn("'SEED_TARGETS.json'", training)
        self.assertLess(training.index("dump(out / 'SEED_TARGETS.json'"), training.index("test_ids = np.flatnonzero"))
        self.assertIn("training_ids[ii]", training)
        settings = read(root / 'DEFAULTS.json')
        self.assertEqual(settings['batch_size'], 32)
        self.assertLess(settings['lr_patience'], settings['early_stopping_patience'])

    def test_reverse_complements_and_syntax(self):
        flags = [rc_flag(i, 2, 20261103) for i in range(1000)]
        self.assertTrue(300 < sum(flags) < 700)
        root = Path(__file__).parent
        for path in root.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))
        for path in root.glob('*.sh'):
            subprocess.run(['bash', '-n', str(path)], check=True)


if __name__ == '__main__':
    unittest.main()
