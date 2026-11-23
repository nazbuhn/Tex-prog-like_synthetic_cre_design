"""Local orchestration tests; native CREsted/Torch checks run on Betty."""
import sys
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from core import objective, best_step, ratio, qc, distinct, freeze_shortlist
from design_core import onehot, dinuc_shuffle, insert
from io_helpers import table, dump, checkpoint, read, reuse
from native_design import ContextAdapter, evolve


class ToyScorer:
    names = ['TARGET_C1', 'OFF_A', 'OFF_B']; target = 0
    batch_size = 32
    model = SimpleNamespace(to=lambda device: None)
    def predict(self, tokens):
        middle = np.asarray(tokens)[:, 924:1124]
        g = (middle == 2).mean(axis=1)
        a = (middle == 0).mean(axis=1)
        # Flank dependence ensures insertion/context mapping is exercised.
        context = np.asarray(tokens)[:, 0] * .02
        return np.stack([1+g+context, .25+.1*a+context, .22+.1*a], axis=-1).astype(np.float32)


class BatchPositionToyScorer(ToyScorer):
    """Controlled numerical perturbation; not a model of actual GPU error."""
    def predict(self, tokens):
        result = super().predict(tokens)
        result[:, 0] += (np.arange(len(tokens)) % self.batch_size)*getattr(self, 'position_scale', 3e-6)
        return result


class ToyOptimizer:
    def __init__(self, optimize_func):
        self.optimize_func = optimize_func
    def get_best(self, **kwargs):
        return self.optimize_func(**kwargs)


def exhaustive_api_double(n_mutations, target, model, n_sequences, starting_sequences,
                          return_intermediate, enhancer_optimizer):
    """Contract double only; does not claim to test installed CREsted."""
    from core import encode
    from design_core import dna
    seq = encode(starting_sequences[0]); changes = [(-1, 'N')]
    predictions = [model.predict(onehot(seq[None]))[0]]
    for _ in range(n_mutations):
        variants = []
        for pos in range(200):
            for base in range(4):
                if base != seq[pos]:
                    mutant = seq.copy(); mutant[pos] = base; variants.append(mutant)
        variants = np.array(variants)
        p = model.predict(onehot(variants))
        best = enhancer_optimizer.get_best(mutated_predictions=p, original_prediction=predictions[-1], target=target)
        prev = seq.copy(); seq = variants[best]
        pos = int(np.flatnonzero(prev != seq)[0]); changes.append((pos, 'ACGT'[seq[pos]])); predictions.append(p[best])
    return [dict(changes=changes, predictions=predictions)], [dna(seq)]


class Tests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(23)
        self.contexts = self.rng.integers(0, 4, (14, 2048), dtype=np.uint8)
        self.initial = self.rng.integers(0, 4, 200, dtype=np.uint8)
    def test_insert_exact_flanks(self):
        out = insert(self.contexts, self.initial)
        np.testing.assert_array_equal(out[:, :924], self.contexts[:, :924])
        np.testing.assert_array_equal(out[:, 1124:], self.contexts[:, 1124:])
        np.testing.assert_array_equal(out[:, 924:1124], np.tile(self.initial, (14, 1)))
    def test_adapter_matches_direct_predictions(self):
        scorer = ToyScorer(); seq = np.array([self.initial, self.initial[::-1]])
        p = ContextAdapter(scorer, self.contexts[:2]).predict(onehot(seq)).reshape(2, 2, 3)
        np.testing.assert_array_equal(p.reshape(-1, 3), scorer.predict(insert(self.contexts[:2], seq)))
    def test_native_contract_exhaustive_and_history(self):
        with patch('native_design.native_api', return_value=(exhaustive_api_double, ToyOptimizer)):
            result = evolve(ToyScorer(), self.contexts[:2], self.initial, np.array([.3, .3]), steps=3)
        self.assertEqual(sum(result['mutation_batches'] == 600), 3)
        self.assertTrue(np.all((result['trajectory'][1:] != result['trajectory'][:-1]).sum(axis=1) == 1))
        self.assertTrue(result['feasible'])
        self.assertEqual(result['predictions'].shape, (4, 2, 3))
    def test_best_step_keeps_earlier_feasible(self):
        p = np.array([[[3, 1]], [[2.5, .5]], [[.1, .001]]])
        step, feasible = best_step(p, 0, np.array([1.]))
        self.assertTrue(feasible); self.assertEqual(step, 1)
    def test_native_selected_sequence_mismatch_rejected(self):
        def corrupt_history(**kwargs):
            info, end = exhaustive_api_double(**kwargs)
            info[0]['predictions'][1] = info[0]['predictions'][1]+1
            return info, end
        with patch('native_design.native_api', return_value=(corrupt_history, ToyOptimizer)):
            with self.assertRaisesRegex(RuntimeError, 'score/history mapping'):
                evolve(ToyScorer(), self.contexts[:2], self.initial, np.array([.3, .3]), steps=3)
    def test_native_repeated_final_callback_supported(self):
        def repeat_final(**kwargs):
            optimizer = kwargs['enhancer_optimizer']; calls = []
            original = optimizer.get_best
            def tracked(**kw):
                calls.append(kw); return original(**kw)
            optimizer.get_best = tracked
            output = exhaustive_api_double(**kwargs)
            original(**calls[-1])
            return output
        with patch('native_design.native_api', return_value=(repeat_final, ToyOptimizer)):
            result = evolve(ToyScorer(), self.contexts[:2], self.initial, np.array([.3, .3]), steps=3, capture_replay=True)
        self.assertEqual(result['replay_frames'].shape, (3, 2, 32, 2048))
    def test_preflight_distinguishes_batch_position_from_mapping_error(self):
        from workflow import preflight
        scorer = BatchPositionToyScorer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'private').mkdir(); source = root/'source'
            model = source/'models/atac_C1'; model.mkdir(parents=True)
            tokens = self.contexts[:4]
            np.save(source/'tokens2048.npy', tokens)
            np.savez_compressed(model/'PREDICTIONS.npz', validation_ids=np.arange(4), validation_predictions=scorer.predict(tokens))
            checkpoint(model, 'COMPLETE', [model/'PREDICTIONS.npz'])
            starts = self.rng.integers(0, 4, (512, 200), dtype=np.uint8)
            np.savez_compressed(root/'private/INPUTS.npz', initial=starts, context_tokens=self.contexts)
            settings = dict(source=str(source), steps=30, designs=1024, isolated_replay_rtol=.003, isolated_replay_atol=.0003)
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda i: 'LOCAL CONTRACT DOUBLE'))
            api = (exhaustive_api_double, ToyOptimizer)
            with patch('native_design.native_api', return_value=api), patch('workflow.native_api', return_value=api), \
                 patch.dict(sys.modules, {'torch': fake_torch}), patch('workflow.metadata.version', return_value='LOCAL_TEST_DOUBLE'):
                preflight(root, settings, scorer)
            report = read(root/'REPLAY_DIAGNOSTICS.json')
            self.assertEqual(report['status'], 'PASS')
            self.assertTrue(report['same_batch_replay_pass'])
            self.assertFalse(report['original_strict_isolated_check_pass'])
            self.assertGreater(report['isolated_max_absolute_difference'], report['same_batch_max_absolute_difference'])
            self.assertTrue(reuse(root, 'NATIVE_COMPLETE'))
            # A large discrepancy must still block the library and leave diagnostics.
            (root/'NATIVE_COMPLETE.json').unlink()
            scorer.position_scale = .03
            np.savez_compressed(model/'PREDICTIONS.npz', validation_ids=np.arange(4), validation_predictions=scorer.predict(tokens))
            checkpoint(model, 'COMPLETE', [model/'PREDICTIONS.npz'])
            with patch('native_design.native_api', return_value=api), patch('workflow.native_api', return_value=api), \
                 patch.dict(sys.modules, {'torch': fake_torch}), patch('workflow.metadata.version', return_value='LOCAL_TEST_DOUBLE'):
                with self.assertRaisesRegex(RuntimeError, 'Replay diagnostic failed'):
                    preflight(root, settings, scorer)
            self.assertEqual(read(root/'REPLAY_DIAGNOSTICS.json')['status'], 'FAIL')
            self.assertFalse((root/'NATIVE_COMPLETE.json').exists())
    def test_prepared_inputs_reuse_is_authenticated(self):
        from prepare_library import main
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'new'; previous = Path(td)/'old'
            root.mkdir(); previous.mkdir(); (previous/'private').mkdir()
            payload = previous/'private/INPUTS.npz'; payload.write_bytes(b'authenticated test fixture')
            checkpoint(previous, 'PREPARE_COMPLETE', [payload])
            hashes = read(previous/'PREPARE_COMPLETE.json')['sha256']
            dump(previous/'SETTINGS.json', dict(run=str(previous), seed=2))
            settings = dict(run=str(root), seed=2, pilot_run='unused', source='unused',
                prepared_source_run=str(previous), expected_prepared_hashes=hashes, designs=1024)
            dump(root/'SETTINGS.json', settings); main(root)
            self.assertEqual((root/'private/INPUTS.npz').read_bytes(), payload.read_bytes())
            self.assertTrue(reuse(root, 'PREPARE_COMPLETE'))
    def test_no_feasible_is_explicit(self):
        _, feasible = best_step(np.array([[[.1, .001]], [[.3, .01]]]), 0, np.array([1.]))
        self.assertFalse(feasible)
    def test_objective_penalizes_all_off(self):
        p = np.array([[[2, 1]], [[.01, 0]]])
        self.assertGreater(objective(p, 0, np.array([1.]))[0], objective(p, 0, np.array([1.]))[1])
    def test_max_competitor_not_mean(self):
        p = np.array([[[2., 1., 10., 0.]]])
        self.assertAlmostEqual(float(ratio(p, 0)[0, 0]), .2)
    def test_exact_dinucleotide_counts(self):
        out = dinuc_shuffle(self.initial, self.rng)
        def count(seq):
            return np.bincount(seq[:-1]*4+seq[1:], minlength=16)
        np.testing.assert_array_equal(count(out), count(self.initial))
        self.assertEqual(out[0], self.initial[0]); self.assertEqual(out[-1], self.initial[-1])
    def test_reverse_complement_diversity(self):
        self.assertFalse(distinct(self.initial, 3-self.initial[::-1]))
        self.assertTrue(distinct(self.initial, self.rng.integers(0, 4, 200, dtype=np.uint8)))
    def test_qc_homopolymer(self):
        self.assertFalse(qc(np.zeros(200, np.uint8))['sequence_QC_pass'])
    def test_shortlist_parent_caps_and_duplicates(self):
        seqs = self.rng.integers(0, 4, (8, 200), dtype=np.uint8); seqs[1] = seqs[0]
        records = [dict(index=i, design_id=str(i), parent_id='parent', selection_eligible=True, selection_score=10-i) for i in range(8)]
        chosen = freeze_shortlist(records, seqs)
        self.assertEqual(len(chosen), 4); self.assertNotIn(1, chosen)
    def test_receipt_detects_changed_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'x').write_text('original'); checkpoint(root, 'TEST', [root/'x'])
            self.assertTrue(reuse(root, 'TEST')); (root/'x').write_text('changed')
            with self.assertRaises(RuntimeError):
                reuse(root, 'TEST')
    def test_selection_cannot_read_audit_values(self):
        from evaluate import select_before_audit
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Deliberately poisonous final-audit values. Selection must use only2:6.
            p = np.full((1, 19, 14, 3), np.nan)
            p[:, :, 2:6] = [.2, .4, .3]
            p[:, 1, 2:6] = [2., .4, .3]
            chosen = select_before_audit(root,
                [dict(design_id='d', parent_id='p', arm='natural', retention_fraction=.5)],
                self.initial[None], p, 0, np.ones(14),
                [dict(distinct_shuffles='16', identical_to_design='0')],
                dict(optimization_contexts=2, selection_contexts=4, shortlist_maximum=1))
            self.assertEqual(chosen, [0])
    def test_submission_resources_dependencies_and_idempotency(self):
        from submit import submit_one
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            settings = dict(designs=1024, designs_per_shard=32, max_concurrent_GPUs=2)
            fake = SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
            with patch('submit.subprocess.run', return_value=fake) as run:
                job = submit_one(root, settings, 'design', dependency='afterok:111', array=True)
                again = submit_one(root, settings, 'design', dependency='afterok:111', array=True)
                self.assertEqual(job, again); self.assertEqual(run.call_count, 1)
                args = run.call_args.args[0]
                self.assertIn('--array=0-31%2', args)
                self.assertIn('--dependency=afterok:111', args)
                self.assertIn('--kill-on-invalid-dep=yes', args)
                self.assertIn('--partition=dgx-b200', args)
    def test_uncertain_submission_prevents_duplicate(self):
        from submit import submit_one
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); dump(root/'SUBMISSION_design.json', dict(status='started_no_ID'))
            with patch('submit.subprocess.run') as run:
                with self.assertRaises(RuntimeError):
                    submit_one(root, {}, 'design')
                run.assert_not_called()
    def test_full_toy_evaluation_and_archives(self):
        from evaluate import main
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'private').mkdir()
            settings = dict(designs=8, audit_shuffles=16, optimization_contexts=2,
                            selection_contexts=4, audit_contexts=8, shortlist_maximum=8, seed=2, steps=2, target_index=0)
            dump(root/'SETTINGS.json', settings); (root/'METHODS.txt').write_text('Toy test')
            plan = []; initials = []; natural = []
            for i in range(8):
                name = f'C1L_{i+1:04d}'; folder = root/'private'/name; folder.mkdir()
                initial = self.rng.integers(0, 4, 200, dtype=np.uint8); final = initial.copy()
                final[np.flatnonzero(final != 2)[:2]] = 2
                trajectory = np.array([initial, initial, final])
                from workflow import predict
                predictions = predict(ToyScorer(), self.contexts[:2], trajectory)
                np.savez_compressed(folder/'RESULT.npz', trajectory=trajectory, selected_step=np.array(2),
                                    predictions=predictions, floor=np.ones(2)*.3, target_index=np.array(0))
                dump(folder/'DESIGN_SUMMARY.json', dict(design_id=name, selected_step=2, selected_feasible=True))
                checkpoint(folder, 'DESIGN_COMPLETE', [folder/'RESULT.npz', folder/'DESIGN_SUMMARY.json'])
                plan.append(dict(index=i, design_id=name, parent_id=f'p{i}', arm='natural' if i < 4 else 'random', retention_fraction=.5))
                initials.append(initial); natural.append(final)
            table(root/'DESIGN_PLAN.tsv', plan)
            table(root/'CONTEXTS.tsv', [dict(context_id=f'fresh_{i+1:02d}') for i in range(14)])
            np.savez_compressed(root/'private/INPUTS.npz', context_tokens=self.contexts, initial=np.array(initials))
            np.save(root/'private/REFERENCE_PRIMARY.npy', np.ones(14))
            fake_keras = SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None))
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
            with patch.dict(sys.modules, {'keras': fake_keras, 'torch': fake_torch}), patch('scorers.secondary', return_value=ToyScorer()):
                main(root, settings, ToyScorer())
            summary = read(root/'SUMMARY.json')
            self.assertEqual(summary['designs'], 8); self.assertFalse(summary['known_motifs_used'])
            self.assertTrue((root/'SHARE_C1_NATIVE_LIBRARY1024_METRICS.zip').exists())
            self.assertTrue((root/'PRIVATE_C1_NATIVE_LIBRARY1024_DESIGNS.zip').exists())
            self.assertTrue(reuse(root, 'EVALUATION_COMPLETE'))
            if os.environ.get('C1_LIBRARY_QA_PATH'):
                import shutil
                shutil.copy2(root/'LIBRARY_OVERVIEW.png', os.environ['C1_LIBRARY_QA_PATH'])
            import zipfile
            with zipfile.ZipFile(root/'SHARE_C1_NATIVE_LIBRARY1024_METRICS.zip') as z:
                self.assertFalse(any('SEQUENCES' in n or n.endswith('.fa') for n in z.namelist()))


if __name__ == '__main__':
    unittest.main()
