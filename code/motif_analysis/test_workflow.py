"""Offline tests of identities, cohort coverage, restart and scheduler barriers.

Native CREsted/Torch tests run separately on Betty before GPU submission.
"""
import copy
import ast
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import workflow as w
import control


class WorkflowTests(unittest.TestCase):
    def make_discovery_fixture(self, root, count):
        w.dump(root/'MODISCO_TASKS.json',[dict(method='integrated_grad',label='test')])
        dest=root/'modisco/integrated_grad/test_modisco_results.h5'
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b'fixture-discovery-content')
        w.receipt(dest.with_suffix('.json'),[dest],parameters=dict(report=False),
                  pattern_counts=dict(pos_patterns=count,neg_patterns=0))
        meme=root/'reference.meme';meme.write_text('known reference fixture')
        w.dump(root/'DATABASE.json',dict(meme=str(meme),meme_sha256=w.sha(meme)))
        return dest

    def test_empty_discovery_is_reported_without_fabricated_motifs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dest=self.make_discovery_fixture(root,0)
            with patch.object(w,'native_api',return_value=None):w.report_task(root,0)
            done=root/'reports/integrated_grad/test.json'
            self.assertTrue(w.valid(done))
            self.assertEqual(w.read(done)['status'],'no_patterns_to_report')

    def test_native_report_uses_jaspar_and_preserves_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dest=self.make_discovery_fixture(root,1)
            def report(**args):
                self.assertTrue(args['report'])
                self.assertEqual(args['meme_db'],str(root/'reference.meme'))
                folder=dest.parent/'test_report';folder.mkdir()
                (folder/'motifs.html').write_text('<html>native fixture report</html>')
            c=SimpleNamespace(tl=SimpleNamespace(modisco=SimpleNamespace(tfmodisco=report)))
            with patch.object(w,'native_api',return_value=c):w.report_task(root,0)
            self.assertTrue(w.valid(root/'reports/integrated_grad/test.json'))
            self.assertTrue(w.valid(dest.with_suffix('.json')))

    def test_html_retry_does_not_rediscover_or_reuse_partial_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dest=self.make_discovery_fixture(root,1)
            folder=dest.parent/'test_report';folder.mkdir()
            (folder/'partial.txt').write_text('interrupted')
            before=w.sha(dest)
            def report(**args):
                self.assertFalse(folder.exists())
                self.assertEqual(w.sha(dest),before)
                folder.mkdir();(folder/'motifs.html').write_text('complete fixture')
            c=SimpleNamespace(tl=SimpleNamespace(modisco=SimpleNamespace(tfmodisco=report)))
            with patch.object(w,'native_api',return_value=c):w.report_task(root,0)
            self.assertTrue(list(dest.parent.glob('test_report.unverified_*')))
            self.assertEqual(w.sha(dest),before)

    def test_receipts_have_no_duplicate_positional_keyword_arguments(self):
        tree=ast.parse((w.PACKAGE/'workflow.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='receipt':
                self.assertFalse(any(k.arg=='outputs' for k in node.keywords),
                                 'Receipt file-list argument duplicated at line '+str(node.lineno))

    def test_paper_baselines_and_no_dinucleotide_substitution(self):
        cfg=w.read(w.PACKAGE/'CONFIG.json')
        self.assertEqual(cfg['methods'],['integrated_grad','expected_integrated_grad'])
        text=(w.PACKAGE/'workflow.py').read_text()
        self.assertIn("baseline='zeros for IG; 25 whole-input mononucleotide permutations per input for EIG'",text)
        self.assertIn('baseline_count=25',text)
        self.assertIn('integration_points=26',text)
        self.assertNotIn('dinucleotide',text)
        self.assertIn("selection='threshold', similarity_metric='pearson'",text)
        self.assertEqual(cfg['correlation_threshold'],.2)

    def test_real_gpu_preflight_precedes_large_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            w.dump(p/'CONFIG.json',{})
            def ready(path):
                return Path(path).name=='PREPARE_COMPLETE.json'
            with patch.object(w,'valid',side_effect=ready),patch.object(control,'job') as submit:
                control.advance(p)
                submit.assert_called_once_with(p,'gpucheck',gpu=True)

    def test_known_reference_scan_uses_all_196_initial_and_final_sequences(self):
        import references
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            w.dump(p/'CONFIG.json',dict(meme_bin='/unused'))
            meme=p/'ref.meme'
            meme.write_text('MEME version 4\nMOTIF MA0001.1 Test\n'
                'letter-probability matrix: alength= 4 w= 1\n.25 .25 .25 .25\n')
            w.dump(p/'DATABASE.json',dict(meme=str(meme),meme_sha256=w.sha(meme)))
            target=p/'known_JASPAR2026_fimo';target.mkdir()
            first=next(iter(w.fasta(w.PACKAGE/'inputs/SELECTED_196.fa')))
            (target/'fimo.tsv').write_text('motif_id\tsequence_name\tstart\tstop\n'
                                          f'MA0001.1\tselected__{first}\t1\t1\n')
            w.scan_known_references(p)
            lines=(p/'ALL_INITIAL_AND_SELECTED.fa').read_text().splitlines()
            self.assertEqual(len(lines),392*2)
            self.assertEqual(sum(x.startswith('>') for x in lines),392)
            frame=pd.read_csv(p/'ALL196_JASPAR2026_SITES.tsv.gz',sep='\t')
            self.assertEqual(len(frame),392)
            self.assertEqual(frame.site_count.sum(),1)
            self.assertTrue(w.valid(p/'KNOWN_REFERENCE_FIMO_COMPLETE.json'))

    def test_installer_uses_official_motif_extra(self):
        setup = (w.PACKAGE / 'SETUP_COMPUTE.sh').read_text()
        self.assertIn("'crested[motif]' 'modisco-lite==3.0.0'", setup)
        self.assertNotIn('memesuite-lite', setup)
        self.assertIn('--constraint "$PACKAGE/MODEL_CONSTRAINTS.txt"', setup)
        self.assertIn("'memelite'", inspect_native := (w.PACKAGE / 'workflow.py').read_text())
        self.assertNotIn("'memesuite-lite'", inspect_native)

    def test_missing_scanpy_does_not_break_constraints(self):
        setup = (w.PACKAGE / 'SETUP_COMPUTE.sh').read_text()
        # Execute the actual setup's version-pin block with absent analysis-only
        # packages, as observed in the Betty error. No installation occurs.
        code = setup.split('"$BASE_PY" - "$PACKAGE/MODEL_CONSTRAINTS.txt" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
        def version(name):
            if name in {'scanpy', 'anndata'}:
                raise metadata.PackageNotFoundError(name)
            return '1.2.3'
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'constraints.txt'
            with patch.object(metadata, 'version', side_effect=version), \
                 patch.object(sys, 'argv', ['-', str(target)]):
                exec(compile(code, 'setup_constraints', 'exec'), {})
            self.assertEqual(set(target.read_text().splitlines()),
                {name+'==1.2.3' for name in ['crested','torch','keras','numpy','scipy','h5py']})
        self.assertIn('logomaker, scanpy, anndata; import crested.tl.modisco', setup)
        self.assertIn("logomaker 'scanpy==1.10.3'", setup)

    def test_slurm_spool_copy_finds_persistent_package(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package = root / 'analysis'
            run = package / 'runs/example'
            run.mkdir(parents=True)
            (package / 'workflow.py').write_text('# test placeholder\n')
            (package / 'SETUP_COMPUTE.sh').write_text('set -eu\ntouch "$1/setup_was_found"\n')
            fake_python = package / '.analysis-env/bin/python'
            fake_python.parent.mkdir(parents=True)
            fake_python.write_text('#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$*" >> "'+str(package/'calls.txt')+'"\n')
            fake_python.chmod(0o755)
            spool = root / 'slurm/spool/job123/slurm_script'
            spool.parent.mkdir(parents=True)
            spool.write_bytes((w.PACKAGE / 'RUN_STAGE.sh').read_bytes())
            subprocess.run(['bash', str(spool), str(run), 'prepare'], check=True, capture_output=True)
            self.assertTrue((package / 'setup_was_found').exists())
            calls = (package / 'calls.txt').read_text().splitlines()
            self.assertEqual(len(calls), 2)
            self.assertIn(str(package / 'workflow.py'), calls[0])
            self.assertIn(str(package / 'control.py'), calls[1])

    def test_resume_before_setup_reenters_prepare(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            run = p / 'runs/example'
            run.mkdir(parents=True)
            with patch.object(control, 'P', p), patch.object(control, 'active', return_value=[]), \
                 patch.object(control, 'job') as submit, \
                 patch.object(sys, 'argv', ['control.py', 'resume', '--run', str(run)]):
                control.main()
                submit.assert_called_once_with(run, 'prepare', large_cpu=True)

    def test_actual_full_cohort(self):
        m = pd.read_csv(w.PACKAGE / 'inputs/DESIGN_METADATA.tsv', sep='\t')
        a = w.fasta(w.PACKAGE / 'inputs/INITIAL_196.fa')
        b = w.fasta(w.PACKAGE / 'inputs/SELECTED_196.fa')
        self.assertEqual(set(m.design_id), set(a))
        self.assertEqual(set(a), set(b))
        self.assertEqual(len(m), 196)
        self.assertEqual(m.arm.value_counts().to_dict(), dict(natural=128, random=68))
        self.assertEqual(m.parent_family.nunique(), 118)
        self.assertTrue(all(len(x) == 1000 for x in list(a.values()) + list(b.values())))
        cells = pd.read_csv(w.PACKAGE / 'inputs/CELL_METADATA.tsv.gz', sep='\t')
        self.assertEqual(len(cells), 111347)
        self.assertTrue(cells.Cell_ID.is_unique)

    def test_context_coordinates(self):
        contexts = np.random.default_rng(1).integers(0, 5, (2, 2048), dtype=np.uint8)
        saved = contexts.copy()
        dna = w.design_tokens(contexts, 'ACGT'*250, 1)
        np.testing.assert_array_equal(contexts, saved)
        np.testing.assert_array_equal(dna[:524], contexts[1, :524])
        np.testing.assert_array_equal(dna[1524:], contexts[1, 1524:])
        np.testing.assert_array_equal(dna[524:1524], np.tile(np.arange(4), 250))
        self.assertEqual(w.onehot(dna[None]).shape, (1, 2048, 4))
        self.assertTrue(np.all(w.onehot(dna[None])[dna[None] == 4] == 0))

    def test_hash_receipts_detect_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            data = p / 'data.txt'
            data.write_text('real result')
            w.receipt(p / 'done.json', [data])
            self.assertTrue(w.valid(p / 'done.json'))
            data.write_text('truncated')
            self.assertFalse(w.valid(p / 'done.json'))

    def test_partial_task_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            w.dump(p / 'EXPLAIN_TASKS.json', [dict(kind='attribute', method='integrated_grad',
                records=[dict(record_id='a'), dict(record_id='b')])])
            dest = p / 'maps/integrated_grad'
            dest.mkdir(parents=True)
            (dest / 'a.npz').write_bytes(b'complete')
            w.receipt(dest / 'a.json', [dest / 'a.npz'])
            self.assertFalse(w.task_done(p, 0))
            (dest / 'b.npz').write_bytes(b'complete')
            w.receipt(dest / 'b.json', [dest / 'b.npz'])
            self.assertTrue(w.task_done(p, 0))

    def test_failed_dispatch_does_not_loop_or_advance(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            w.dump(p / 'CONFIG.json', {})
            with patch.object(control, 'missing', return_value=[3, 7]), \
                 patch.object(control, 'job') as submit:
                with self.assertRaisesRegex(RuntimeError, 'No automatic retry loop'):
                    control.advance(p, 'explain')
                submit.assert_not_called()
                self.assertEqual(w.read(p / 'NEEDS_RESUME.json')['incomplete_tasks'], [3, 7])

    def test_array_batches_obey_one_gpu_and_do_not_skip_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            w.dump(p / 'CONFIG.json', dict(array_batch_size=400, gpu_array_concurrency=1))
            calls = []
            def submit(run, phase, args, **kwargs):
                calls.append((phase, args, kwargs))
                return str(100 + len(calls))
            with patch.object(control, 'job', side_effect=submit):
                control.queue_array(p, 'explain', list(range(1003)), True)
            self.assertEqual([x[0] for x in calls], ['explain']*3 + ['dispatch'])
            covered = []
            for phase, args, kwargs in calls[:-1]:
                covered.extend(w.read(args[0]))
                self.assertTrue(kwargs['array'].endswith('%1'))
            self.assertEqual(covered, list(range(1003)))
            self.assertIsNone(calls[0][2]['dependency'])
            self.assertEqual(calls[1][2]['dependency'], '101')
            self.assertEqual(calls[2][2]['dependency'], '102')
            self.assertEqual(calls[3][2]['dependency'], '103')

    def test_resume_only_queues_missing_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            w.dump(p / 'CONFIG.json', dict(methods=['integrated_grad']))
            with patch.object(w, 'valid', return_value=True), \
                 patch.object(control, 'missing', return_value=[4, 9]), \
                 patch.object(control, 'queue_array') as array:
                control.advance(p)
                array.assert_called_once_with(p, 'predict', [4, 9], True)


if __name__ == '__main__':
    unittest.main()
