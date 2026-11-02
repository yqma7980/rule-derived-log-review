"""Generated-fixture user flows. No real archives, network access or model fits."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORCE = 'workflows/force/code/'
KGS = 'workflows/kgs/code/'
PREP = 'workflows/kgs/source_tools/'
GUARD = '''import runpy,sys,socket
from pathlib import Path
from unittest.mock import patch
script=sys.argv[1];sys.argv=sys.argv[1:];sys.path.insert(0,str(Path(script).parent))
with patch('sklearn.ensemble.RandomForestClassifier.fit',side_effect=AssertionError('NO FIT ALLOWED')), patch('socket.socket.connect',side_effect=AssertionError('NO NETWORK ALLOWED')), patch('socket.getaddrinfo',side_effect=AssertionError('NO DNS ALLOWED')):
 runpy.run_path(script,run_name='__main__')
'''


class UserFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rule flow ')
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def cli(self, script, *args, expected=0):
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', GUARD,
            str(ROOT/script), *map(str, args)], cwd=self.work, capture_output=True,
            encoding='utf-8', errors='replace', timeout=60)
        self.assertEqual(result.returncode, expected, result.stdout+'\n'+result.stderr)
        return result

    def csv_input(self, aliases=False):
        rows = [{'well': f'SYNTHETIC_{w}', 'depth': i*.152, 'GR': 50+i/100, 'DEN': 2.4}
                for w in range(3) for i in range(60)]
        data = pd.DataFrame(rows)
        data.loc[30, 'GR'] = np.nan
        if aliases:
            data = data.rename(columns={'GR': 'gamma', 'DEN': 'density', 'depth': 'feet'})
            data['feet'] /= .3048
            data['density'] *= 1000
        path = self.work/'synthetic input.csv'
        data.to_csv(path, index=False)
        return path

    def test_all_public_help_is_offline_and_read_only(self):
        scripts = [FORCE+n for n in ('prepare_workspace.py', 'acquire_force.py',
                   'run_external.py', 'verify_external.py')]
        scripts += [KGS+n for n in ('external_review_trial.py', 'verify_external_trial.py')]
        scripts += [PREP+n for n in ('scan_kgs_archive.py', 'prepare_kgs.py')]
        for script in scripts:
            with self.subTest(script=script):
                self.assertIn('usage:', self.cli(script, '--help').stdout)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_force_explicit_mode_and_download_opt_in(self):
        self.cli(FORCE+'run_external.py', expected=2)
        self.cli(FORCE+'acquire_force.py', '--workspace', self.work/'unused', expected=2)
        self.cli(FORCE+'run_external.py', '--inspect', '--workspace', self.work/'missing', expected=2)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_force_workspace_and_input_inspection_no_fit(self):
        out = self.work/'force output'
        self.cli(FORCE+'prepare_workspace.py', '--workspace', out)
        sentinel = out/'data/external/preserved.txt'
        sentinel.write_text('preserve')
        self.cli(FORCE+'prepare_workspace.py', '--workspace', out)
        self.assertEqual(sentinel.read_text(), 'preserve')
        data = pd.read_csv(self.csv_input()).rename(columns={'well':'WELL','depth':'DEPTH_MD','DEN':'RHOB'})
        data['FORCE_2020_LITHOFACIES_LITHOLOGY'] = 0
        with zipfile.ZipFile(out/'data/external/FORCE_train.zip', 'w') as archive:
            archive.writestr('synthetic.csv', data.to_csv(index=False, sep=';'))
        (out/'qa/SOURCE_DOWNLOAD.json').write_text('{}')
        self.cli(FORCE+'run_external.py', '--inspect', '--workspace', out)
        inventory = json.loads((out/'qa/DATA_INVENTORY.json').read_text())
        self.assertEqual((inventory['raw_rows'], inventory['eligible_rows'], inventory['segments']), (180,179,4))
        self.assertEqual(list((out/'data/cache').iterdir()), [])
        before = (out/'results/force_population.csv').read_bytes()
        self.cli(FORCE+'run_external.py', '--inspect', '--workspace', out, expected=2)
        self.assertEqual((out/'results/force_population.csv').read_bytes(), before)

    def test_kgs_csv_inspection_and_partial_output_protection(self):
        source = self.csv_input()
        out = self.work/'kgs inspection'
        self.cli(KGS+'external_review_trial.py', '--input', source, '--out', out,
                 '--max-wells', 0, '--inspect-only')
        inventory = json.loads((out/'inventory.json').read_text())
        self.assertEqual((inventory['selected_rows'], inventory['selected_wells'], inventory['selected_segments']), (179,3,4))
        self.assertEqual(inventory['input'], str(source.resolve()))
        self.assertFalse((out/'cache').exists())
        self.assertFalse((out/'COMPLETE.json').exists())
        before = (out/'inventory.json').read_bytes()
        self.cli(KGS+'external_review_trial.py', '--input', source, '--out', out,
                 '--inspect-only', expected=2)
        self.assertEqual((out/'inventory.json').read_bytes(), before)

    def test_kgs_explicit_aliases_and_unit_conversion(self):
        source = self.csv_input(aliases=True)
        out = self.work/'aliases'
        self.cli(KGS+'external_review_trial.py', '--input', source, '--out', out,
                 '--depth-column', 'feet', '--gr-column', 'gamma', '--den-column', 'density',
                 '--depth-to-m', .3048, '--den-to-gcc', .001, '--max-wells', 0, '--inspect-only')
        inventory = json.loads((out/'inventory.json').read_text())
        self.assertEqual(inventory['selected_rows'], 179)
        self.assertEqual(inventory['depth_to_m'], .3048)

    def test_kgs_invalid_inputs_fail_before_output(self):
        source = self.csv_input()
        cases = [('--depth-to-m','nan'), ('--den-to-gcc','0'), ('--gr-column','well'),
                 ('--n-jobs','0'), ('--gr-column','absent')]
        for i, args in enumerate(cases):
            out = self.work/f'invalid{i}'
            self.cli(KGS+'external_review_trial.py', '--input', source, '--out', out,
                     '--inspect-only', *args, expected=2)
            self.assertFalse(out.exists())
        self.cli(KGS+'external_review_trial.py', '--input', self.work/'missing.csv',
                 '--out', self.work/'unused', '--inspect-only', expected=2)
        self.assertFalse((self.work/'unused').exists())

    def test_kgs_archive_hash_rejection_and_missing_inputs(self):
        self.cli(PREP+'prepare_kgs.py', '--archive', self.work/'missing.zip',
                 '--inventory', self.work/'missing.json', '--output-dir', self.work/'out', expected=2)
        archive = self.work/'synthetic.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('empty.txt', 'SYNTHETIC')
        inventory = self.work/'inventory.json'
        inventory.write_text('[]')
        self.cli(PREP+'prepare_kgs.py', '--archive', archive, '--inventory', inventory,
                 '--output-dir', self.work/'out', expected=2)
        self.cli(PREP+'scan_kgs_archive.py', '--archive', archive,
                 '--output-dir', self.work/'scan', expected=1)
        self.assertFalse((self.work/'out').exists())
        self.assertFalse((self.work/'scan').exists())

    def test_kgs_synthetic_las_screening_with_test_only_pin_override(self):
        header = '~W\nAPI . 0000000001 : synthetic\nWELL . SYNTHETIC_ONLY : name\nNULL . -999.25 : null\n~C\nDEPT .FT : depth\nGR .API : gamma\nRHOB .G/CC : density\n~A\n'
        values = ''.join(f'{i} 50 2.4\n' for i in range(30))
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, 'w') as z:
            z.writestr('synthetic.las', header+values)
            z.writestr('wrong_units.las', header.replace('GR .API','GR .UNKNOWN')+values)
        archive = self.work/'synthetic.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('nested.zip', inner.getvalue())
        spec = importlib.util.spec_from_file_location('synthetic_scanner', ROOT/PREP/'scan_kgs_archive.py')
        scanner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scanner)
        scanner.EXPECTED_SHA256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        out = self.work/'prepared'
        summary = scanner.scan(archive, out)
        self.assertEqual(summary['candidate_las_files'], 2)
        with self.assertRaises(FileExistsError):
            scanner.scan(archive, out)
        real_sha = hashlib.sha256
        raw = archive.read_bytes()
        class PinnedFixture:
            def hexdigest(self):
                return '325059152a76c89fb1a798390fe955f6256e07e00e9a40305691122bcc940edd'
        def fixture_hash(data=b''):
            return PinnedFixture() if data == raw else real_sha(data)
        argv = [str(ROOT/PREP/'prepare_kgs.py'), '--archive', str(archive),
                '--inventory', str(out/'gr_density_inventory.json'), '--output-dir', str(out)]
        with patch.object(sys, 'argv', argv), patch('hashlib.sha256', side_effect=fixture_hash), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(argv[0], run_name='__main__')
        data = pd.read_csv(out/'kgs_selected_wells.csv')
        self.assertEqual(len(data), 30)
        self.assertAlmostEqual(data.depth.iloc[1], .3048)
        summary = json.loads((out/'input_summary.json').read_text())
        self.assertEqual(summary['selected_wells'], 1)
        self.assertEqual(summary['status_counts']['excluded_missing_or_unrecognized_input_units'], 1)
        before = (out/'kgs_selected_wells.csv').read_bytes()
        with patch.object(sys, 'argv', argv), patch('hashlib.sha256', side_effect=fixture_hash), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runpy.run_path(argv[0], run_name='__main__')
        self.assertEqual((out/'kgs_selected_wells.csv').read_bytes(), before)

    def test_temperature_scaling_positive_T_and_clipped_softmax(self):
        sys.path.insert(0, str(ROOT/FORCE/'inherited'))
        from audit_core import temperature_scale
        p = np.array([[0., .1, .2, .3, .4, 0.], [.2,.2,.2,.1,.1,.2]])
        expected = np.sqrt(np.clip(p, 1e-12, 1))
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(temperature_scale(p, 2.), expected)
        np.testing.assert_array_equal(temperature_scale(p, 2.).argmax(axis=1), p.argmax(axis=1))
        for bad in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                temperature_scale(p, bad)

    def test_scientific_helpers_unchanged_from_preserved_source(self):
        for relative in [FORCE+'run_external.py', KGS+'external_review_trial.py']:
            old = ast.parse((ROOT/'provenance/s23_original'/relative).read_text(encoding='utf-8-sig'))
            new = ast.parse((ROOT/relative).read_text(encoding='utf-8-sig'))
            funcs = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != 'main'}
            self.assertEqual(funcs(old), funcs(new))


if __name__ == '__main__':
    unittest.main(verbosity=2)
