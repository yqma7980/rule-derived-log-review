"""Portable wrapper tests. Real scientific fitting is prohibited."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('liuji_entry',ROOT/'workflows/liuji/run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class LiujiEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='liuji paths with spaces ')
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.fit_guard = patch('sklearn.ensemble.RandomForestClassifier.fit',side_effect=AssertionError('NO TRAINING'))
        self.fit_guard.start()
        self.addCleanup(self.fit_guard.stop)
        self.config = dict(measurement_csv='measurements.csv',groups_csv='groups.csv',output_dir='new output',
             protocol=str(r.HERE/'config/paper_protocol.json'),python_main=sys.executable,
             python_matched=sys.executable,stage_timeout_seconds=60)
        common,_,_ = r.modules(self.home/'scratch')
        self.frame = pd.DataFrame([{'well_name':f'SYNTHETIC_{w}', 'depth':i*.05,'RT':10.,'GR':50.,
             'DEN':2.4,'Subfacies':common.FACIES[2]['cn']} for w in range(3) for i in range(24)])
        self.frame.to_csv(self.home/'measurements.csv',index=False)
        pd.DataFrame({'well_id':[f'SYNTHETIC_{w}' for w in range(3)],'spatial_block':[1,2,3]}).to_csv(self.home/'groups.csv',index=False)
        self.path = self.home/'config.json'
        r.save(self.path,self.config)

    def cfg(self):
        return r.configuration(self.path)

    def test_paths_resolve_relative_to_config_not_cwd(self):
        self.assertEqual(self.cfg()['measurement_csv'],self.home/'measurements.csv')
        self.assertEqual(self.cfg()['output_dir'],self.home/'new output')

    def test_plan_uses_distinct_interpreters_and_all_six_stages(self):
        cfg = self.cfg()
        cfg['python_matched'] = 'MATCHED_INTERPRETER'
        plan = r.plan(cfg)
        self.assertEqual([x['stage'] for x in plan],list(r.STAGES))
        self.assertEqual(plan[-1]['command'][0],'MATCHED_INTERPRETER')
        self.assertTrue(all(x['command'][0]==sys.executable for x in plan[:-1]))
        self.assertFalse(cfg['output_dir'].exists())

    def test_missing_input_fails_before_output_creation(self):
        self.config['measurement_csv'] = 'missing.csv'
        r.save(self.path,self.config)
        with self.assertRaises(FileNotFoundError): self.cfg()
        self.assertFalse((self.home/'new output').exists())

    def test_unknown_config_key_rejected(self):
        self.config['historical_predictions'] = 'old.npz'
        r.save(self.path,self.config)
        with self.assertRaises(ValueError): self.cfg()

    def test_modified_protocol_rejected(self):
        protocol = r.read(r.HERE/'config/paper_protocol.json')
        protocol['seed'] = 99
        r.save(self.home/'protocol.json',protocol)
        self.config['protocol'] = 'protocol.json'
        r.save(self.path,self.config)
        with self.assertRaises(ValueError): self.cfg()

    def test_source_output_and_input_overlap_rejected(self):
        for output in [str(ROOT/'test'),str(ROOT.parent),str(self.home)]:
            self.config['output_dir'] = output
            r.save(self.path,self.config)
            with self.assertRaises(ValueError): self.cfg()

    def test_prepare_is_fresh_and_stages_no_old_npz(self):
        cfg = self.cfg()
        r.prepare(cfg)
        self.assertEqual(list(cfg['output_dir'].rglob('*.npz')),[])
        self.assertEqual(r.sha(cfg['output_dir']/'runtime/IJCG投稿/data/measurement_data.csv'),r.sha(cfg['measurement_csv']))
        self.assertEqual(r.read(cfg['output_dir']/'checkpoint.json'),r.file_inventory(cfg['output_dir']/'runtime'))
        with self.assertRaises(FileExistsError): r.prepare(cfg)

    def test_generated_fixture_qc_features_and_folds(self):
        _,matrix,cols,report = r.inspect_data(self.cfg(),enforce_counts=False)
        self.assertEqual(matrix.shape[0],72)
        self.assertEqual(len(cols),26)
        self.assertEqual(len(report['folds']),6)
        with self.assertRaises(ValueError): r.inspect_data(self.cfg())

    def test_missing_and_duplicate_groups_rejected(self):
        p = self.home/'groups.csv'
        groups = pd.read_csv(p)
        groups.iloc[:2].to_csv(p,index=False)
        with self.assertRaises(ValueError): r.inspect_data(self.cfg(),False)
        pd.concat([groups,groups.iloc[:1]]).to_csv(p,index=False)
        with self.assertRaises(ValueError): r.inspect_data(self.cfg(),False)

    def test_environment_mismatch_is_not_silently_accepted(self):
        with patch.object(r.platform,'python_version',return_value='0.0.0'):
            with self.assertRaises(RuntimeError): r.environment('main')

    def test_all_bundled_module_imports_without_cad_or_author_paths(self):
        out = self.home/'imports'
        out.mkdir()
        r.modules(out)
        for name in ['run_nrr_stage1','run_nrr_stage2','retained_metrics']:
            r.importlib.import_module(name)
        for name,rel in [('cal','originals/s6/build_major_revision_evidence.py'),
                         ('phase','originals/s6/build_submission_readiness_evidence.py'),
                         ('triage','originals/s12/scripts/run_analysis.py'),('paired','originals/s17/run_s17.py')]:
            r.load('test_'+name,r.SOURCE/rel)
        with patch.dict(os.environ,{'M001_CONTROLLED_ROOT':str(out),'M001_RECOVERED_S12':str(r.SOURCE/'originals/s12'),
                 'M001_DIAG_HELPER':str(r.SOURCE/'adapters/_diag_mechanism_intervals.py'),'M001_RECOVERY_OUTPUT':str(out/'matched')}):
            r.load('test_matched',r.SOURCE/'adapters/_run_control_export_fixed.py')

    def test_dispatch_calls_original_functions_with_new_paths(self):
        cfg = self.cfg()
        r.prepare(cfg)
        out = cfg['output_dir']
        data,matrix,cols,_ = r.inspect_data(cfg,False)
        fake = MagicMock()
        fake.SCHEMES = ('one','two')
        fake.fit_predict_oof.return_value = (None,None,[{'cache_reused':False}])
        fake.run_primary_calibration.return_value = [pd.DataFrame({'calibration_target_met':[True]})]+[pd.DataFrame({'x':[1]}) for _ in range(4)]
        fake.run_phase_complete_downsampling.return_value = [pd.DataFrame({'x':[1]}) for _ in range(3)]
        with patch.object(r,'modules',return_value=(None,SimpleNamespace(SCHEMES=('one','two')),None)), \
             patch.object(r,'load',return_value=fake),patch.object(r.importlib,'import_module',return_value=fake):
            for stage in r.STAGES:
                r.execute_stage(stage,out,data,matrix,cols)
        self.assertEqual(fake.fit_predict_oof.call_count,2)
        fake.run_primary_calibration.assert_called_once_with(matrix,cols)
        fake.run_phase_complete_downsampling.assert_called_once()
        fake.triage.assert_called_once_with(data)
        self.assertEqual(fake.main.call_count,2)
        self.assertTrue((out/'runtime/GAS投稿/stage12_substantive_revision_20260907/results/calibration_fold_details.csv').exists())

    def test_worker_rejects_foreign_cache_before_training(self):
        cfg = self.cfg()
        r.prepare(cfg)
        (cfg['output_dir']/'runtime/old.npz').write_bytes(b'FOREIGN')
        with self.assertRaisesRegex(RuntimeError,'Runtime modified'): r.worker(cfg,'class')

    def test_worker_rejects_source_mutation(self):
        cfg = self.cfg()
        r.prepare(cfg)
        with patch.object(r,'source_identity',return_value={'changed':True}):
            with self.assertRaisesRegex(RuntimeError,'source changed'): r.worker(cfg,'class')

    def test_worker_rejects_out_of_order_stage(self):
        cfg = self.cfg()
        r.prepare(cfg)
        with self.assertRaisesRegex(RuntimeError,'Prior stage'): r.worker(cfg,'matched')

    def test_launch_passes_arg_lists_and_does_not_compare_historical_data(self):
        cfg = self.cfg()
        with patch.object(r.subprocess,'run') as invoke:
            r.launch(cfg)
        self.assertEqual(invoke.call_count,8)
        self.assertTrue(all(isinstance(c.args[0],list) for c in invoke.call_args_list))
        self.assertEqual(r.read(cfg['output_dir']/'COMPLETE.json')['historical_comparison'],'NOT_RUN')

    def test_npz_comparison_uses_values_not_compressed_bytes(self):
        spec = importlib.util.spec_from_file_location('compare_npz',ROOT/'tools/compare_arrays.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        a,b = self.home/'a.npz',self.home/'b.npz'
        np.savez(a,x=np.array([1.,np.nan]))
        np.savez_compressed(b,x=np.array([1.,np.nan]))
        self.assertTrue(m.compare(a,b)['equal'])
        np.savez(b,x=np.array([2.,np.nan]))
        self.assertEqual(m.compare(a,b)['arrays']['x']['different_elements'],1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
