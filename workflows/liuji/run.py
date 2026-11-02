"""Configurable main-result replay using bundled, recovered scientific functions.

Plan, environment check, and inspection never fit models. Only `run
--allow-training` starts the six-stage, isolated replay. No historical prediction
or temperature file is an input to normal generation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE/'source'
STAGES = ('class', 'calibration', 'phases', 'triage', 'paired', 'matched')
REQUIRED = {
    'main': {'python': '3.12.10', 'numpy': '2.4.3', 'pandas': '2.3.3',
             'scipy': '1.17.1', 'scikit-learn': '1.8.0', 'joblib': '1.5.3', 'matplotlib': '3.10.8'},
    'matched': {'python': '3.13.9', 'numpy': '2.3.5', 'pandas': '2.3.3',
                'scipy': '1.16.3', 'scikit-learn': '1.7.2', 'joblib': '1.5.2', 'matplotlib': '3.10.6'},
}
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Nonempty path required')
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base/p).resolve()


def interpreter(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Interpreter must be a path or executable name')
    if '/' in value or '\\' in value or Path(value).is_absolute():
        p = resolve_path(value, base)
        if not p.is_file():
            raise FileNotFoundError(p)
        return str(p)
    found = shutil.which(value)
    if not found:
        raise FileNotFoundError('Interpreter not found: '+value)
    return str(Path(found).resolve())


def configuration(path):
    path = Path(path).resolve()
    cfg = read(path)
    keys = {'measurement_csv','groups_csv','output_dir','protocol','python_main','python_matched','stage_timeout_seconds'}
    if set(cfg) != keys:
        raise ValueError('Config keys must be: '+', '.join(sorted(keys)))
    for key in ['measurement_csv','groups_csv','output_dir','protocol']:
        cfg[key] = resolve_path(cfg[key], path.parent)
    for key in ['python_main','python_matched']:
        cfg[key] = interpreter(cfg[key], path.parent)
    if read(cfg['protocol']) != read(HERE/'config/paper_protocol.json'):
        raise ValueError('Protocol differs from the accepted paper protocol; do not silently retune replay')
    if not isinstance(cfg['stage_timeout_seconds'], int) or cfg['stage_timeout_seconds'] < 1:
        raise ValueError('stage_timeout_seconds must be a positive integer')
    out = cfg['output_dir']
    if out == ROOT or out.is_relative_to(ROOT) or ROOT.is_relative_to(out):
        raise ValueError('Output must be outside and not an ancestor of the code package')
    for key in ['measurement_csv','groups_csv']:
        p = cfg[key]
        if not p.is_file():
            raise FileNotFoundError(p)
        if p.is_relative_to(out):
            raise ValueError('Input must not be inside output directory')
    if cfg['measurement_csv'] == cfg['groups_csv']:
        raise ValueError('Measurements and groups must be different files')
    return cfg


def plan(cfg):
    return [{'stage': s, 'environment': 'matched' if s == 'matched' else 'main',
             'command': [cfg['python_matched' if s == 'matched' else 'python_main'],
                         '-B', '-X', 'utf8', str(Path(__file__).resolve()), '_stage',
                         '--config', str(cfg['output_dir']/'resolved_config.json'), '--stage', s]}
            for s in STAGES]


def environment(kind):
    expected = REQUIRED[kind]
    found = {'python': platform.python_version()}
    for name in expected:
        if name == 'python':
            continue
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = 'NOT_INSTALLED'
    bad = {k: {'expected': v, 'actual': found[k]} for k,v in expected.items() if found[k] != v}
    if bad:
        raise RuntimeError('Environment mismatch: '+json.dumps(bad))
    return {'environment': kind, 'versions': found, 'platform': platform.platform()}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def modules(runtime):
    sys.path[:0] = [str(SOURCE/'originals/ijcg_analysis'), str(SOURCE/'originals/s12/code')]
    expected = {'common':SOURCE/'originals/ijcg_analysis/common.py',
                'nrr_stage_common':SOURCE/'originals/ijcg_analysis/nrr_stage_common.py',
                'audit_core':SOURCE/'originals/s12/code/audit_core.py',
                'historical_rule':SOURCE/'originals/s12/code/historical_rule.py',
                'legacy_common':SOURCE/'originals/s12/code/legacy_common.py'}
    for name,path in expected.items():
        current = sys.modules.get(name)
        if current is not None and Path(current.__file__).resolve() != path.resolve():
            del sys.modules[name]
    common = importlib.import_module('common')
    nrr = importlib.import_module('nrr_stage_common')
    core = importlib.import_module('audit_core')
    for module in [common,nrr,core]:
        if not Path(module.__file__).resolve().is_relative_to(SOURCE):
            raise ImportError('Scientific module must come from bundled source')
    nrr.OUTPUT_DIR = runtime/'IJCG投稿/results/nrr_stage1_stage2'
    return common, nrr, core


def inspect_data(cfg, enforce_counts=True):
    import numpy as np
    import pandas as pd
    common, nrr, _ = modules(cfg['output_dir']/'runtime')
    raw, _ = common.load_data(cfg['measurement_csv'], False)
    data, rejected = common.load_data(cfg['measurement_csv'], True)
    blocks = pd.read_csv(cfg['groups_csv'], usecols=['well_id','spatial_block']).rename(columns={'well_id':'well_name'})
    if blocks.well_name.duplicated().any() or blocks.isna().any().any():
        raise ValueError('Groups require one nonmissing assignment per well')
    data = data.merge(blocks, on='well_name', how='left', validate='many_to_one', sort=False)
    if data.spatial_block.isna().any() or not data.source_row.is_unique:
        raise ValueError('Missing group or nonunique source row')
    matrix, groups = common.add_window_features(data)
    cols = [x for x in groups['raw_through_w100'] if x.startswith(('GR','DEN'))]
    counts = dict(raw_rows=len(raw), eligible_rows=len(data), excluded_rows=len(rejected),
                  wells=data.well_name.nunique(), spatial_blocks=data.spatial_block.nunique(),
                  segments=data.groupby(['well_name','segment_id']).ngroups, features=len(cols))
    protocol = read(cfg['protocol'])
    if enforce_counts and any(counts[k] != protocol[k] for k in counts):
        raise ValueError('Input identity differs from paper: '+str(counts))
    expected_cols = read(HERE/'config/feature_contract.json')['class_models_ordered_columns_26']
    if cols != expected_cols or not np.isfinite(matrix[cols].to_numpy()).all():
        raise ValueError('Feature order or finite-value contract failed')
    return data, matrix, cols, {'counts': counts, 'features': cols,
                              'folds': nrr.split_audit_rows(data), 'synthetic_only': not enforce_counts}


def file_inventory(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def source_identity():
    paths = list(SOURCE.rglob('*.py')) + [Path(__file__), HERE/'config/paper_protocol.json', HERE/'config/feature_contract.json']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def prepare(cfg):
    out = cfg['output_dir']
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output must be new or empty: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    for rel in ['logs','status','runtime/NRR投稿/stage6_submission_readiness/cache',
                'runtime/NRR投稿/stage6_submission_readiness/evidence',
                'runtime/GAS投稿/stage12_substantive_revision_20260907/results',
                'runtime/GAS投稿/stage12_substantive_revision_20260907/private',
                'paired/results','paired/qa','matched']:
        (out/rel).mkdir(parents=True, exist_ok=True)
    for key,rel in [('measurement_csv','IJCG投稿/data/measurement_data.csv'),
                    ('groups_csv','IJCG投稿/outputs/geology_aware_subfacies_validation_v1/well_spatial_blocks_v1.csv')]:
        dest = out/'runtime'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg[key], dest)
    save(out/'resolved_config.json', cfg)
    save(out/'RUN.json', {'status':'PREPARED', 'source_identity':source_identity(),
         'input_sha256':{key:sha(cfg[key]) for key in ['measurement_csv','groups_csv']},
         'config_sha256':sha(out/'resolved_config.json')})
    save(out/'checkpoint.json', file_inventory(out/'runtime'))


def guard_outputs(out, stage):
    log = (out/'logs'/f'{stage}_file_reads.jsonl').open('w', encoding='utf-8')
    seen = set()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str,bytes)):
            return
        try:
            p = Path(os.fsdecode(args[0])).resolve()
        except (ValueError,OSError):
            return
        mode = str(args[1])
        if p.suffix.lower() in ['.npz','.joblib','.pkl'] and not p.is_relative_to(out):
            raise PermissionError('Prediction/model access outside the new run is prohibited: '+str(p))
        if p.is_relative_to(out/'runtime') and (str(p),mode) not in seen:
            seen.add((str(p),mode))
            log.write(json.dumps({'path':p.relative_to(out).as_posix(),'mode':mode})+'\n')
            log.flush()
    sys.addaudithook(audit)
    return log


def execute_stage(stage, out, data, matrix, cols):
    """Only path bindings differ from the accepted runner; algorithms are bundled."""
    import pandas as pd
    runtime = out/'runtime'
    _, nrr, _ = modules(runtime)
    s6 = runtime/'NRR投稿/stage6_submission_readiness'
    s12 = runtime/'GAS投稿/stage12_substantive_revision_20260907'
    if stage == 'class':
        m = importlib.import_module('run_nrr_stage1')
        timing = []
        for scheme in nrr.SCHEMES:
            _, _, rows = m.fit_predict_oof(matrix, cols, scheme, 'RF_MULTISCALE','NO_RT_GR_DEN')
            if any(r['cache_reused'] for r in rows):
                raise RuntimeError('Class cache unexpectedly reused')
            timing += rows
        pd.DataFrame(timing).to_csv(out/'class_fit_times.csv', index=False)
    elif stage == 'calibration':
        m = load('liuji_calibration', SOURCE/'originals/s6/build_major_revision_evidence.py')
        m.CACHE_DIR = s6/'cache'
        m.EVIDENCE_DIR = s6/'evidence'
        outputs = m.run_primary_calibration(matrix, cols)
        names = ['calibration_fold_audit_gr_den.csv','calibration_per_well_gr_den.csv',
                 'calibration_summary_gr_den.csv','reliability_gr_den.csv','confidence_curves_gr_den.csv']
        for frame,name in zip(outputs,names):
            frame.to_csv(s6/'evidence'/name,index=False)
        folds = outputs[0].copy()
        folds['fallback_used'] = ~folds.calibration_target_met
        folds.to_csv(s12/'results/calibration_fold_details.csv',index=False)
    elif stage == 'phases':
        m = load('liuji_phases', SOURCE/'originals/s6/build_submission_readiness_evidence.py')
        m.RULE_RESOURCE = SOURCE/'originals/s6'
        outputs = m.run_phase_complete_downsampling(matrix, cols, m.load_rule_module())
        for frame,name in zip(outputs,['downsample_all_phases_per_well.csv','downsample_by_phase.csv','downsample_phase_aggregate.csv']):
            frame.to_csv(s6/'evidence'/name,index=False)
    elif stage == 'triage':
        m = load('liuji_triage', SOURCE/'originals/s12/scripts/run_analysis.py')
        m.ROOT, m.S6, m.S12, m.RESULTS = runtime, s6, s12, s12/'results'
        save(m.RESULTS/'static_audit.json', {'gate':'input and fold checks',
             'historical_boundary_audit_run':False, 'calibration_records':'THIS_RUN'})
        m.triage(data)
    elif stage == 'paired':
        paired = out/'paired'
        (paired/'ANALYSIS_PROTOCOL.md').write_text('# Fixed Liuji protocol\n\n```json\n'+
             (HERE/'config/paper_protocol.json').read_text(encoding='utf-8')+'\n```\n',encoding='utf-8')
        (paired/'qa/baseline_manifest.csv').write_text('path,sha256\nANALYSIS_PROTOCOL.md,'+
             sha(paired/'ANALYSIS_PROTOCOL.md')+'\n',encoding='utf-8')
        m = load('liuji_paired', SOURCE/'originals/s17/run_s17.py')
        m.HERE = paired
        previous = sys.argv
        try:
            sys.argv = ['run_s17.py','--data-root',str(runtime)]
            m.main()
        finally:
            sys.argv = previous
        # The legacy field tests protocol preservation, not old-result equality.
        save(paired/'qa/BASELINE_FIELD_SCOPE.json', {'baseline_unchanged_means':'only the bundled protocol stayed unchanged',
             'historical_result_comparison':'NOT_RUN'})
    elif stage == 'matched':
        os.environ.update(M001_CONTROLLED_ROOT=str(runtime), M001_RECOVERED_S12=str(SOURCE/'originals/s12'),
             M001_DIAG_HELPER=str(SOURCE/'adapters/_diag_mechanism_intervals.py'), M001_RECOVERY_OUTPUT=str(out/'matched'))
        m = load('liuji_matched', SOURCE/'adapters/_run_control_export_fixed.py')
        m.main()
    else:
        raise ValueError(stage)


def worker(cfg, stage):
    out = cfg['output_dir']
    run = read(out/'RUN.json')
    if run['config_sha256'] != sha(out/'resolved_config.json') or run['source_identity'] != source_identity():
        raise RuntimeError('Run configuration or source changed')
    if read(out/'checkpoint.json') != file_inventory(out/'runtime'):
        raise RuntimeError('Runtime modified outside the completed stages')
    for s in STAGES[:STAGES.index(stage)]:
        if not (out/'status'/f'{s}_COMPLETE.json').is_file():
            raise RuntimeError('Prior stage is incomplete: '+s)
    started = out/'status'/f'{stage}_STARTED.json'
    if started.exists():
        raise RuntimeError('Stage already started; use a fresh run after failure')
    env = environment('matched' if stage == 'matched' else 'main')
    os.environ['MPLCONFIGDIR'] = str(out/'mplconfig')
    access = guard_outputs(out,stage)
    staged = dict(cfg, measurement_csv=out/'runtime/IJCG投稿/data/measurement_data.csv',
                  groups_csv=out/'runtime/IJCG投稿/outputs/geology_aware_subfacies_validation_v1/well_spatial_blocks_v1.csv')
    data,matrix,cols,identity = inspect_data(staged)
    save(out/'status'/f'{stage}_INPUT.json',identity)
    save(started,{'stage':stage,'environment':env})
    from sklearn.ensemble import RandomForestClassifier
    original_fit = RandomForestClassifier.fit
    fits = []
    def counted_fit(model,X,y,*args,**kwargs):
        fits.append({'rows':len(y),'features':X.shape[1],'params':model.get_params()})
        save(out/'logs'/f'{stage}_fits.json',fits)
        return original_fit(model,X,y,*args,**kwargs)
    RandomForestClassifier.fit = counted_fit
    start = time.monotonic()
    execute_stage(stage,out,data,matrix,cols)
    expected = read(cfg['protocol'])['fit_counts'][stage]
    if len(fits) != expected:
        raise RuntimeError(f'Fit count mismatch: {len(fits)} != {expected}')
    save(out/'checkpoint.json',file_inventory(out/'runtime'))
    save(out/'status'/f'{stage}_COMPLETE.json',{'stage':stage,'new_RF_fits':len(fits),
         'seconds':time.monotonic()-start,'status':'COMPUTED_NOT_INDEPENDENTLY_ACCEPTED'})
    access.flush()


def launch(cfg):
    # Environment checks occur before creating an output directory or fitting.
    for kind in ['main','matched']:
        subprocess.run([cfg['python_'+kind],'-B','-X','utf8',str(Path(__file__).resolve()),
                        'check-env','--environment',kind],check=True,timeout=60)
    prepare(cfg)
    for item in plan(cfg):
        log = cfg['output_dir']/'logs'/f"{item['stage']}.log"
        with log.open('w',encoding='utf-8') as f:
            subprocess.run(item['command'],check=True,stdout=f,stderr=subprocess.STDOUT,
                           timeout=cfg['stage_timeout_seconds'])
    save(cfg['output_dir']/'COMPLETE.json',{'status':'COMPUTED_NOT_INDEPENDENTLY_ACCEPTED',
         'stages':list(STAGES),'historical_comparison':'NOT_RUN'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan','inspect','check-env','run','_stage'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--environment',choices=['main','matched'],default='main')
    parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--allow-training',action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'check-env':
            print(json.dumps(environment(args.environment),indent=2))
            return
        if not args.config:
            parser.error('--config is required')
        cfg = configuration(args.config)
        if args.action == 'plan':
            print(json.dumps(plan(cfg),indent=2))
        elif args.action == 'inspect':
            print(json.dumps(inspect_data(cfg)[3],default=str,indent=2))
        elif args.action == 'run':
            if not args.allow_training:
                parser.error('Full computation requires --allow-training')
            launch(cfg)
        else:
            if args.stage is None:
                parser.error('--stage is required for internal worker')
            worker(cfg,args.stage)
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as e:
        parser.exit(2,str(e)+'\n')


if __name__ == '__main__':
    main()
