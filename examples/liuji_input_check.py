"""Generated input/schema demo using the original QC and feature functions; no fit."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('liuji_runner',ROOT/'workflows/liuji/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def create_fixture(out):
    out = Path(out).resolve()
    if out == ROOT or ROOT.is_relative_to(out):
        raise ValueError('Do not use a source ancestor')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh directory')
    out.mkdir(parents=True,exist_ok=True)
    common,_,core = runner.modules(out/'unused_runtime')
    data = pd.DataFrame([{'well_name':f'SYNTHETIC_{w}', 'depth':i*.05,
                          'RT':10.,'GR':55.+30.*np.sin(i/6.),'DEN':2.35+.1*np.cos(i/8.),
                          'Subfacies':common.FACIES[2]['cn']}
                         for w in range(3) for i in range(80)])
    data.to_csv(out/'synthetic_measurements.csv',index=False)
    pd.DataFrame({'well_id':[f'SYNTHETIC_{w}' for w in range(3)],'spatial_block':[1,2,3]}).to_csv(out/'synthetic_groups.csv',index=False)
    cfg = dict(measurement_csv=str(out/'synthetic_measurements.csv'),groups_csv=str(out/'synthetic_groups.csv'),
               output_dir=str(out/'new_run'),protocol=str(ROOT/'workflows/liuji/config/paper_protocol.json'),
               python_main=sys.executable,python_matched=sys.executable,stage_timeout_seconds=60)
    runner.save(out/'synthetic_config.json',cfg)
    resolved = runner.configuration(out/'synthetic_config.json')
    table,_,_,report = runner.inspect_data(resolved,enforce_counts=False)
    report.update(kind='SYNTHETIC SCHEMA/FEATURE EXAMPLE, NOT PAPER EVIDENCE',new_RF_fits=0,
                  native_rule_rows=len(core.apply_grouped_rule(table)),
                  note='This config intentionally fails paper-population and matched-environment gates.')
    runner.save(out/'summary.json',report)
    print(json.dumps(report,indent=2,default=str))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    create_fixture(p.parse_args().output)
