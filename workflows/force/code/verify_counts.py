"""Reaggregate retained FORCE integer counts without fitting any classifier."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=pd.read_csv(ROOT/'results/force_paired_counts.csv')
    assert (p[list('ABCDE')].sum(axis=1)==p.n).all()
    assert (p.C+p.D+p.E==p.rule_changed).all()
    assert (p.B+p.C+p.E==p.RF_vs_fine).all()
    assert (p.B-p.D==p.RF_minus_rule).all()
    cols=list('ABCDE')+['rule_changed','RF_vs_fine','RF_minus_rule']
    p[cols]=p[cols].div(p.n,axis=0)
    perwell=p.groupby(['stride','well'])[cols].mean()
    means=perwell.groupby('stride').mean()
    expected=pd.read_csv(ROOT/'results/force_paired_summary.csv')
    for row in expected.itertuples():
        assert np.isclose(means.loc[row.stride,row.metric],row.mean,rtol=0,atol=1e-12)
    native=pd.read_csv(ROOT/'results/force_native_port_disagreement.csv')
    output={'paired_well_phase_records':len(p),
            'native_context_disagreement_equal_well_percent':float((native.physical_vs_index/native.n).mean()*100),
            'native_context_disagreement_pooled_percent':float(native.physical_vs_index.sum()/native.n.sum()*100),
            'scope':'Aggregation check of retained outputs; not raw-data training replay'}
    print(json.dumps(output,indent=2))
    print((means*100).round(6).to_string())


if __name__=='__main__': main()
