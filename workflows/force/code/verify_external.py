"""Independent readback of saved predictions and every per-well accounting row."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_external import load, SCENARIOS, ROOT
import run_external
from external_core import rule


def scalar_reference(gr, den, z, step, j):
    lith=[]
    for g,d in zip(gr,den):
        if d<1.75 and g<70:k=0
        elif g>=80 and d>=2.3:k=1
        elif 60<=g<80:k=2
        elif 40<=g<60:k=3
        elif 20<=g<40:k=4
        elif g<20:k=5
        else:k=1
        lith.append(k)
    k=lith[j]
    if k==0:return 0
    if k==1:return 5 if gr[j]>90 else 4
    if k==2:return 3
    if k in (4,5):return 1
    left=right=j
    while left and lith[left-1]==k:left-=1
    while right+1<len(lith) and lith[right+1]==k:right+=1
    width=z[right]-z[left]+step
    neighborhood={lith[q] for q in range(len(z)) if -.5-1e-8<=z[q]-z[j]<.5-1e-8}
    deeper=any(lith[q] in (1,2) for q in range(j+1,len(z)) if z[q]-z[j]<=.25+1e-8)
    shallow=any(lith[q] in (4,5) for q in range(j) if z[j]-z[q]<=.25+1e-8)
    return 3 if width<2.-1e-8 and len(neighborhood)>=3 and {2,3}<=neighborhood and not(deeper and shallow) else 2


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    args=parser.parse_args()
    ROOT=args.workspace.resolve()
    if not (ROOT/'qa/EXTERNAL_ANALYSIS_QA.json').is_file():
        parser.error('A completed FORCE workspace is required')
    run_external.ROOT=ROOT
    data, inventory=load()
    pairs=pd.read_csv(ROOT/'results/force_paired_counts.csv')
    folds=pd.read_csv(ROOT/'results/force_folds.csv').set_index('well').fold
    expectedfold=data.well.map(folds).to_numpy()
    wellidx=data.groupby('well').indices
    rows=[]; independent=[]
    for stride,phase in SCENARIOS:
        a=np.load(ROOT/f'data/cache/prediction_s{stride}_p{phase}.npz')
        np.testing.assert_array_equal(a['source_row'],data.source_row)
        np.testing.assert_array_equal(a['fold'],expectedfold)
        f,c,p=[a[k] for k in ['fine','coarse','prediction']]
        # Encode the three pairwise equalities rather than calling the production partition.
        bits=(f==c).astype(int)*4+(c==p).astype(int)*2+(f==p).astype(int)
        assert set(np.unique(bits)) <= {0,1,2,4,7}
        for w,idx in wellidx.items():
            b=bits[idx]; abcde=[int((b==v).sum()) for v in [7,4,2,1,0]]
            expected=pairs[(pairs.well==w)&(pairs.stride==stride)&(pairs.phase==phase)].iloc[0]
            assert abcde==[int(expected[k]) for k in 'ABCDE']
            r=sum(abcde[2:]); m=abcde[1]+abcde[2]+abcde[4]
            assert r==expected.rule_changed and m==expected.RF_vs_fine
            assert m-r==abcde[1]-abcde[3]==expected.RF_minus_rule
            independent.append({'well':w,'stride':stride,'phase':phase,'rule_changed':r/len(idx),'RF_vs_fine':m/len(idx),'RF_minus_rule':(m-r)/len(idx)})
        if stride==1:
            fine=f.copy()
            pd.DataFrame({'well':data.well,'state':f}).groupby(['well','state']).size().rename('rows').reset_index().to_csv(ROOT/'results/force_operational_state_coverage.csv',index=False)
    independent=pd.DataFrame(independent).groupby(['stride','well']).mean(numeric_only=True).reset_index()
    original=pd.read_csv(ROOT/'results/force_paired_summary.csv')
    for stride,g in independent.groupby('stride'):
        for metric in ['rule_changed','RF_vs_fine','RF_minus_rule']:
            got=float(g[metric].mean())
            target=original[(original.stride==stride)&(original.metric==metric)].iloc[0]['mean']
            assert abs(got-target)<1e-12
        rows.append({'stride':int(stride),'wells_RF_above_rule':int((g.RF_minus_rule>0).sum()),
                     'wells_RF_below_rule':int((g.RF_minus_rule<0).sum()),'wells_equal':int((g.RF_minus_rule==0).sum())})
    pd.DataFrame(rows).to_csv(ROOT/'results/force_direction_counts.csv',index=False)
    rng=np.random.default_rng(20261031); checked=0; max_step_deviation=0
    for _,g in data.groupby('segment'):
        gr,den,z=g.GR.to_numpy(),g.DEN.to_numpy(),g.depth.to_numpy(); step=g.step.iloc[0]
        max_step_deviation=max(max_step_deviation,float(np.max(abs(np.diff(z)-step))))
        out=rule(gr,den,z,step)[0]
        for j in rng.choice(len(g),min(5,len(g)),replace=False):
            assert scalar_reference(gr,den,z,step,j)==out[j]
            checked+=1
    stats=data[['GR','DEN','depth']].describe(percentiles=[.01,.5,.99]).round(6)
    stats.to_csv(ROOT/'results/force_curve_ranges.csv')
    s17=ROOT.parent/'M001_CG_S17_20261031_01'
    manifest_path=ROOT/'qa/S17_PRESERVATION_MANIFEST.csv'
    preserved=0
    preservation='NOT_CHECKED_BASELINE_NOT_AVAILABLE'
    if s17.is_dir() and manifest_path.is_file():
        manifest=pd.read_csv(manifest_path)
        for item in manifest.itertuples():
            p=s17/item.path
            assert p.exists() and p.stat().st_size==item.size
            assert hashlib.sha256(p.read_bytes()).hexdigest().upper()==item.sha256.upper()
        preserved=len(manifest)
        preservation='PASS'
    qa={'source_row_and_fold_alignment_scenarios':len(SCENARIOS),'independent_pair_rows':len(pairs),
        'independent_equal_well_recomputation':'PASS','scalar_rule_checks':checked,
        'max_native_step_deviation_m':max_step_deviation,'S17_preserved_files':preserved,
        'S17_preservation':preservation,'model_comparison_scope':'New external RF fits, not a frozen Liuji model',
        'bootstrap_scope':'Descriptive well variability conditional on fixed fitted folds; no model-refit or basin-level uncertainty'}
    (ROOT/'qa/INDEPENDENT_READBACK.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
