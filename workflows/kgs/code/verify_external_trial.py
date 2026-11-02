"""Independent accounting over retained input and rule-score cache.

Does not refit RF models or claim independent RF point-prediction verification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rebuild_rows(path):
    frame = pd.read_csv(path)
    frame = frame.rename(columns={'gr':'GR','den':'DEN'})
    frame['well'] = frame.well.astype(str)
    duplicate = frame.duplicated(['well','depth'],keep=False)
    good = np.isfinite(frame[['depth','GR','DEN']]).all(axis=1) & (frame.GR>0)&(frame.DEN>0)&(frame.DEN<=5)&~duplicate
    parts=[]
    segno=0
    for well,raw in frame.groupby('well',sort=True):
        raw=raw.sort_values('depth',kind='stable')
        d=np.sort(raw.depth.dropna().unique())
        step=float(np.median(np.diff(d)[np.diff(d)>0]))
        ok=good.loc[raw.index].to_numpy()
        depths=raw.depth.to_numpy()
        # Each uninterrupted chain must have all valid rows and native spacing.
        connection=ok[1:]&ok[:-1]&np.isclose(np.diff(depths),step,rtol=.01,atol=1e-5)
        start=0
        for stop in np.r_[np.flatnonzero(~connection)+1,len(raw)]:
            if stop-start>=21 and ok[start:stop].all():
                part=raw.iloc[start:stop].copy()
                part['segment']=segno
                part['step']=step
                segno+=1
                parts.append(part)
            start=stop
    return pd.concat(parts,ignore_index=True)


def select_rule(score,part,budget=.1):
    blocks=[]
    # Reconstruct every physical block using explicit interval scans.
    for _,g in part.groupby('segment',sort=False):
        loc=part.index.get_indexer(g.index)
        z=g.depth.to_numpy()
        code=np.floor((z-z[0]+1e-8)/1.).astype(int)
        for value in np.unique(code):
            blocks.append(loc[code==value])
    mean=np.array([np.mean(score[b]) for b in blocks])
    well=str(part.well.iloc[0])
    keys=np.array([int.from_bytes(hashlib.sha256(f'20260907|{well}|{i}'.encode()).digest()[:8],'big') for i in range(len(blocks))],dtype=np.uint64)
    order=sorted(range(len(blocks)),key=lambda i:(-mean[i],keys[i]))
    need=int(np.floor(budget*len(part)))
    chosen=[]
    for i in order:
        take=min(need-len(chosen),len(blocks[i]))
        chosen.extend(blocks[i][:take].tolist())
        if len(chosen)==need:break
    assert len(chosen)==need and len(set(chosen))==need
    return np.asarray(chosen,dtype=int)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out)
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    data=rebuild_rows(protocol['input'])
    selected=pd.read_csv(out/'well_selection.csv',dtype={'well':str})
    data=data[data.well.isin(selected.loc[selected.selected,'well'])].reset_index(drop=True)
    fine=np.load(out/'cache/native_rule.npy')
    assert len(data)==len(fine)
    results=pd.read_csv(out/'capture_by_well_phase.csv',dtype={'well':str})
    groups=data.groupby('well',sort=True).indices
    checks=[]
    for stride,phase in protocol['scenarios']:
        saved=np.load(out/f'cache/outcomes_s{stride}_p{phase}.npz')
        target=saved['coarse']!=fine
        assert np.array_equal(target,saved['target'])
        for well,idx in groups.items():
            part=data.iloc[idx]
            t=target[idx]
            records=results[(results.well==well)&(results.stride==stride)&(results.phase==phase)]
            assert len(records)==3
            assert (records.target_rows==t.sum()).all()
            assert (records.reviewed_rows==int(np.floor(.1*len(idx)))).all()
            chosen=select_rule(saved['stability'][idx],part)
            capture=int(t[chosen].sum())
            r=records[records.method=='rule_stability'].iloc[0]
            assert capture==r.captured_rows
            if t.sum():
                assert np.isclose(capture/t.sum(),r.capture_rate,atol=1e-14)
            else:
                assert pd.isna(r.capture_rate)
            checks.append({'well':well,'stride':stride,'phase':phase,'rows':len(idx),
                           'targets':int(t.sum()),'rule_captured':capture,'reviewed_rows':len(chosen)})
    checks=pd.DataFrame(checks)
    checks.to_csv(out/'independent_rule_checks.csv',index=False)
    assessed=[]
    for stride,g in checks.groupby('stride'):
        bywell=g.groupby('well').agg(phases=('phase','size'),nonzero_phases=('targets',lambda v:int((v>0).sum())))
        assessed.append({'stride':int(stride),'all_wells':len(bywell),'all_well_phases':len(g),
                         'nonzero_target_well_phases':int((g.targets>0).sum()),
                         'zero_target_well_phases':int((g.targets==0).sum()),
                         'wells_with_any_nonzero_target_phase':int((bywell.nonzero_phases>0).sum()),
                         'wells_with_all_nonzero_target_phases':int((bywell.nonzero_phases==bywell.phases).sum())})
    pd.DataFrame(assessed).to_csv(out/'assessable_denominators.csv',index=False)
    folds=pd.read_csv(out/'folds.csv',dtype={'well':str})
    foldcheck=json.loads((out/'fold_checks.json').read_text())
    assert folds.well.nunique()==len(groups)
    assert set(folds.well)==set(groups)
    for fold in foldcheck:
        assert not set(fold['train_wells'])&set(fold['test_wells'])
        assert set(fold['test_wells'])==set(folds.loc[folds.fold==fold['fold'],'well'])
        assert fold['training_rows']<=len(fold['train_wells'])*protocol['train_cap_per_well_total']
    output={'input_qc_rows_independently_rebuilt':len(data),'wells':len(groups),
            'segments':int(data.segment.nunique()),'well_phase_target_counts_verified':len(checks),
            'method_budget_records_verified':len(results),'rule_capture_records_independently_selected':len(checks),
            'well_fold_separation':'PASS','training_cap':'PASS',
            'assessable_denominators':assessed,
            'RF_models_refitted_by_verifier':False,'RF_point_predictions_independently_verified':False,
            'scope':'Independent data-row accounting, target comparison, strict budget, and selection from saved rule scores; not a second implementation of rule physics or RF.'}
    (out/'independent_verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
