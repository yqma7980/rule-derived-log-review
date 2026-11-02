"""Whole-well external application of one fixed RF and depth-aware diagnostics."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import zipfile
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from external_core import rule,features,interpolate,partition,bootstrap_mean

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/inherited'))
from audit_core import simplified_rule
SCENARIOS=[(1,0)]+[(s,p) for s in (2,4,5) for p in range(s)]
SEED=20261031

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonout(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def self_test():
    rng=np.random.default_rng(SEED)
    for n in (21,40,101,1000):
        gr=rng.uniform(1,150,n); den=rng.uniform(1.3,3.,n); z=100.+np.arange(n)*.05
        np.testing.assert_array_equal(rule(gr,den,z,.05)[0],simplified_rule(gr,den))
    # Test threshold-aligned runs and clipped boundaries, not only random traces.
    gr=np.repeat([10.,50.,70.,50.,95.],[10,39,10,40,10]); den=np.full(len(gr),2.4)
    np.testing.assert_array_equal(rule(gr,den,np.arange(len(gr))*.05,.05)[0],simplified_rule(gr,den))
    t=np.array([(f,c,p) for f in range(6) for c in range(6) for p in range(6)])
    m=partition(*t.T)
    for j,(f,c,p) in enumerate(t):
        k=0 if f==c==p else 1 if f==c else 2 if p==c else 3 if p==f else 4
        assert m[k,j]
    np.testing.assert_array_equal((t[:,2]!=t[:,0]).astype(int)-(t[:,1]!=t[:,0]).astype(int),m[1].astype(int)-m[3].astype(int))
    z=np.arange(30)*.152; x=np.arange(30.)
    for s,p in SCENARIOS:
        np.testing.assert_allclose(interpolate(x,z,s,p),x,atol=1e-12)
    return {'uniform_grid_rule_equivalence_traces':5,'partition_triplets':216,'linear_interpolation_tests':12}

def load():
    path=ROOT/'data/external/FORCE_train.zip'
    with zipfile.ZipFile(path) as archive:
        csvs=[n for n in archive.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX')]
        assert len(csvs)==1,csvs
        df=pd.read_csv(archive.open(csvs[0]),sep=';',usecols=['WELL','DEPTH_MD','GR','RHOB','FORCE_2020_LITHOFACIES_LITHOLOGY'])
    df=df.rename(columns={'WELL':'well','DEPTH_MD':'depth','RHOB':'DEN','FORCE_2020_LITHOFACIES_LITHOLOGY':'interpreted_lithology'})
    df['source_row']=np.arange(2,len(df)+2)
    raw_count=len(df)
    finite=np.isfinite(df[['depth','GR','DEN']]).all(axis=1)
    valid=finite&(df.GR>0)&(df.DEN>0)&(df.DEN<=5)
    duplicate=df.duplicated(['well','depth'],keep=False)
    valid &= ~duplicate
    groups=[]; populations=[]; gid=0
    for well, raw in df.groupby('well',sort=True):
        raw=raw.sort_values('depth',kind='stable')
        dz=np.diff(raw.depth.dropna().unique()); step=float(np.median(dz[dz>0]))
        if not np.isfinite(step) or not .01<step<1.:raise ValueError((well,step))
        part=raw[valid.loc[raw.index]].copy()
        if len(part):
            gaps=np.r_[True,np.diff(part.depth)>1.5*step]
            # Any removed raw row breaks context, even when the depth gap is small.
            positions=pd.Series(np.arange(len(raw)),index=raw.index).loc[part.index].to_numpy()
            gaps[1:] |= np.diff(positions)>1
            ids=np.cumsum(gaps)
        else:ids=np.array([])
        kept=0; segs=0; short=0
        for _, g in part.groupby(ids,sort=False):
            if len(g)<21:short+=len(g);continue
            g=g.copy();g['segment']=gid;g['step']=step
            gid+=1;groups.append(g);kept+=len(g);segs+=1
        populations.append({'well':well,'raw_rows':len(raw),'invalid_rows':int((~valid.loc[raw.index]).sum()),
                           'short_segment_rows':short,'eligible_rows':kept,'segments':segs,'native_step_m':step,
                           'missing_GR_rows':int(raw.GR.isna().sum()),'missing_density_rows':int(raw.DEN.isna().sum())})
    data=pd.concat(groups,ignore_index=True)
    pop=pd.DataFrame(populations)
    assert pop.raw_rows.sum()==raw_count and (pop.invalid_rows+pop.short_segment_rows+pop.eligible_rows).sum()==raw_count
    pop.to_csv(ROOT/'results/force_population.csv',index=False)
    data.groupby(['well','interpreted_lithology']).size().rename('rows').reset_index().to_csv(ROOT/'results/force_interpretation_coverage.csv',index=False)
    inventory={'raw_rows':raw_count,'raw_wells':int(df.well.nunique()),'eligible_rows':len(data),
               'eligible_wells':int(data.well.nunique()),'segments':gid,'duplicate_key_rows':int(duplicate.sum()),
               'excluded_invalid':int(pop.invalid_rows.sum()),'excluded_short_segments':int(pop.short_segment_rows.sum()),
               'native_steps_m':sorted(pop.native_step_m.round(6).unique().tolist()),'source_sha256':sha(path),
               'data_columns':list(df.columns),'external_interpretation_used_for_training':False}
    jsonout(ROOT/'qa/DATA_INVENTORY.json',inventory)
    print(json.dumps(inventory,indent=2),flush=True)
    return data,inventory

def summary(table,keys,cols,name):
    perwell=table.groupby(keys+['well'],as_index=False)[cols].mean()
    perwell.to_csv(ROOT/f'results/{name}_by_well.csv',index=False)
    rows=[]
    for key,g in perwell.groupby(keys,sort=True):
        if not isinstance(key,tuple):key=(key,)
        for col in cols:
            v=g[col].dropna();lo,hi=bootstrap_mean(v)
            rows.append(dict(zip(keys,key))|{'metric':col,'n_wells':len(v),'mean':v.mean(),
                         'min':v.min(),'max':v.max(),'bootstrap_low':lo,'bootstrap_high':hi})
    pd.DataFrame(rows).to_csv(ROOT/f'results/{name}_summary.csv',index=False)
    return perwell

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inspect',action='store_true');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    tests=self_test()
    if args.self_test:print(tests);return
    data,inventory=load()
    if args.inspect:return
    cache=ROOT/'data/cache';cache.mkdir(exist_ok=True)
    groups=data.groupby('segment',sort=False).indices
    wells=data.groupby('well',sort=True).indices
    names=np.array(sorted(wells)); shuffled=np.random.default_rng(SEED).permutation(names)
    foldmap={w:i%5 for i,w in enumerate(shuffled)}
    fold=data.well.map(foldmap).to_numpy()
    pd.DataFrame([{'well':w,'fold':foldmap[w],'eligible_rows':len(wells[w])} for w in names]).to_csv(ROOT/'results/force_folds.csv',index=False)
    fine=np.empty(len(data),np.int8);base=fine.copy();run=np.empty(len(data)); indexfine=fine.copy()
    rawX=np.empty((len(data),26),np.float32);spans=[]
    for seg,idx in groups.items():
        g=data.iloc[idx];step=g.step.iloc[0];z=g.depth.to_numpy();gr=g.GR.to_numpy();den=g.DEN.to_numpy()
        fine[idx],base[idx],run[idx]=rule(gr,den,z,step)
        indexfine[idx]=simplified_rule(gr,den)
        rawX[idx],actual=features(gr,den,z,step)
        spans.append({'segment':int(seg),'step':step,'span1':actual[0],'span2':actual[1],'span3':actual[2]})
    pd.DataFrame(spans).to_csv(ROOT/'results/force_feature_spans.csv',index=False)
    pd.DataFrame([{'well':w,'n':len(idx),'physical_vs_index':int((fine[idx]!=indexfine[idx]).sum())} for w,idx in wells.items()]).to_csv(ROOT/'results/force_native_port_disagreement.csv',index=False)
    models=[];foldchecks=[]
    for k in range(5):
        trainnames=[w for w in names if foldmap[w]!=k];testnames=[w for w in names if foldmap[w]==k]
        assert set(trainnames).isdisjoint(testnames)
        chosen=np.concatenate([wells[w][np.linspace(0,len(wells[w])-1,min(2000,len(wells[w])),dtype=int)] for w in trainnames])
        assert np.all(fold[chosen]!=k) and len(np.unique(chosen))==len(chosen)
        model=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,max_features='sqrt',
                    class_weight='balanced_subsample',random_state=SEED+k,n_jobs=8)
        model.fit(rawX[chosen],fine[chosen]);models.append(model)
        joblib.dump(model,cache/f'rf_fold{k}.joblib',compress=1)
        foldchecks.append({'fold':k,'train_wells':len(trainnames),'test_wells':len(testnames),
                           'train_rows':len(chosen),'test_rows':int((fold==k).sum()),
                           'train_classes':model.classes_.tolist(),'well_overlap':0})
        print(f'Fitted fixed RF fold {k}: {len(chosen)} training rows',flush=True)
    del rawX
    jsonout(ROOT/'qa/FOLD_CHECKS.json',foldchecks)
    pairrows=[];channelrows=[];stratarows=[];portrows=[]
    margin=np.minimum(np.min(np.abs(data.GR.to_numpy()[:,None]-[20,40,60,70,80,90]),axis=1),
                      np.min(np.abs(data.DEN.to_numpy()[:,None]-[1.75,2.30]),axis=1)/.01)
    for stride,phase in SCENARIOS:
        c=np.empty_like(fine);b=np.empty_like(base);ci=np.empty_like(fine)
        ch={key:np.empty_like(fine) for key in ['GR_only','DEN_only']}
        X=np.empty((len(data),26),np.float32)
        for seg,idx in groups.items():
            g=data.iloc[idx];z=g.depth.to_numpy();step=g.step.iloc[0];gr=g.GR.to_numpy();den=g.DEN.to_numpy()
            cg=interpolate(gr,z,stride,phase);cd=interpolate(den,z,stride,phase)
            c[idx],b[idx],_=rule(cg,cd,z,step);ci[idx]=simplified_rule(cg,cd)
            ch['GR_only'][idx]=rule(cg,den,z,step)[0];ch['DEN_only'][idx]=rule(gr,cd,z,step)[0]
            X[idx]=features(cg,cd,z,step)[0]
        prediction=np.empty_like(fine)
        for k,model in enumerate(models):
            idx=np.flatnonzero(fold==k);prediction[idx]=model.predict(X[idx])
        del X
        masks=partition(fine,c,prediction)
        np.testing.assert_array_equal((prediction!=fine).astype(int)-(c!=fine).astype(int),masks[1].astype(int)-masks[3].astype(int))
        if stride==1:
            assert np.array_equal(fine,c) and np.array_equal(indexfine,ci)
            assert all(np.array_equal(fine,v) for v in ch.values())
        np.savez_compressed(cache/f'prediction_s{stride}_p{phase}.npz',source_row=data.source_row.to_numpy(),fine=fine,coarse=c,prediction=prediction,fold=fold)
        sup=run/(data.step.to_numpy()*stride)
        bins={'fine_run_support_ratio':np.digitize(sup,[1,2,4],right=True),
              'literal_threshold_distance':np.digitize(margin,[1,3],right=True)}
        ch['GR_and_DEN']=c
        for w,idx in wells.items():
            n=len(idx);step=data.step.iloc[idx[0]]
            counts={key:int(m[idx].sum()) for key,m in zip('ABCDE',masks)}
            assert sum(counts.values())==n
            pairrows.append({'well':w,'fold':foldmap[w],'stride':stride,'phase':phase,'spacing_m':step*stride,'n':n,
                **counts,'rule_changed':int((c[idx]!=fine[idx]).sum()),'RF_vs_fine':int((prediction[idx]!=fine[idx]).sum()),
                'RF_minus_rule':counts['B']-counts['D']})
            for key,v in ch.items():
                channelrows.append({'well':w,'stride':stride,'phase':phase,'channel':key,'n':n,'changed':int((v[idx]!=fine[idx]).sum())})
            portrows.append({'well':w,'stride':stride,'phase':phase,'n':n,
                 'index_changed':int((ci[idx]!=indexfine[idx]).sum()),'physical_changed':int((c[idx]!=fine[idx]).sum()),
                 'ports_disagree':int((ci[idx]!=c[idx]).sum()),
                 'context_route':int(((c[idx]!=fine[idx])&(b[idx]==base[idx])).sum()),
                 'context_compensated':int(((c[idx]==fine[idx])&(b[idx]!=base[idx])).sum())})
            for desc,levels in bins.items():
                for level in range(4 if desc=='fine_run_support_ratio' else 3):
                    select=idx[levels[idx]==level]
                    stratarows.append({'well':w,'stride':stride,'phase':phase,'descriptor':desc,'stratum':level,
                                       'n':len(select),'changed':int((c[select]!=fine[select]).sum())})
        print(f'Evaluated stride {stride} phase {phase}',flush=True)
    pair=pd.DataFrame(pairrows);pair.to_csv(ROOT/'results/force_paired_counts.csv',index=False)
    pcols=list('ABCDE')+['rule_changed','RF_vs_fine','RF_minus_rule']
    rated=pair.copy();rated[pcols]=rated[pcols].div(rated.n,axis=0)
    pw=summary(rated,['stride'],pcols,'force_paired')
    pooled=pair.groupby(['stride','phase'])[['n']+pcols].sum()
    pooled[pcols]=pooled[pcols].div(pooled.n,axis=0)
    pooled.groupby('stride')[pcols].mean().to_csv(ROOT/'results/force_pooled_sensitivity.csv')
    for name,rows,keys,cols in [
        ('force_channels',channelrows,['stride','channel'],['changed']),
        ('force_ports',portrows,['stride'],['index_changed','physical_changed','ports_disagree','context_route','context_compensated']),
        ('force_strata',stratarows,['stride','descriptor','stratum'],['changed'])]:
        t=pd.DataFrame(rows);t.to_csv(ROOT/f'results/{name}_counts.csv',index=False)
        t[cols]=t[cols].div(t.n.replace(0,np.nan),axis=0)
        summary(t,keys,cols,name)
    qa={'completed_utc':datetime.now(timezone.utc).isoformat(),'inventory':inventory,'tests':tests,
        'models_fitted':5,'classifier_families':1,'hyperparameter_search':False,
        'external_rule_thresholds_tuned':False,'scenarios':12,'fold_well_overlap':0,
        'count_identity':'PASS','zero_stride_rule_control':'PASS',
        'protocol_sha256':sha(ROOT/'ANALYSIS_PROTOCOL.md'),
        'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','scipy','joblib']}}
    jsonout(ROOT/'qa/EXTERNAL_ANALYSIS_QA.json',qa)
    print(json.dumps(qa,indent=2),flush=True)

if __name__=='__main__':main()
