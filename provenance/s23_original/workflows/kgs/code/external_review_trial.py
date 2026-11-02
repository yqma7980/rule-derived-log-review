"""Fixed-protocol external review-priority pilot; not a frozen-model transfer test.

Inputs: CSV with well, depth (m), GR (API), DEN (g/cm3), or explicit aliases.
All outcomes are synthetic operational-rule changes, not geological errors.
Only processed curves and their rule outputs enter held-out-well predictors.
"""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from external_core import features, interpolate, rule
from review_priority import evaluate_blocks, self_test as budget_test

SCENARIOS = [(s, p) for s in (2, 4, 5) for p in range(s)]
METHODS = ['rule_stability', 'category_support', 'change_prediction']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def load_data(args, out):
    names = [args.well_column, args.depth_column, args.gr_column, args.den_column]
    raw = pd.read_csv(args.input, sep=args.separator, usecols=names)
    raw = raw.rename(columns=dict(zip(names, ['well', 'depth', 'GR', 'DEN'])))
    raw['source_row'] = np.arange(2, len(raw) + 2)
    raw['depth'] = pd.to_numeric(raw.depth, errors='coerce') * args.depth_to_m
    raw['GR'] = pd.to_numeric(raw.GR, errors='coerce')
    raw['DEN'] = pd.to_numeric(raw.DEN, errors='coerce') * args.den_to_gcc
    missing_well = raw.well.isna()
    raw['well'] = raw.well.astype(str)
    finite = np.isfinite(raw[['depth', 'GR', 'DEN']]).all(axis=1)
    duplicate = raw.duplicated(['well', 'depth'], keep=False)
    valid = finite & ~missing_well & ~duplicate & (raw.GR > 0) & (raw.DEN > 0) & (raw.DEN <= 5)
    groups, rows = [], []
    seg_id = 0
    for well, g in raw.groupby('well', sort=True):
        g = g.sort_values('depth', kind='stable')
        dz = np.diff(g.depth.dropna().unique())
        dz = dz[dz > 0]
        step = float(np.median(dz)) if len(dz) else np.nan
        keep = g[valid.loc[g.index]].copy()
        eligible = short = nsegments = 0
        if len(keep) and np.isfinite(step) and .001 <= step <= 2:
            positions = pd.Series(np.arange(len(g)), index=g.index).loc[keep.index].to_numpy()
            # Nonuniform gaps break segments; no interpolation across missing runs.
            breaks = np.r_[True, ~np.isclose(np.diff(keep.depth), step, rtol=.01, atol=1e-5)]
            breaks[1:] |= np.diff(positions) != 1
            for _, part in keep.groupby(np.cumsum(breaks), sort=False):
                if len(part) < 21:
                    short += len(part)
                    continue
                part = part.copy()
                part['segment'] = seg_id
                part['step'] = step
                groups.append(part)
                seg_id += 1
                nsegments += 1
                eligible += len(part)
        else:
            short = len(keep)
        rows.append({'well': well, 'raw_rows': len(g), 'invalid_rows': len(g)-len(keep),
                     'short_or_irregular_rows': short, 'eligible_rows': eligible,
                     'segments': nsegments, 'native_step_m': step})
    pop = pd.DataFrame(rows)
    pop.to_csv(out/'population.csv', index=False)
    assert int(pop[['invalid_rows', 'short_or_irregular_rows', 'eligible_rows']].sum().sum()) == len(raw)
    if not groups:
        raise ValueError('No eligible continuous GR/DEN segments')
    data = pd.concat(groups, ignore_index=True)
    eligible_rows = len(data)
    eligible_wells = int(data.well.nunique())
    all_names = sorted(data.well.unique(), key=lambda w: hashlib.sha256(str(w).encode()).hexdigest())
    chosen_names = all_names[:args.max_wells] if args.max_wells else all_names
    pd.DataFrame([{'well':w,'sha256':hashlib.sha256(str(w).encode()).hexdigest(),
                   'selected':w in chosen_names} for w in all_names]).to_csv(out/'well_selection.csv',index=False)
    data = data[data.well.isin(chosen_names)].reset_index(drop=True)
    inv = {'input': str(Path(args.input).resolve()), 'sha256': digest(args.input),
           'raw_rows': len(raw), 'raw_wells': int(raw.well.nunique()),
           'eligible_rows': eligible_rows, 'eligible_wells': eligible_wells,
           'selected_rows':len(data),'selected_wells':int(data.well.nunique()),
           'eligible_segments': seg_id,'selected_segments':int(data.segment.nunique()),
           'duplicates_excluded': int(duplicate.sum()),
           'columns': names, 'depth_to_m': args.depth_to_m, 'density_to_gcc': args.den_to_gcc}
    dump(out/'inventory.json', inv)
    return data, inv


def build_cache(data, out):
    """Native curves are read here solely to generate diagnostic target labels."""
    cache = out/'cache'
    cache.mkdir(exist_ok=True)
    groups = data.groupby('segment', sort=False).indices
    fine = np.empty(len(data), np.int8)
    for idx in groups.values():
        g = data.iloc[idx]
        fine[idx] = rule(g.GR, g.DEN, g.depth, g.step.iloc[0])[0]
    np.save(cache/'native_rule.npy', fine)
    spans = []
    for stride, phase in SCENARIOS:
        x = np.lib.format.open_memmap(cache/f'X_s{stride}_p{phase}.npy', mode='w+', dtype=np.float32, shape=(len(data), 33))
        coarse = np.empty_like(fine)
        stability = np.empty(len(data), np.float32)
        for seg, idx in groups.items():
            g = data.iloc[idx]
            z = g.depth.to_numpy()
            step = float(g.step.iloc[0])
            cg = interpolate(g.GR.to_numpy(), z, stride, phase)
            cd = interpolate(g.DEN.to_numpy(), z, stride, phase)
            cl = rule(cg, cd, z, step)[0]
            base_features, actual = features(cg, cd, z, step)
            # No native labels, native inputs, depth, well ID, target, or fold IDs in X.
            x[idx, :26] = base_features
            x[idx, 26:32] = np.eye(6, dtype=np.float32)[cl]
            x[idx, 32] = cl
            coarse[idx] = cl
            risk = np.zeros(len(g), np.float32)
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    if a or b:
                        risk += rule(cg+a, cd+b*.01, z, step)[0] != cl
            stability[idx] = risk/8
            if stride == 2 and phase == 0:
                spans.append({'segment': int(seg), 'native_step_m': step,
                              **{f'feature_span_{j+1}_m': v for j, v in enumerate(actual)}})
        x.flush()
        del x
        np.savez(cache/f'outcomes_s{stride}_p{phase}.npz', coarse=coarse, target=coarse != fine, stability=stability)
        print(f'Prepared stride={stride}, phase={phase}', flush=True)
    pd.DataFrame(spans).to_csv(out/'feature_spans.csv', index=False)
    return fine


def training_rows(target, well_indices, names, cap, rng):
    selected = []
    for w in names:
        ix = well_indices[w]
        pos = ix[target[ix]]
        neg = ix[~target[ix]]
        nneg = min(len(neg), 3*len(pos))
        sel = np.r_[pos, rng.choice(neg, size=nneg, replace=False)]
        # Pre-specified computational cap, identical for both learning objectives.
        if cap and len(sel) > cap:
            sel = rng.choice(sel, size=cap, replace=False)
        selected.append(sel)
    return np.concatenate(selected)


def physical_blocks(part, span=1.0):
    starts, sizes = [], []
    for idx in part.groupby('segment', sort=False).indices.values():
        depth = part.iloc[idx].depth.to_numpy()
        bins = np.floor((depth-depth[0]+1e-8)/span).astype(np.int64)
        loc = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
        end = np.r_[loc[1:], len(idx)]
        starts.extend(idx[loc].tolist())
        sizes.extend((end-loc).tolist())
    return np.asarray(starts), np.asarray(sizes)


def summarize(rows, out, seed):
    table = pd.DataFrame(rows)
    table.to_csv(out/'capture_by_well_phase.csv', index=False)
    pw = table.groupby(['well', 'stride', 'method'], as_index=False).agg(
        capture_rate=('capture_rate', 'mean'), review_yield=('review_yield', 'mean'),
        native_step_m=('native_step_m', 'first'), target_fraction=('target_fraction', 'mean'))
    pw.to_csv(out/'capture_by_well.csv', index=False)
    pivot = pw.pivot(index=['well', 'stride'], columns='method', values='capture_rate').reset_index()
    pivot['change_minus_rule_pp'] = 100*(pivot.change_prediction-pivot.rule_stability)
    pivot['change_minus_category_pp'] = 100*(pivot.change_prediction-pivot.category_support)
    pivot.to_csv(out/'paired_differences.csv', index=False)
    summary = []
    rng = np.random.default_rng(seed)
    for stride, p in pivot.groupby('stride', sort=True):
        r = {'stride': int(stride), 'wells': len(p)}
        for method in METHODS:
            r[method+'_capture_pct'] = float(100*p[method].mean())
        for metric in ['change_minus_rule_pp', 'change_minus_category_pp']:
            v = p[metric].dropna().to_numpy()
            ci = np.quantile(v[rng.integers(0,len(v),size=(2000,len(v)))].mean(axis=1), [.025,.975]) if len(v) else [np.nan,np.nan]
            r[metric] = float(v.mean()) if len(v) else np.nan
            r[metric+'_positive_wells'] = int((v>0).sum())
            r[metric+'_negative_wells'] = int((v<0).sum())
            r[metric+'_boot_low'] = float(ci[0])
            r[metric+'_boot_high'] = float(ci[1])
        summary.append(r)
    pd.DataFrame(summary).to_csv(out/'summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--well-column', default='well')
    ap.add_argument('--depth-column', default='depth')
    ap.add_argument('--gr-column', default='GR')
    ap.add_argument('--den-column', default='DEN')
    ap.add_argument('--separator', default=',')
    ap.add_argument('--depth-to-m', type=float, default=1)
    ap.add_argument('--den-to-gcc', type=float, default=1)
    ap.add_argument('--train-cap-per-well-total', type=int, default=2000)
    ap.add_argument('--max-wells',type=int,default=24)
    ap.add_argument('--dataset-name',default='external')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fold-seed', type=int, default=20261101)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--inspect-only', action='store_true')
    args = ap.parse_args()
    if args.folds < 2 or args.max_wells < 0 or (0 < args.train_cap_per_well_total < len(SCENARIOS)):
        raise ValueError('Need at least two folds, nonnegative well cap, and training cap 0 or at least 11')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        raise FileExistsError('Completed experiment exists; choose an isolated output directory')
    started = time.time()
    budget_test()
    protocol = vars(args) | {'scenarios': SCENARIOS, 'budget': .10,
        'pre_outcome_amendment':'After input-only QC found 49 eligible KGS wells and approximately 130k rows, use all eligible wells instead of an arbitrary 24-well subsample; no rule-change or model result was examined to make this choice.' if args.dataset_name.startswith('KGS') and args.max_wells==0 else None,
        'review_block_m': 1.0, 'rule_perturbations': {'GR_API': [-1,0,1], 'DEN_gcc': [-.01,0,.01], 'omit_joint_zero': True},
        'objectives': ['native_rule_class', 'processed_vs_native_rule_change'],
        'category_score':'1 - P(native_rule_class == processed_rule_class); zero class support if absent from training classes',
        'native_rule_features':'never included; processed-rule state is an available predictor, not the category learning target',
        'test_features': 'processed GR/DEN 26 features + processed-rule onehot + numeric state',
        'well_selection':'first max-wells eligible wells in SHA256(well ID) order, independent of target and model outcomes',
        'training_selection': 'per training well and scenario: all positives + up to 3x negatives, then predetermined random cap; per-well total cap allocated evenly among 11 scenarios before outcomes',
        'mode': 'external-dataset retraining with whole-well holdout; NOT frozen-model transfer',
        'geological_truth_evaluated': False, 'algorithm_version': 'new bounded external task pilot, not an S19 reproduction',
        'code_sha256': {p.name:digest(p) for p in [Path(__file__), Path(__file__).with_name('external_core.py'), Path(__file__).with_name('review_priority.py')]}}
    dump(out/'PROTOCOL.json', protocol)
    data, inventory = load_data(args, out)
    if args.inspect_only:
        print(json.dumps(inventory, indent=2)); return
    wells = data.groupby('well', sort=True).indices
    spacing_rows=[]
    for w,idx in wells.items():
        step=float(data.step.iloc[idx[0]])
        for stride in (2,4,5):
            spacing_rows.append({'well':w,'native_step_m':step,'stride':stride,'spacing_m':step*stride,'rows':len(idx)})
    pd.DataFrame(spacing_rows).to_csv(out/'absolute_spacings.csv',index=False)
    names = np.array(sorted(wells))
    if len(names) < 3:
        raise ValueError('At least three eligible wells required for this pilot')
    nfolds = min(args.folds, len(names))
    shuffled = np.random.default_rng(args.fold_seed).permutation(names)
    foldmap = {w:i%nfolds for i,w in enumerate(shuffled)}
    pd.DataFrame([{'well':w,'fold':foldmap[w],'rows':len(wells[w])} for w in names]).to_csv(out/'folds.csv', index=False)
    fine = build_cache(data, out)
    rows, foldqa, aucrows, selections = [], [], [], []
    cache = out/'cache'
    for fold in range(nfolds):
        trainnames = [w for w in names if foldmap[w] != fold]
        testnames = [w for w in names if foldmap[w] == fold]
        assert set(trainnames).isdisjoint(testnames)
        testidx = np.concatenate([wells[w] for w in testnames])
        rng = np.random.default_rng(args.seed + fold)
        xx, yy, yc = [], [], []
        for scenario_number,(s,p) in enumerate(SCENARIOS):
            outcome = np.load(cache/f'outcomes_s{s}_p{p}.npz')
            total_cap=args.train_cap_per_well_total
            scenario_cap=(total_cap//len(SCENARIOS)+int(scenario_number<total_cap%len(SCENARIOS))) if total_cap else 0
            sel = training_rows(outcome['target'], wells, trainnames, scenario_cap, rng)
            assert not np.isin(sel,testidx).any()
            x = np.load(cache/f'X_s{s}_p{p}.npy', mmap_mode='r')
            xx.append(np.asarray(x[sel])); yy.append(outcome['target'][sel]); yc.append(fine[sel])
            selections.append({'fold':fold,'stride':s,'phase':p,'training_rows':len(sel),'positive_rows':int(outcome['target'][sel].sum()),
                               'index_sha256':hashlib.sha256(sel.astype('<i8').tobytes()).hexdigest()})
        xtrain = np.concatenate(xx); ychange = np.concatenate(yy); ycategory = np.concatenate(yc)
        if args.train_cap_per_well_total:
            assert len(xtrain)<=len(trainnames)*args.train_cap_per_well_total
        del xx,yy,yc
        if len(np.unique(ychange)) != 2:
            raise ValueError('Change model training requires both binary states')
        kw = dict(n_estimators=100,max_depth=10,max_features='sqrt',min_samples_leaf=1,
                  min_samples_split=5,class_weight='balanced_subsample',random_state=args.seed,n_jobs=args.n_jobs)
        mchg = RandomForestClassifier(**kw).fit(xtrain,ychange)
        mcat = RandomForestClassifier(**kw).fit(xtrain,ycategory)
        foldqa.append({'fold':fold,'train_wells':trainnames,'test_wells':testnames,'training_rows':len(xtrain),
                       'test_rows':len(testidx),'well_overlap':0,'positive_training_rows':int(ychange.sum())})
        del xtrain,ychange,ycategory
        print(f'Fit fold {fold+1}/{nfolds}; elapsed {time.time()-started:.1f}s',flush=True)
        classmap = {int(c):i for i,c in enumerate(mcat.classes_)}
        for s,p in SCENARIOS:
            x = np.load(cache/f'X_s{s}_p{p}.npy',mmap_mode='r')
            outcome = np.load(cache/f'outcomes_s{s}_p{p}.npz')
            for w in testnames:
                idx = wells[w]
                xtest = np.asarray(x[idx])
                change = mchg.predict_proba(xtest)[:,int(np.flatnonzero(mchg.classes_==1)[0])]
                probs = mcat.predict_proba(xtest)
                category = np.ones(len(idx))
                coarse = outcome['coarse'][idx]
                for c,col in classmap.items():
                    mask = coarse == c
                    category[mask] = 1-probs[mask,col]
                target = outcome['target'][idx]
                part = data.iloc[idx]
                starts,sizes = physical_blocks(part)
                scores = [outcome['stability'][idx],category,change]
                for method,score in zip(METHODS,scores):
                    means = np.asarray([score[a:a+n].mean() for a,n in zip(starts,sizes)])
                    result = evaluate_blocks(means,starts,sizes,target,.1,well=str(w))
                    assert result['reviewed_rows']==int(np.floor(.1*len(idx)))
                    rows.append({'fold':fold,'well':w,'stride':s,'phase':p,'method':method,
                                 'native_step_m':float(part.step.iloc[0]),'spacing_m':float(part.step.iloc[0]*s),
                                 'rows':len(idx),'target_fraction':float(target.mean()),**result})
                aucrows.append({'fold':fold,'well':w,'stride':s,'phase':p,
                                'auc':float(roc_auc_score(target,change)) if len(np.unique(target))==2 else np.nan})
        del mchg,mcat
    summarize(rows,out,args.seed)
    pd.DataFrame(aucrows).to_csv(out/'auc_by_well_phase.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'training_selection.csv',index=False)
    dump(out/'fold_checks.json',foldqa)
    dump(out/'COMPLETE.json',{'elapsed_seconds':time.time()-started,'inventory':inventory,
         'models_fitted':2*nfolds,'classifiers':['RandomForestClassifier'],'hyperparameter_search':False,
         'seed_count':1,'frozen_external_transfer':False,'fold_well_overlap':0,
         'prediction_feature_count':33,'strict_10pct_budget':True,'phases':11,
         'test_target_used_in_ranking':False,'versions':{m:importlib.metadata.version(m) for m in ['numpy','pandas','scikit-learn']}})


if __name__=='__main__':
    main()
