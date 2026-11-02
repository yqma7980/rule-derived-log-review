"""Matched class/change models with per-point and per-block exports.

Predictions are indexed by validation design, stride/phase, and method.
Each test point is written exactly once and retains its outer-fold identity.
Spatial-block summaries average wells within blocks, then weight blocks equally.
Budget ties use the actual last selected block and exact score equality.
The fixed training rows, 33 features, sampling ratio, and model settings are
those used by the independently replayed main comparison.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = os.environ["M001_CONTROLLED_ROOT"]
S12 = ROOT + "/GAS投稿/stage12_substantive_revision_20260907"
sys.path.insert(0, os.environ["M001_RECOVERED_S12"] + "/code")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import audit_core as core            # noqa: E402
import legacy_common as common       # noqa: E402
import importlib.util
spec = importlib.util.spec_from_file_location(
    "diag", os.environ["M001_DIAG_HELPER"])
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

OUT = os.environ["M001_RECOVERY_OUTPUT"]
os.makedirs(OUT, exist_ok=True)
SCHEMES = ("LEAVE_ONE_WELL_OUT", "LEAVE_ONE_SPATIAL_BLOCK_OUT")
SCENARIOS = [(s, p) for s in (2, 4, 5) for p in range(s)]
BUDGETS = (0.05, 0.10, 0.20)
NEG_RATIO = 3
SEED = 42
LEARN = ("CATEGORY_SAME_ROWS", "CHANGE_SAME_ROWS")
METHODS = ("STABILITY_1", "STABILITY_1_FROZEN", "THRESHOLD_DISTANCE",
           "RF_SUPPORT_CAL", "RF_SUPPORT_RAW", "CATEGORY_SAME_ROWS", "CHANGE_SAME_ROWS", "RANDOM")


def build():
    data, _ = common.load_data(ROOT + "/IJCG投稿/data/measurement_data.csv", apply_qc=True)
    blocks = pd.read_csv(ROOT + "/IJCG投稿/outputs/geology_aware_subfacies_validation_v1/well_spatial_blocks_v1.csv",
                         usecols=["well_id", "spatial_block"]).rename(columns={"well_id": "well_name"})
    data = data.merge(blocks, on="well_name", how="left", validate="many_to_one", sort=False)
    fine = core.apply_grouped_rule(data)
    packs = {}
    for stride, phase in SCENARIOS:
        frame = diag.degrade(data, stride, phase)
        labels = core.apply_grouped_rule(frame)
        aug, groups = common.add_window_features(frame)
        feats = [c for c in groups["raw_through_w100"] if c.startswith(("GR", "DEN"))]
        if len(feats) != 26:
            raise RuntimeError("baseline feature count changed: %d" % len(feats))
        X = aug[feats].to_numpy(dtype=np.float32)
        onehot = np.zeros((len(data), 6), np.float32)
        onehot[np.arange(len(data)), labels] = 1.0
        X = np.hstack([X, onehot, labels.reshape(-1, 1).astype(np.float32)])
        if X.shape[1] != 33:
            raise RuntimeError("model input width changed: %d" % X.shape[1])
        gates = diag.baseline_gates(frame)
        r_full, r_frozen = diag.risks(frame, labels, gates)
        packs[stride, phase] = {"X": X, "labels": labels, "target": labels != fine,
                                "STABILITY_1": r_full, "STABILITY_1_FROZEN": r_frozen,
                                "THRESHOLD_DISTANCE": core.threshold_risk(frame)}
        print("prepared %d/%d" % (stride, phase), flush=True)
    return data, fine, packs


def main():
    data, fine, packs = build()
    folds = pd.read_csv(S12 + "/results/calibration_fold_details.csv")
    n = len(data)

    # Fitting, calibration, and test wells must be pairwise disjoint.
    for row in folds.itertuples():
        fit = set(str(row.fit_wells).split(";"))
        tst = set(str(row.test_wells).split(";"))
        cal = {str(row.calibration_well)}
        if fit & tst or fit & cal or cal & tst:
            raise RuntimeError("fold %s %d well roles overlap" % (row.validation_scheme, row.fold))

    prob = {(sc, k, m): np.full(n, np.nan, np.float32)
            for sc in SCHEMES for k in SCENARIOS for m in LEARN}
    fold_id = {sc: np.full(n, -1, np.int16) for sc in SCHEMES}
    writes = {(sc, k, m): np.zeros(n, np.int16) for sc in SCHEMES for k in SCENARIOS for m in LEARN}

    for scheme in SCHEMES:
        for fold in folds[folds.validation_scheme == scheme].itertuples():
            t0 = time.perf_counter()
            fit_mask = data.well_name.isin(str(fold.fit_wells).split(";")).to_numpy()
            test_mask = data.well_name.isin(str(fold.test_wells).split(";")).to_numpy()
            rng = np.random.default_rng(SEED)
            Xs, y_chg, y_cat = [], [], []
            for key, pk in packs.items():
                idx = np.flatnonzero(fit_mask)
                pos = idx[pk["target"][idx]]
                neg = idx[~pk["target"][idx]]
                neg = rng.choice(neg, size=min(len(neg), NEG_RATIO * len(pos)), replace=False)
                sel = np.r_[pos, neg]
                Xs.append(pk["X"][sel])
                y_chg.append(pk["target"][sel].astype(np.int8))
                y_cat.append(fine[sel].astype(np.int8))
            X = np.vstack(Xs)
            y_chg = np.concatenate(y_chg)
            y_cat = np.concatenate(y_cat)
            kw = dict(n_estimators=100, max_depth=10, max_features="sqrt", min_samples_leaf=1,
                      min_samples_split=5, random_state=SEED, n_jobs=4, class_weight="balanced_subsample")
            m_chg = RandomForestClassifier(**kw).fit(X, y_chg)
            m_cat = RandomForestClassifier(**kw).fit(X, y_cat)
            cls = m_cat.classes_
            for key, pk in packs.items():
                te = np.flatnonzero(test_mask)
                prob[scheme, key, "CHANGE_SAME_ROWS"][te] = m_chg.predict_proba(pk["X"][te])[:, 1]
                P = m_cat.predict_proba(pk["X"][te])
                col = np.searchsorted(cls, pk["labels"][te])
                prob[scheme, key, "CATEGORY_SAME_ROWS"][te] = 1.0 - P[np.arange(len(te)), col]
                for m in LEARN:
                    writes[scheme, key, m][te] += 1
                fold_id[scheme][te] = fold.fold
            print("[%s fold %d] fit %d rows  %.0fs" % (scheme, fold.fold, len(X), time.perf_counter() - t0), flush=True)

    # Require one finite prediction per test point.
    for sc in SCHEMES:
        if not (writes[sc, SCENARIOS[0], LEARN[0]] == 1).all():
            raise RuntimeError("%s: some points written !=1 times" % sc)
        if not (fold_id[sc] >= 0).all():
            raise RuntimeError("%s: unassigned fold" % sc)
    for key in prob:
        if not np.isfinite(prob[key]).all():
            raise RuntimeError("non-finite prediction in %s" % (key,))
    # Diagnose pointwise differences between validation designs.
    diff_report = {}
    for k in SCENARIOS:
        for m in LEARN:
            d = np.abs(prob["LEAVE_ONE_WELL_OUT", k, m] - prob["LEAVE_ONE_SPATIAL_BLOCK_OUT", k, m])
            diff_report["%d/%d|%s" % (k[0], k[1], m)] = {"max_abs_diff": float(d.max()),
                                                         "identical_fraction": float((d == 0).mean())}
    identical = [k for k, v in diff_report.items() if v["max_abs_diff"] == 0]
    print("schemes identical in %d/%d series" % (len(identical), len(diff_report)), flush=True)

    for (stride, phase) in SCENARIOS:
        for scheme in SCHEMES:
            np.savez_compressed(
                OUT + f"/points_{scheme.lower()}_s{stride}_p{phase}.npz",
                source_row=data.source_row.to_numpy(),
                rule_class=packs[stride, phase]["labels"].astype(np.int16),
                target=packs[stride, phase]["target"],
                change_score=prob[scheme, (stride, phase), "CHANGE_SAME_ROWS"],
                category_score=prob[scheme, (stride, phase), "CATEGORY_SAME_ROWS"],
                fold_id=fold_id[scheme])

    rows, ties = [], []
    for (stride, phase), pk in packs.items():
        for scheme in SCHEMES:
            z = np.load(S12 + f"/private/triage_{scheme.lower()}_s{stride}_p{phase}.npz")
            q = z["probability"].astype(np.float64)
            well_t = {}
            for r in folds[folds.validation_scheme == scheme].itertuples():
                for w in str(r.test_wells).split(";"):
                    well_t[w.strip()] = float(r.temperature)
            T = np.array([well_t[w] for w in data.well_name])
            p = q ** T[:, None]
            p = p / p.sum(axis=1, keepdims=True)
            s = np.column_stack([
                pk["STABILITY_1"], pk["STABILITY_1_FROZEN"], pk["THRESHOLD_DISTANCE"],
                1.0 - q[np.arange(n), pk["labels"]], 1.0 - p[np.arange(n), pk["labels"]],
                prob[scheme, (stride, phase), "CATEGORY_SAME_ROWS"],
                prob[scheme, (stride, phase), "CHANGE_SAME_ROWS"],
                np.zeros(n)])
            bstart, bsize, bwell, bblock, bscore, bsel = [], [], [], [], [], []
            for w, idx in data.groupby("well_name", sort=True).indices.items():
                part = data.iloc[idx]
                blocks = core.review_blocks(part.segment_id.to_numpy())
                tgt = pk["target"][idx]
                bs = np.array([[s[idx[b], m].mean() for b in blocks] for m in range(len(METHODS))])
                for m_i, name in enumerate(METHODS):
                    order = np.lexsort((core.stable_ties(w, len(blocks)), -bs[m_i]))
                    for budget in BUDGETS:
                        sel = core.select_review(blocks, order, int(np.floor(budget * len(idx))), len(idx))
                        cap = (sel & tgt).sum() / tgt.sum() if tgt.sum() else np.nan
                        rows.append({"scheme": scheme, "stride": stride, "spacing_m": stride * .05, "phase": phase,
                                     "well": w, "block": str(part.spatial_block.iloc[0]), "method": name,
                                     "budget": budget, "capture": cap, "target_rows": int(tgt.sum()),
                                     "reviewed_rows": int(sel.sum()), "captured_rows": int((sel & tgt).sum()),
                                     "test_fold": int(fold_id[scheme][idx[0]])})
                    if name in ("STABILITY_1", "CHANGE_SAME_ROWS"):
                        need = int(np.floor(0.10 * len(idx)))
                        acc = 0
                        for k in range(len(order)):
                            acc += len(blocks[order[k]])
                            if acc >= need:
                                break
                        last = order[k]                      # Actual last selected block.
                        ties.append({"scheme": scheme, "stride": stride, "phase": phase, "well": w, "method": name,
                                     "blocks": len(blocks), "selected_blocks": k + 1,
                                     "budget_rows": need, "completed_rows": int(acc),
                                     "tail_block_rows": int(len(blocks[last])),
                                     "tail_block_used": int(min(len(blocks[last]), max(need - (acc - len(blocks[last])), 0))),
                                     "tied_at_boundary": int(np.count_nonzero(bs[m_i] == bs[m_i][last]))})
                for b in range(len(blocks)):
                    bstart.append(int(idx[blocks[b][0]]))
                    bsize.append(int(len(blocks[b])))
                    bwell.append(w)
                    bblock.append(str(part.spatial_block.iloc[0]))
                bscore.append(bs.T)
                bsel.append(np.array([core.select_review(
                    blocks, np.lexsort((core.stable_ties(w, len(blocks)), -bs[m])),
                    int(np.floor(0.10 * len(idx))), len(idx)).mean() for m in range(len(METHODS))]))
            np.savez_compressed(OUT + f"/blocks_{scheme.lower()}_s{stride}_p{phase}.npz",
                                start=np.array(bstart), size=np.array(bsize), well=np.array(bwell),
                                spatial_block=np.array(bblock),
                                score=np.vstack(bscore), selected_fraction_at_010=np.vstack(bsel),
                                methods=np.array(METHODS))
            print("scored %d/%d %s" % (stride, phase, scheme), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT + "/per_well_budget_capture.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ties).to_csv(OUT + "/tie_diagnostics.csv", index=False, encoding="utf-8-sig")

    # LOOWO weights wells equally; LOSBO averages wells, then weights blocks equally.
    def aggregate(frame):
        f = frame.copy()
        f["test_group"] = np.where(f.scheme == "LEAVE_ONE_WELL_OUT", f.well, f.block)
        g = f.groupby(["scheme", "spacing_m", "budget", "method", "test_group"], as_index=False).capture.mean()
        return g.groupby(["scheme", "spacing_m", "budget", "method"], as_index=False).capture.mean()

    s = aggregate(df)
    s["capture"] = (s.capture * 100).round(3)
    s.to_csv(OUT + "/per_design_budget_summary.csv", index=False, encoding="utf-8-sig")
    json.dump({"rows": len(df), "ties": len(ties), "schemes_identical": identical,
               "scheme_diff": diff_report},
              open(OUT + "/_done.json", "w"), indent=2)
    pd.set_option("display.width", 260)
    print(s.pivot_table(index=["budget", "spacing_m"], columns=["scheme", "method"], values="capture").to_string())


if __name__ == "__main__":
    main()
