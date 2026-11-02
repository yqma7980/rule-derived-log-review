"""Reproduce S12 audits and the fixed review-prioritization experiment locally."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

S12 = Path(__file__).resolve().parents[1]
ROOT = S12.parents[1]
S6 = ROOT / "NRR投稿" / "stage6_submission_readiness"
sys.path.insert(0, str(S12 / "code"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import audit_core as core
import historical_rule
import legacy_common as common

SCHEMES = ("LEAVE_ONE_WELL_OUT", "LEAVE_ONE_SPATIAL_BLOCK_OUT")
SCENARIOS = [(1, 0)] + [(stride, phase) for stride in (2, 4, 5) for phase in range(stride)]
RESULTS = S12 / "results"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_data():
    source = ROOT / "IJCG投稿" / "data" / "measurement_data.csv"
    raw, _ = common.load_data(source, apply_qc=False)
    data, rejected = common.load_data(source, apply_qc=True)
    blocks = pd.read_csv(ROOT / "IJCG投稿" / "outputs" / "geology_aware_subfacies_validation_v1" / "well_spatial_blocks_v1.csv",
                         usecols=["well_id", "spatial_block"]).rename(columns={"well_id": "well_name"})
    data = data.merge(blocks, on="well_name", how="left", validate="many_to_one", sort=False)
    assert len(raw) == 101639 and len(data) == 101615 and len(rejected) == 24
    assert data.spatial_block.notna().all()
    for scheme in SCHEMES:
        with np.load(S6 / "cache" / f"{scheme.lower()}__rf_gr_den_group_calibrated.npz") as cache:
            np.testing.assert_array_equal(data.source_row.to_numpy(), cache["source_row"])
    return raw, data


def saved_primary(data, scheme):
    path = ROOT / "IJCG投稿" / "results" / "nrr_stage1_stage2" / "stage1_cache" / f"{scheme.lower()}__rf_multiscale__no_rt_gr_den.npz"
    with np.load(path) as cache:
        np.testing.assert_array_equal(data.source_row.to_numpy(), cache["source_row"])
        return cache["prediction"].astype(int)


def static_audit(raw, data):
    marker = RESULTS / "static_audit.json"
    if marker.exists():
        raise RuntimeError("Static audit already exists")
    rule_rows, diagnostic_rows = [], []
    for mode, frame, cols in (("ORIGINAL_COMPLETE_WELL", raw, ["well_name"]),
                               ("QC_CONTIGUOUS_SEGMENT", data, ["well_name", "segment_id"])):
        for group, positions in frame.groupby(cols, sort=False).indices.items():
            local = frame.iloc[positions]
            archived = historical_rule.apply_operational_rule(local[["depth", "GR", "DEN"]], enforce_nominal_grid=False).Subfacies.to_numpy() - 1
            simplified, diag = core.simplified_rule(local.GR, local.DEN, True)
            delta = int(np.count_nonzero(archived != simplified))
            assert delta == 0
            source_delta = int(np.count_nonzero(archived != local.facies_id.to_numpy()))
            if mode == "ORIGINAL_COMPLETE_WELL":
                assert source_delta == 0
            rule_rows.append({"mode": mode, "group": str(group), "rows": len(local), "simplification_disagreements": delta, "saved_label_disagreements": source_delta})
            for label in range(6):
                diagnostic_rows.append({"mode": mode, "group": str(group), "label": common.ID_TO_CODE[label],
                                        "rows": int(np.count_nonzero(simplified == label)),
                                        "fine_context_changed_to_SS": int(np.count_nonzero(diag["fine_context_changes_default"] & (simplified == label)))})
    pd.DataFrame(rule_rows).to_csv(RESULTS / "rule_equivalence.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(RESULTS / "rule_branch_counts.csv", index=False)
    counts, match_rows, iou_rows = [], [], []
    for scheme in SCHEMES:
        pred = saved_primary(data, scheme)
        for (well, segment), idx in data.groupby(["well_name", "segment_id"], sort=False).indices.items():
            part = data.iloc[idx]
            y, p = part.facies_id.to_numpy(), pred[idx]
            a, b = core.transitions(part.depth, y), core.transitions(part.depth, p)
            for direction, scores in (("reference_to_prediction", core.directional_run_iou(y, p)),
                                       ("prediction_to_reference", core.directional_run_iou(p, y))):
                iou_rows.append({"scheme": scheme, "well": well, "segment": int(segment), "direction": direction, "runs": len(scores), "sum_iou": float(scores.sum())})
            for tol in (.05, .10, .25, .50):
                for aware in (False, True):
                    for method in ("greedy", "optimal"):
                        matches = core.boundary_matches(a, b, tol, aware, method)
                        counts.append({"scheme": scheme, "well": well, "segment": int(segment), "tolerance_m": tol,
                                       "metric_type": "TRANSITION_AWARE" if aware else "CLASS_AGNOSTIC", "method": method,
                                       "observed": len(a), "predicted": len(b), "matched": len(matches), "offset_sum_m": sum(x[2] for x in matches)})
                        for i, j, offset in matches:
                            match_rows.append({"scheme": scheme, "well": well, "segment": int(segment), "tolerance_m": tol,
                                               "metric_type": "TRANSITION_AWARE" if aware else "CLASS_AGNOSTIC", "method": method,
                                               "observed_index": i, "predicted_index": j, "offset_m": offset})
        print(f"Static boundaries and run IoU completed: {scheme}", flush=True)
    counts = pd.DataFrame(counts)
    counts.to_csv(RESULTS / "boundary_segment_counts.csv", index=False)
    pd.DataFrame(match_rows).to_csv(S12 / "private" / "boundary_matches.csv.gz", index=False, compression="gzip")
    wells = counts.groupby(["scheme", "well", "tolerance_m", "metric_type", "method"], as_index=False)[["observed", "predicted", "matched", "offset_sum_m"]].sum()
    wells["f1"] = 2 * wells.matched / (wells.observed + wells.predicted)
    wells.to_csv(RESULTS / "boundary_per_well.csv", index=False)
    summary = wells.groupby(["scheme", "tolerance_m", "metric_type", "method"], as_index=False).agg(
        mean_well_f1=("f1", "mean"), min_well_f1=("f1", "min"), max_well_f1=("f1", "max"),
        observed=("observed", "sum"), predicted=("predicted", "sum"), matched=("matched", "sum"), offset_sum_m=("offset_sum_m", "sum"))
    summary["pooled_f1"] = 2 * summary.matched / (summary.observed + summary.predicted)
    summary.to_csv(RESULTS / "boundary_summary.csv", index=False)
    historic = pd.read_csv(S6 / "evidence" / "boundary_tolerance_sensitivity.csv")
    for row in summary.loc[summary.method == "greedy"].itertuples():
        old = historic.loc[(historic.validation_scheme == row.scheme) & (historic.metric_type == row.metric_type) & np.isclose(historic.tolerance_m, row.tolerance_m)].iloc[0]
        assert row.matched == old.matched_boundaries
        assert abs(row.mean_well_f1 - old.mean_per_well_f1) < 1e-12
    pivot = wells.pivot(index=["scheme", "well", "tolerance_m", "metric_type"], columns="method", values="matched")
    assert (pivot.optimal >= pivot.greedy).all()
    iou = pd.DataFrame(iou_rows).groupby(["scheme", "well", "direction"], as_index=False)[["runs", "sum_iou"]].sum()
    iou["mean_run_iou"] = iou.sum_iou / iou.runs
    iou.to_csv(RESULTS / "run_iou_per_well.csv", index=False)
    fold = pd.read_csv(S6 / "evidence" / "calibration_fold_audit_gr_den.csv")
    fold["fallback_used"] = ~fold.calibration_target_met
    assert (fold.fit_test_well_overlap == 0).all() and (fold.calibration_test_well_overlap == 0).all()
    fold.to_csv(RESULTS / "calibration_fold_details.csv", index=False)
    pd.crosstab(data.well_name, data.facies_id).rename(columns=common.ID_TO_CODE).to_csv(RESULTS / "class_support_per_well.csv")
    evidence = {"raw_rows": len(raw), "qc_rows": len(data), "rule_equivalence_pass": True,
                "original_replay_mismatches": sum(x["saved_label_disagreements"] for x in rule_rows if x["mode"] == "ORIGINAL_COMPLETE_WELL"),
                "qc_segment_replay_vs_saved_label_mismatches": sum(x["saved_label_disagreements"] for x in rule_rows if x["mode"] == "QC_CONTIGUOUS_SEGMENT"),
                "greedy_archive_recovery_pass": True, "optimal_never_lower_cardinality": True,
                "calibration_target_failures": fold.groupby("validation_scheme").outer_test_error_target_met.apply(lambda x: int((~x).sum())).to_dict(),
                "calibration_fallbacks": int(fold.fallback_used.sum())}
    write_json(marker, evidence)


def degrade(data, stride, phase):
    out = data.copy()
    if stride > 1:
        for idx in out.groupby(["well_name", "segment_id"], sort=False).indices.values():
            part = out.iloc[idx]
            anchors = np.unique(np.r_[0, np.arange(phase, len(idx), stride), len(idx)-1]).astype(int)
            for log in ("GR", "DEN"):
                out.loc[idx, log] = np.interp(part.depth, part.depth.iloc[anchors], part[log].iloc[anchors])
    return out


def triage(data):
    if (RESULTS / "triage_decision.json").exists():
        raise RuntimeError("Triage outcomes already exist")
    assert (RESULTS / "static_audit.json").exists()
    folds = pd.read_csv(RESULTS / "calibration_fold_details.csv")
    native, features = common.add_window_features(data)
    features = [x for x in features["raw_through_w100"] if x.startswith("GR") or x.startswith("DEN")]
    assert len(features) == 26
    fine_rule = core.apply_grouped_rule(data)
    degraded, risks, targets = {}, {}, {}
    started = time.perf_counter()
    cost = []
    for stride, phase in SCENARIOS:
        t = time.perf_counter()
        frame = degrade(data, stride, phase)
        labels = core.apply_grouped_rule(frame)
        targets[stride, phase] = labels != fine_rule
        d = {"THRESHOLD_DISTANCE": core.threshold_risk(frame)}
        cost.append({"component": "threshold_score", "stride": stride, "phase": phase, "seconds": time.perf_counter()-t})
        for amplitude in (.5, 1., 2.):
            t = time.perf_counter()
            d[f"STABILITY_{amplitude:g}"] = core.perturbation_risk(frame, amplitude, labels)
            cost.append({"component": f"stability_{amplitude:g}", "stride": stride, "phase": phase, "seconds": time.perf_counter()-t})
        augmented, _ = common.add_window_features(frame)
        degraded[stride, phase] = (augmented[features], labels)
        risks[stride, phase] = d
        print(f"Prepared scoring inputs: stride={stride} phase={phase}, target not used in scoring", flush=True)
    result_rows, cache_checks, random_rows = [], [], []
    for scheme in SCHEMES:
        scores_by_scenario = {key: np.full(len(data), np.nan) for key in SCENARIOS}
        full_probs = {key: np.full((len(data), 6), np.nan, np.float32) for key in SCENARIOS}
        with np.load(S6 / "cache" / f"{scheme.lower()}__rf_gr_den_group_calibrated.npz") as z:
            archived_native = z["calibrated_probability"].copy()
            archived_raw = z["raw_probability"].copy()
        for fold in folds.loc[folds.validation_scheme == scheme].itertuples():
            fit_idx = np.flatnonzero(data.well_name.isin(fold.fit_wells.split(";")))
            test_idx = np.flatnonzero(data.well_name.isin(fold.test_wells.split(";")))
            assert len(fit_idx) == fold.fit_rows and len(test_idx) == fold.test_rows
            assert not set(data.iloc[fit_idx].well_name) & set(data.iloc[test_idx].well_name)
            assert fold.calibration_well not in set(data.iloc[fit_idx].well_name)
            t = time.perf_counter()
            model = RandomForestClassifier(n_estimators=100, max_depth=10, max_features="sqrt", min_samples_leaf=1,
                min_samples_split=5, class_weight="balanced_subsample", random_state=42, n_jobs=4)
            model.fit(native.iloc[fit_idx][features], data.facies_id.iloc[fit_idx])
            fit_time = time.perf_counter() - t
            cost.append({"component": "RF_fit", "scheme": scheme, "fold": fold.fold, "seconds": fit_time})
            for stride, phase in SCENARIOS:
                t = time.perf_counter()
                matrix, rule_class = degraded[stride, phase]
                p = np.zeros((len(test_idx), 6))
                p[:, model.classes_.astype(int)] = model.predict_proba(matrix.iloc[test_idx])
                q = core.temperature_scale(p, fold.temperature)
                if stride == 1:
                    raw_delta = float(np.max(np.abs(p - archived_raw[test_idx])))
                    cal_delta = float(np.max(np.abs(q - archived_native[test_idx])))
                    labels_equal = bool(np.array_equal(q.argmax(axis=1), archived_native[test_idx].argmax(axis=1)))
                    cache_checks.append({"scheme": scheme, "fold": int(fold.fold), "raw_max_abs_diff": raw_delta,
                                         "calibrated_max_abs_diff": cal_delta, "labels_equal": labels_equal})
                    if raw_delta > 2e-6 or cal_delta > 2e-6 or not labels_equal:
                        pd.DataFrame(cache_checks).to_csv(RESULTS / "native_cache_recovery_FAILED.csv", index=False)
                        raise RuntimeError("Native-cache recovery failed; no new result claims permitted")
                scores_by_scenario[stride, phase][test_idx] = 1-q[np.arange(len(test_idx)), rule_class[test_idx]]
                full_probs[stride, phase][test_idx] = q.astype(np.float32)
                cost.append({"component": "RF_predict_score", "scheme": scheme, "fold": fold.fold, "stride": stride, "phase": phase, "seconds": time.perf_counter()-t})
            print(f"Native recovery passed and degraded scoring completed: {scheme} fold {fold.fold}", flush=True)
        for stride, phase in SCENARIOS:
            assert np.isfinite(scores_by_scenario[stride, phase]).all()
            all_scores = {**risks[stride, phase], "RF_RULE_SUPPORT": scores_by_scenario[stride, phase]}
            np.savez_compressed(S12 / "private" / f"triage_{scheme.lower()}_s{stride}_p{phase}.npz", source_row=data.source_row.to_numpy(), probability=full_probs[stride, phase], rule_class=degraded[stride, phase][1], target=targets[stride, phase])
            for well, idx in data.groupby("well_name", sort=True).indices.items():
                part = data.iloc[idx]
                meta = {"scheme": scheme, "well": well, "block": str(part.spatial_block.iloc[0]), "stride": stride, "phase": phase, "spacing_m": stride*.05}
                for budget in (.05, .10, .20):
                    for method, score in all_scores.items():
                        stats = core.review_scores(score[idx], targets[stride, phase][idx], part.segment_id, well, budget)
                        result_rows.append({**meta, "budget": budget, "method": method, **stats})
                    random_stats = []
                    for seed in range(100):
                        stats = core.review_scores(np.zeros(len(idx)), targets[stride, phase][idx], part.segment_id, well, budget, random_seed=20260907+seed)
                        random_stats.append(stats)
                        random_rows.append({**meta, "budget": budget, "seed": seed, **stats})
                    random_df = pd.DataFrame(random_stats)
                    result_rows.append({**meta, "budget": budget, "method": "RANDOM", **random_df.mean().to_dict()})
            print(f"Budget evaluation completed: {scheme}, stride={stride} phase={phase}", flush=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(RESULTS / "triage_per_well_phase.csv", index=False)
    pd.DataFrame(random_rows).to_csv(S12 / "private" / "triage_random_replicates.csv.gz", index=False, compression="gzip")
    pd.DataFrame(cache_checks).to_csv(RESULTS / "native_cache_recovery.csv", index=False)
    pd.DataFrame(cost).to_csv(RESULTS / "computational_cost.csv", index=False)
    key = ["scheme", "well", "block", "stride", "spacing_m", "budget", "method"]
    per_well = results.groupby(key, as_index=False).mean(numeric_only=True).drop(columns="phase")
    per_well.to_csv(RESULTS / "triage_per_well_phase_mean.csv", index=False)
    groups = []
    for scheme in SCHEMES:
        subset = per_well.loc[per_well.scheme == scheme].copy()
        subset["test_group"] = subset.well if scheme == SCHEMES[0] else subset.block
        subset = subset.groupby(["scheme", "test_group", "stride", "spacing_m", "budget", "method"], as_index=False).mean(numeric_only=True)
        groups.append(subset)
    groups = pd.concat(groups, ignore_index=True)
    groups.to_csv(RESULTS / "triage_test_group_phase_mean.csv", index=False)
    summary = groups.groupby(["scheme", "stride", "spacing_m", "budget", "method"], as_index=False).agg(
        group_count=("test_group", "count"), mean_capture=("capture", "mean"), min_capture=("capture", "min"), max_capture=("capture", "max"),
        mean_residual_error=("residual_error", "mean"), mean_reviewed_length_m=("reviewed_length_m", "mean"), mean_intervals=("merged_intervals", "mean"))
    summary.to_csv(RESULTS / "triage_summary.csv", index=False)
    criteria = []
    primary = groups.loc[(groups.stride > 1) & np.isclose(groups.budget, .10)]
    for (scheme, stride), group in primary.groupby(["scheme", "stride"]):
        pivot = group.pivot(index="test_group", columns="method", values="capture")
        for baseline in ("THRESHOLD_DISTANCE", "STABILITY_1"):
            delta = pivot.RF_RULE_SUPPORT - pivot[baseline]
            criteria.append({"scheme": scheme, "stride": int(stride), "baseline": baseline, "groups": len(delta),
                             "mean_capture_gain": float(delta.mean()), "positive_groups": int((delta > 0).sum()),
                             "passed": bool(delta.mean() > 0 and (delta > 0).sum() > len(delta)/2)})
    pd.DataFrame(criteria).to_csv(RESULTS / "rf_incremental_utility_criteria.csv", index=False)
    native = results.loc[results.stride == 1]
    assert (native.target_rows == 0).all() and native.capture.isna().all()
    write_json(RESULTS / "triage_decision.json", {"finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter()-started,
        "native_cache_recovery_pass": True, "no_degradation_control_pass": True,
        "all_preregistered_descriptive_criteria_pass": all(x["passed"] for x in criteria),
        "passed_comparisons": sum(x["passed"] for x in criteria), "total_comparisons": len(criteria),
        "interpretation": "Exploratory retrospective operational discrepancy triage, not blind geological or human productivity validation"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["static", "triage"])
    args = parser.parse_args()
    (S12 / "private").mkdir(exist_ok=True)
    protocol = S12 / "GAS_S12_PROTOCOL.md"
    freeze_path = S12 / "provenance" / "analysis_protocol_freeze.json"
    if not freeze_path.exists():
        paths = [protocol, Path(__file__), *sorted((S12 / "code").glob("*.py")), *sorted((S12 / "tests").glob("*.py"))]
        write_json(freeze_path, {"created_utc": datetime.now(timezone.utc).isoformat(), "files": {str(p.relative_to(S12)): sha(p) for p in paths},
                                "environment": {x: importlib.metadata.version(x) for x in ("numpy", "pandas", "scipy", "scikit-learn")}})
    else:
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        for name, digest in frozen["files"].items():
            if sha(S12 / name) != digest:
                raise RuntimeError(f"Analysis code/protocol changed after freeze: {name}")
    raw, data = load_data()
    static_audit(raw, data) if args.mode == "static" else triage(data)


if __name__ == "__main__":
    main()
