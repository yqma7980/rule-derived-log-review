from __future__ import annotations

import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import f1_score, recall_score


HERE = Path(__file__).resolve().parent
STAGE_DIR = HERE.parent
REPO = STAGE_DIR.parents[1]
IJCG_ANALYSIS = REPO / "IJCG投稿" / "analysis"
if str(IJCG_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(IJCG_ANALYSIS))

from common import ALL_LABELS, ID_TO_CODE, bootstrap_mean_ci  # noqa: E402
from nrr_stage_common import (  # noqa: E402
    SCHEMES,
    apply_temperature,
    build_analysis_data,
    build_manifest,
    cache_path,
    choose_calibration_well,
    degrade_logs_by_stride,
    fit_temperature,
    frozen_random_forest,
    load_valid_prediction_cache,
    match_boundary_offsets,
    probability_metrics,
    reliability_rows,
    run_intervals,
    sample_support_weights,
    source_hashes,
    validation_splits,
)
from run_nrr_stage1 import minority_labels  # noqa: E402
from run_nrr_stage2 import (  # noqa: E402
    CONFIDENCE_GRID,
    MIN_CALIBRATION_COVERAGE,
    TARGET_ACCEPTED_ERROR,
    choose_confidence_threshold,
    confidence_result,
    observed_run_matches,
    target_interval_facies_occupancy,
    thickness_metrics,
)


EVIDENCE_DIR = STAGE_DIR / "evidence"
CACHE_DIR = STAGE_DIR / "cache"
SOURCE_STAGE1 = REPO / "IJCG投稿" / "results" / "nrr_stage1_stage2" / "stage1"
BOUNDARY_TOLERANCE_M = 0.5
THIN_RUN_THRESHOLD_M = 0.5
DOWNSAMPLE_STRIDES = (1, 2, 4, 5)
PRIMARY_SCENARIO = "NO_RT_GR_DEN"
PRIMARY_LABEL = "GR_DEN_MULTISCALE"


def load_legacy_rule_module():
    path = (
        REPO
        / "论文中代码数据"
        / "10口井统一曲线形质划分岩相结果（最终版划分结果）"
        / "代码.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_operational_rule", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy rule: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_rule_predictions(frame: pd.DataFrame, rule) -> np.ndarray:
    """Re-execute the historical rule within each QC-contiguous segment."""
    prediction = np.empty(len(frame), dtype=int)
    groups = frame.groupby(["well_name", "segment_id"], sort=False).indices
    for positions in groups.values():
        positions = np.asarray(positions, dtype=int)
        local = frame.iloc[positions][["GR", "DEN"]].reset_index(drop=True).copy()
        local["Lithology"] = local.apply(
            rule.classify_lithology,
            axis=1,
            thresholds=rule.LITHOLOGY_THRESHOLDS,
        )
        local = rule.calculate_layer_thickness(local, depth_step=0.05)
        local = rule.identify_cycles(local)
        local = rule.classify_subfacies_window(
            local,
            window_size=rule.SUBFACIES_CONFIG["window_size"],
        )
        prediction[positions] = local["Subfacies"].to_numpy(int) - 1
    return prediction


def summarize_model_selection(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SOURCE_STAGE1 / "strict_experiment_summary.csv")
    per_well = pd.read_csv(SOURCE_STAGE1 / "strict_per_well_metrics.csv")
    keep = summary.loc[
        summary["model"].eq("RF_MULTISCALE")
        & summary["scenario"].isin(["FULL_RT_GR_DEN", PRIMARY_SCENARIO])
    ].copy()

    rows = []
    paired_rows = []
    for scheme in SCHEMES:
        subset = per_well.loc[
            per_well["validation_scheme"].eq(scheme)
            & per_well["model"].eq("RF_MULTISCALE")
            & per_well["scenario"].isin(["FULL_RT_GR_DEN", PRIMARY_SCENARIO])
        ]
        pivot = subset.pivot(
            index="well_name", columns="scenario", values="macro_f1_present"
        ).dropna()
        pivot["gr_den_minus_rt_gr_den"] = (
            pivot[PRIMARY_SCENARIO] - pivot["FULL_RT_GR_DEN"]
        )
        low, high = bootstrap_mean_ci(pivot["gr_den_minus_rt_gr_den"].to_numpy(float))
        rows.append(
            {
                "validation_scheme": scheme,
                "paired_wells": int(len(pivot)),
                "wells_gr_den_better": int(
                    pivot["gr_den_minus_rt_gr_den"].gt(0).sum()
                ),
                "wells_tied": int(pivot["gr_den_minus_rt_gr_den"].eq(0).sum()),
                "mean_well_macro_f1_delta": float(
                    pivot["gr_den_minus_rt_gr_den"].mean()
                ),
                "delta_ci_low": low,
                "delta_ci_high": high,
                "primary_model_decision": PRIMARY_LABEL,
                "decision_basis": "SIMPLER_AND_HIGHER_GROUPED_TRANSFER_SCORE",
            }
        )
        for well_name, values in pivot.iterrows():
            paired_rows.append(
                {
                    "validation_scheme": scheme,
                    "well_name": str(well_name),
                    "rt_gr_den_macro_f1": float(values["FULL_RT_GR_DEN"]),
                    "gr_den_macro_f1": float(values[PRIMARY_SCENARIO]),
                    "gr_den_minus_rt_gr_den": float(
                        values["gr_den_minus_rt_gr_den"]
                    ),
                    "gr_den_better": bool(values["gr_den_minus_rt_gr_den"] > 0),
                }
            )
    keep.to_csv(
        EVIDENCE_DIR / "primary_model_candidate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def calibration_cache_path(scheme: str) -> Path:
    return CACHE_DIR / f"{scheme.lower()}__rf_gr_den_group_calibrated.npz"


def serial_random_forest():
    """Use the frozen estimator with deterministic single-process execution."""
    estimator = frozen_random_forest()
    estimator.set_params(n_jobs=1)
    return estimator


def run_primary_calibration(
    data: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    well_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    y_true = data["facies_id"].to_numpy(int)
    source_row = data["source_row"].to_numpy(int)

    for scheme in SCHEMES:
        raw_oof = np.full((len(data), len(ALL_LABELS)), np.nan, dtype=np.float32)
        calibrated_oof = np.full_like(raw_oof, np.nan)
        for fold, (outer_train_index, test_index) in enumerate(
            validation_splits(data, scheme), start=1
        ):
            outer_train = data.iloc[outer_train_index]
            test = data.iloc[test_index]
            calibration_well = choose_calibration_well(outer_train)
            calibration_mask = outer_train["well_name"].eq(calibration_well).to_numpy()
            calibration_index = outer_train_index[calibration_mask]
            fit_index = outer_train_index[~calibration_mask]
            if set(data.iloc[fit_index]["well_name"]) & set(test["well_name"]):
                raise RuntimeError("Base fit overlaps outer test well")
            if set(data.iloc[calibration_index]["well_name"]) & set(test["well_name"]):
                raise RuntimeError("Calibration well overlaps outer test well")

            started = time.perf_counter()
            estimator = serial_random_forest()
            estimator.fit(data.iloc[fit_index][features], y_true[fit_index])
            calibration_probability = np.zeros(
                (len(calibration_index), len(ALL_LABELS)), dtype=float
            )
            test_probability = np.zeros((len(test_index), len(ALL_LABELS)), dtype=float)
            calibration_probability[:, estimator.classes_.astype(int)] = estimator.predict_proba(
                data.iloc[calibration_index][features]
            )
            test_probability[:, estimator.classes_.astype(int)] = estimator.predict_proba(
                test[features]
            )
            temperature, calibration_nll = fit_temperature(
                y_true[calibration_index], calibration_probability
            )
            calibration_scaled = apply_temperature(
                calibration_probability, temperature
            )
            test_scaled = apply_temperature(test_probability, temperature)
            threshold, target_met, calibration_coverage, calibration_error = (
                choose_confidence_threshold(
                    y_true[calibration_index], calibration_scaled
                )
            )
            test_result = confidence_result(
                y_true[test_index], test_scaled, threshold
            )
            raw_oof[test_index] = test_probability.astype(np.float32)
            calibrated_oof[test_index] = test_scaled.astype(np.float32)
            fold_rows.append(
                {
                    "validation_scheme": scheme,
                    "fold": fold,
                    "test_wells": ";".join(sorted(test["well_name"].unique())),
                    "calibration_well": calibration_well,
                    "fit_wells": ";".join(
                        sorted(data.iloc[fit_index]["well_name"].unique())
                    ),
                    "fit_rows": int(len(fit_index)),
                    "calibration_rows": int(len(calibration_index)),
                    "test_rows": int(len(test_index)),
                    "fit_test_well_overlap": 0,
                    "calibration_test_well_overlap": 0,
                    "temperature": temperature,
                    "calibration_negative_log_loss": calibration_nll,
                    "selected_confidence_threshold": threshold,
                    "calibration_target_met": target_met,
                    "calibration_threshold_coverage": calibration_coverage,
                    "calibration_threshold_error": calibration_error,
                    "test_threshold_coverage": test_result["coverage"],
                    "test_threshold_accepted_error_rate": test_result[
                        "accepted_error_rate"
                    ],
                    "test_threshold_errors_referred_fraction": test_result[
                        "errors_referred_fraction"
                    ],
                    "outer_test_error_target_met": bool(
                        test_result["accepted_error_rate"] <= TARGET_ACCEPTED_ERROR
                    ),
                    "fit_calibrate_predict_seconds": float(
                        time.perf_counter() - started
                    ),
                    "feature_set": PRIMARY_LABEL,
                }
            )
            print(
                f"calibration {scheme} fold={fold} test={fold_rows[-1]['test_wells']} "
                f"T={temperature:.3f}",
                flush=True,
            )

        if not np.isfinite(raw_oof).all() or not np.isfinite(calibrated_oof).all():
            raise RuntimeError(f"Incomplete calibration predictions for {scheme}")
        np.savez_compressed(
            calibration_cache_path(scheme),
            source_row=source_row.astype(np.int32),
            raw_probability=raw_oof,
            calibrated_probability=calibrated_oof,
        )

        current_wells = []
        for well_name, positions in data.groupby("well_name", sort=True).indices.items():
            positions = np.asarray(positions, dtype=int)
            for probability_type, probability in (
                ("UNCALIBRATED_REDUCED_TRAIN", raw_oof),
                ("GROUP_CALIBRATED", calibrated_oof),
            ):
                row = {
                    "validation_scheme": scheme,
                    "well_name": str(well_name),
                    "probability_type": probability_type,
                    "rows": int(len(positions)),
                    **probability_metrics(y_true[positions], probability[positions]),
                }
                well_rows.append(row)
                current_wells.append(row)

        well_frame = pd.DataFrame(current_wells)
        for probability_type, probability in (
            ("UNCALIBRATED_REDUCED_TRAIN", raw_oof),
            ("GROUP_CALIBRATED", calibrated_oof),
        ):
            subset = well_frame.loc[
                well_frame["probability_type"].eq(probability_type)
            ]
            row = {
                "validation_scheme": scheme,
                "probability_type": probability_type,
                "feature_set": PRIMARY_LABEL,
                **probability_metrics(y_true, probability),
            }
            for metric in ("negative_log_loss", "multiclass_brier", "ece_15bin"):
                values = subset[metric].to_numpy(float)
                low, high = bootstrap_mean_ci(values)
                row[f"mean_well_{metric}"] = float(values.mean())
                row[f"mean_well_{metric}_ci_low"] = low
                row[f"mean_well_{metric}_ci_high"] = high
            summary_rows.append(row)
            for item in reliability_rows(y_true, probability):
                reliability.append(
                    {
                        "validation_scheme": scheme,
                        "probability_type": probability_type,
                        **item,
                    }
                )
            confidence = probability.max(axis=1)
            prediction = probability.argmax(axis=1)
            for threshold in CONFIDENCE_GRID:
                accepted = confidence >= threshold
                curves.append(
                    {
                        "validation_scheme": scheme,
                        "probability_type": probability_type,
                        "threshold": float(threshold),
                        "coverage": float(accepted.mean()),
                        "accuracy_among_retained_predictions": float(
                            np.mean(prediction[accepted] == y_true[accepted])
                        )
                        if np.any(accepted)
                        else math.nan,
                        "accepted_rows": int(accepted.sum()),
                    }
                )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(well_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(reliability),
        pd.DataFrame(curves),
    )


def well_classification_rows(
    data: pd.DataFrame,
    prediction: np.ndarray,
    scheme: str,
    method: str,
    stride: int,
    minority: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    y_true = data["facies_id"].to_numpy(int)
    for well_name, positions in data.groupby("well_name", sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        truth = y_true[positions]
        predicted = prediction[positions]
        present = np.unique(truth)
        rows.append(
            {
                "validation_scheme": scheme,
                "method": method,
                "stride": stride,
                "effective_anchor_spacing_m": 0.05 * stride,
                "well_name": str(well_name),
                "rows": int(len(positions)),
                "accuracy": float(np.mean(truth == predicted)),
                "macro_f1_present": float(
                    f1_score(
                        truth,
                        predicted,
                        labels=present,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "minority_mean_recall": float(
                    recall_score(
                        truth,
                        predicted,
                        labels=minority,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    return rows


def run_downsample_comparison(
    data: pd.DataFrame,
    features: list[str],
    rule,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = data["facies_id"].to_numpy(int)
    source_row = data["source_row"].to_numpy(int)
    minority = minority_labels(data)
    degraded = {1: data}
    for stride in DOWNSAMPLE_STRIDES[1:]:
        degraded[stride] = degrade_logs_by_stride(data, stride)

    print("Re-executing the historical rule on QC-contiguous degraded segments", flush=True)
    rule_prediction = {
        stride: exact_rule_predictions(frame, rule)
        for stride, frame in degraded.items()
    }

    per_well_rows: list[dict[str, object]] = []
    for scheme in SCHEMES:
        primary = load_valid_prediction_cache(
            cache_path("1", scheme, "RF_MULTISCALE", PRIMARY_SCENARIO),
            source_row,
        )
        if primary is None:
            raise RuntimeError(f"Missing primary Stage 1 cache for {scheme}")
        predictions = {1: primary[0]}
        for stride in DOWNSAMPLE_STRIDES[1:]:
            predictions[stride] = np.full(len(data), -1, dtype=int)

        for fold, (train_index, test_index) in enumerate(
            validation_splits(data, scheme), start=1
        ):
            estimator = serial_random_forest()
            estimator.fit(data.iloc[train_index][features], y_true[train_index])
            for stride in DOWNSAMPLE_STRIDES[1:]:
                predictions[stride][test_index] = estimator.predict(
                    degraded[stride].iloc[test_index][features]
                ).astype(int)
            print(f"downsample {scheme} fold={fold}", flush=True)

        for stride in DOWNSAMPLE_STRIDES:
            if np.any(predictions[stride] < 0):
                raise RuntimeError(f"Incomplete downsample predictions: {scheme} x{stride}")
            per_well_rows.extend(
                well_classification_rows(
                    data,
                    predictions[stride],
                    scheme,
                    "RF_GR_DEN_MULTISCALE",
                    stride,
                    minority,
                )
            )
            per_well_rows.extend(
                well_classification_rows(
                    data,
                    rule_prediction[stride],
                    scheme,
                    "LEGACY_RULE_QC_SEGMENT_REEXECUTION",
                    stride,
                    minority,
                )
            )

    per_well = pd.DataFrame(per_well_rows)
    summary_rows = []
    for keys, group in per_well.groupby(
        ["validation_scheme", "method", "stride", "effective_anchor_spacing_m"],
        sort=True,
    ):
        scheme, method, stride, spacing = keys
        row = {
            "validation_scheme": scheme,
            "method": method,
            "stride": int(stride),
            "effective_anchor_spacing_m": float(spacing),
            "wells": int(group["well_name"].nunique()),
            "mean_well_accuracy": float(group["accuracy"].mean()),
            "mean_well_macro_f1": float(group["macro_f1_present"].mean()),
            "mean_well_minority_recall": float(group["minority_mean_recall"].mean()),
        }
        for metric in ("accuracy", "macro_f1_present", "minority_mean_recall"):
            low, high = bootstrap_mean_ci(group[metric].to_numpy(float))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary_rows.append(row)
    return per_well, pd.DataFrame(summary_rows)


def transition_records(depth: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    changes = np.flatnonzero(labels[1:] != labels[:-1])
    return pd.DataFrame(
        {
            "depth_m": 0.5 * (depth[changes] + depth[changes + 1]),
            "from_id": labels[changes].astype(int),
            "to_id": labels[changes + 1].astype(int),
        }
    )


def boundary_metrics_for_prediction(
    data: pd.DataFrame,
    prediction: np.ndarray,
    scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = data["facies_id"].to_numpy(int)
    segment_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    for (well_name, segment_id), positions in data.groupby(
        ["well_name", "segment_id"], sort=False
    ).indices.items():
        positions = np.asarray(positions, dtype=int)
        depth = data.iloc[positions]["depth"].to_numpy(float)
        observed = transition_records(depth, y_true[positions])
        predicted = transition_records(depth, prediction[positions])

        for metric_type in ("CLASS_AGNOSTIC", "TRANSITION_AWARE"):
            matches: list[tuple[float, float, float, int, int]] = []
            if metric_type == "CLASS_AGNOSTIC":
                raw = match_boundary_offsets(
                    observed["depth_m"].to_numpy(float),
                    predicted["depth_m"].to_numpy(float),
                    BOUNDARY_TOLERANCE_M,
                )
                for left, right, offset in raw:
                    obs_row = observed.iloc[
                        int(np.argmin(np.abs(observed["depth_m"].to_numpy() - left)))
                    ]
                    matches.append(
                        (left, right, offset, int(obs_row.from_id), int(obs_row.to_id))
                    )
            else:
                transition_types = sorted(
                    set(zip(observed["from_id"], observed["to_id"]))
                    | set(zip(predicted["from_id"], predicted["to_id"]))
                )
                for from_id, to_id in transition_types:
                    observed_subset = observed.loc[
                        observed["from_id"].eq(from_id)
                        & observed["to_id"].eq(to_id),
                        "depth_m",
                    ].to_numpy(float)
                    predicted_subset = predicted.loc[
                        predicted["from_id"].eq(from_id)
                        & predicted["to_id"].eq(to_id),
                        "depth_m",
                    ].to_numpy(float)
                    for left, right, offset in match_boundary_offsets(
                        observed_subset,
                        predicted_subset,
                        BOUNDARY_TOLERANCE_M,
                    ):
                        matches.append(
                            (left, right, offset, int(from_id), int(to_id))
                        )

            segment_rows.append(
                {
                    "validation_scheme": scheme,
                    "metric_type": metric_type,
                    "well_name": str(well_name),
                    "segment_id": int(segment_id),
                    "observed_boundaries": int(len(observed)),
                    "predicted_boundaries": int(len(predicted)),
                    "matched_boundaries": int(len(matches)),
                }
            )
            for left, right, offset, from_id, to_id in matches:
                match_rows.append(
                    {
                        "validation_scheme": scheme,
                        "metric_type": metric_type,
                        "well_name": str(well_name),
                        "segment_id": int(segment_id),
                        "observed_boundary_depth_m": left,
                        "predicted_boundary_depth_m": right,
                        "absolute_offset_m": offset,
                        "observed_from_id": from_id,
                        "observed_from_code": ID_TO_CODE[from_id],
                        "observed_to_id": to_id,
                        "observed_to_code": ID_TO_CODE[to_id],
                    }
                )

    segments = pd.DataFrame(segment_rows)
    matches = pd.DataFrame(match_rows)
    summary_rows = []
    for metric_type, group in segments.groupby("metric_type", sort=True):
        well_rows = []
        for well_name, well in group.groupby("well_name", sort=True):
            observed_count = int(well["observed_boundaries"].sum())
            predicted_count = int(well["predicted_boundaries"].sum())
            matched_count = int(well["matched_boundaries"].sum())
            precision = matched_count / predicted_count if predicted_count else math.nan
            recall = matched_count / observed_count if observed_count else math.nan
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            well_rows.append(
                {
                    "well_name": str(well_name),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        well_frame = pd.DataFrame(well_rows)
        observed_count = int(group["observed_boundaries"].sum())
        predicted_count = int(group["predicted_boundaries"].sum())
        matched_count = int(group["matched_boundaries"].sum())
        precision = matched_count / predicted_count
        recall = matched_count / observed_count
        pooled_f1 = 2.0 * precision * recall / (precision + recall)
        offsets = matches.loc[
            matches["metric_type"].eq(metric_type), "absolute_offset_m"
        ].to_numpy(float)
        low, high = bootstrap_mean_ci(well_frame["f1"].to_numpy(float))
        summary_rows.append(
            {
                "validation_scheme": scheme,
                "metric_type": metric_type,
                "tolerance_m": BOUNDARY_TOLERANCE_M,
                "observed_boundaries": observed_count,
                "predicted_boundaries": predicted_count,
                "matched_boundaries": matched_count,
                "pooled_precision": precision,
                "pooled_recall": recall,
                "pooled_f1": pooled_f1,
                "mean_well_f1": float(well_frame["f1"].mean()),
                "mean_well_f1_ci_low": low,
                "mean_well_f1_ci_high": high,
                "median_offset_m": float(np.median(offsets)),
                "p95_offset_m": float(np.quantile(offsets, 0.95)),
            }
        )
    return segments, matches, pd.DataFrame(summary_rows)


def run_primary_interval_analysis(
    data: pd.DataFrame,
    picks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = data["facies_id"].to_numpy(int)
    source_row = data["source_row"].to_numpy(int)
    weights = sample_support_weights(data)
    observed_runs = run_intervals(data, y_true)
    run_frames = []
    thickness_frames = []
    occupancy_frames = []
    boundary_segments = []
    boundary_matches = []
    summaries = []
    for scheme in SCHEMES:
        cached = load_valid_prediction_cache(
            cache_path("1", scheme, "RF_MULTISCALE", PRIMARY_SCENARIO), source_row
        )
        if cached is None:
            raise RuntimeError(f"Missing primary prediction cache for {scheme}")
        prediction = cached[0]
        predicted_runs = run_intervals(data, prediction)
        runs = observed_run_matches(observed_runs, predicted_runs, scheme)
        thickness = thickness_metrics(data, prediction, weights, scheme)
        occupancy = target_interval_facies_occupancy(data, picks, prediction, scheme)
        segments, matches, boundary_summary = boundary_metrics_for_prediction(
            data, prediction, scheme
        )
        aware = boundary_summary.loc[
            boundary_summary["metric_type"].eq("TRANSITION_AWARE")
        ].iloc[0]
        agnostic = boundary_summary.loc[
            boundary_summary["metric_type"].eq("CLASS_AGNOSTIC")
        ].iloc[0]
        summaries.append(
            {
                "validation_scheme": scheme,
                "feature_set": PRIMARY_LABEL,
                "observed_runs": int(len(runs)),
                "median_best_same_class_iou": float(
                    runs["best_same_class_iou"].median()
                ),
                "run_iou_ge_0p5_fraction": float(
                    runs["matched_iou_ge_0p5"].mean()
                ),
                "thin_runs": int(runs["is_thin_run_le_0p5m"].sum()),
                "thin_run_iou_ge_0p5_fraction": float(
                    runs.loc[
                        runs["is_thin_run_le_0p5m"], "matched_iou_ge_0p5"
                    ].mean()
                ),
                "class_agnostic_boundary_f1": float(agnostic["mean_well_f1"]),
                "transition_aware_boundary_f1": float(aware["mean_well_f1"]),
                "transition_aware_p95_offset_m": float(aware["p95_offset_m"]),
                "mean_abs_class_thickness_bias_m": float(
                    thickness["absolute_bias_m"].mean()
                ),
                "mean_abs_predicted_vs_operational_swamp_occupancy_m": float(
                    occupancy[
                        "predicted_minus_operational_swamp_occupancy_m"
                    ].abs().mean()
                ),
            }
        )
        run_frames.append(runs)
        thickness_frames.append(thickness)
        occupancy_frames.append(occupancy)
        boundary_segments.append(segments)
        boundary_matches.append(matches)

    return (
        pd.concat(run_frames, ignore_index=True),
        pd.concat(thickness_frames, ignore_index=True),
        pd.concat(occupancy_frames, ignore_index=True),
        pd.concat(boundary_segments, ignore_index=True),
        pd.concat(boundary_matches, ignore_index=True),
        pd.DataFrame(summaries),
    )


def write_report(
    model_decision: pd.DataFrame,
    calibration_folds: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    downsample_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
) -> None:
    violations = calibration_folds.loc[
        ~calibration_folds["outer_test_error_target_met"]
    ]
    lines = [
        "# NRR Major Revision Evidence",
        "",
        "## Primary model",
        "",
    ]
    for row in model_decision.itertuples(index=False):
        lines.append(
            f"- {row.validation_scheme}: GR+DEN improved {row.wells_gr_den_better}/"
            f"{row.paired_wells} wells; mean paired macro-F1 delta "
            f"{row.mean_well_macro_f1_delta:.4f}."
        )
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "Pooled metrics and mean-well metrics are reported separately. "
            "Fold-selected confidence thresholds are exploratory triage decisions, "
            "not a guaranteed outer-test error-control procedure.",
            "",
            f"Outer folds above the {TARGET_ACCEPTED_ERROR:.0%} accepted-error target: "
            f"{len(violations)}/{len(calibration_folds)}.",
            "",
        ]
    )
    for row in calibration_summary.loc[
        calibration_summary["probability_type"].eq("GROUP_CALIBRATED")
    ].itertuples(index=False):
        lines.append(
            f"- {row.validation_scheme}: pooled ECE={row.ece_15bin:.4f}; "
            f"mean-well ECE={row.mean_well_ece_15bin:.4f} "
            f"({row.mean_well_ece_15bin_ci_low:.4f}--"
            f"{row.mean_well_ece_15bin_ci_high:.4f}); pooled NLL="
            f"{row.negative_log_loss:.4f}; pooled Brier={row.multiclass_brier:.4f}."
        )
    lines.extend(["", "## Downsampled GR/DEN and direct rule comparison", ""])
    for row in downsample_summary.itertuples(index=False):
        lines.append(
            f"- {row.validation_scheme}, {row.method}, {row.effective_anchor_spacing_m:.2f} m: "
            f"mean-well macro F1={row.mean_well_macro_f1:.4f}."
        )
    lines.extend(["", "## Boundary definitions", ""])
    for row in boundary_summary.itertuples(index=False):
        lines.append(
            f"- {row.validation_scheme}, {row.metric_type}: mean-well F1="
            f"{row.mean_well_f1:.4f}, P95 matched offset={row.p95_offset_m:.3f} m."
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "All targets are operational well-log interpretations. These results "
            "measure reproducibility under grouped transfer and measurement stress; "
            "they do not independently validate depositional facies, net coal, "
            "resources, reserves, permeability, or production.",
            "",
        ]
    )
    (EVIDENCE_DIR / "MAJOR_REVISION_EVIDENCE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    association, data, _, picks, multiscale, _ = build_analysis_data()
    if len(association) != 101_639 or len(data) != 101_615:
        raise RuntimeError("Unexpected source or analysis row count")
    features = multiscale[PRIMARY_SCENARIO]
    rule = load_legacy_rule_module()

    model_decision, paired_model = summarize_model_selection(data)
    model_decision.to_csv(
        EVIDENCE_DIR / "primary_model_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired_model.to_csv(
        EVIDENCE_DIR / "primary_model_paired_wells.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibration_folds, calibration_wells, calibration_summary, reliability, curves = (
        run_primary_calibration(data, features)
    )
    calibration_folds.to_csv(
        EVIDENCE_DIR / "calibration_fold_audit_gr_den.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration_wells.to_csv(
        EVIDENCE_DIR / "calibration_per_well_gr_den.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration_summary.to_csv(
        EVIDENCE_DIR / "calibration_summary_gr_den.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability.to_csv(
        EVIDENCE_DIR / "calibration_reliability_gr_den.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curves.to_csv(
        EVIDENCE_DIR / "confidence_coverage_gr_den.csv",
        index=False,
        encoding="utf-8-sig",
    )

    downsample_wells, downsample_summary = run_downsample_comparison(
        data, features, rule
    )
    downsample_wells.to_csv(
        EVIDENCE_DIR / "downsample_rule_model_per_well.csv",
        index=False,
        encoding="utf-8-sig",
    )
    downsample_summary.to_csv(
        EVIDENCE_DIR / "downsample_rule_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        run_matches,
        thickness,
        occupancy,
        boundary_segments,
        boundary_matches,
        interval_summary,
    ) = run_primary_interval_analysis(data, picks)
    run_matches.to_csv(
        EVIDENCE_DIR / "primary_observed_run_iou_metrics.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    thickness.to_csv(
        EVIDENCE_DIR / "primary_class_thickness_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    occupancy.to_csv(
        EVIDENCE_DIR / "primary_target_interval_facies_occupancy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boundary_segments.to_csv(
        EVIDENCE_DIR / "boundary_count_audit_class_aware.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boundary_matches.to_csv(
        EVIDENCE_DIR / "matched_boundary_offsets_class_aware.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boundary_summary_frames = []
    for scheme in SCHEMES:
        scheme_segments = boundary_segments.loc[
            boundary_segments["validation_scheme"].eq(scheme)
        ]
        scheme_matches = boundary_matches.loc[
            boundary_matches["validation_scheme"].eq(scheme)
        ]
        for metric_type, group in scheme_segments.groupby("metric_type", sort=True):
            well_rows = []
            for well_name, well in group.groupby("well_name", sort=True):
                observed = int(well["observed_boundaries"].sum())
                predicted = int(well["predicted_boundaries"].sum())
                matched = int(well["matched_boundaries"].sum())
                precision = matched / predicted if predicted else math.nan
                recall = matched / observed if observed else math.nan
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                well_rows.append({"well_name": well_name, "f1": f1})
            wells = pd.DataFrame(well_rows)
            observed = int(group["observed_boundaries"].sum())
            predicted = int(group["predicted_boundaries"].sum())
            matched = int(group["matched_boundaries"].sum())
            precision = matched / predicted
            recall = matched / observed
            pooled_f1 = 2 * precision * recall / (precision + recall)
            offsets = scheme_matches.loc[
                scheme_matches["metric_type"].eq(metric_type), "absolute_offset_m"
            ].to_numpy(float)
            low, high = bootstrap_mean_ci(wells["f1"].to_numpy(float))
            boundary_summary_frames.append(
                {
                    "validation_scheme": scheme,
                    "metric_type": metric_type,
                    "tolerance_m": BOUNDARY_TOLERANCE_M,
                    "observed_boundaries": observed,
                    "predicted_boundaries": predicted,
                    "matched_boundaries": matched,
                    "pooled_precision": precision,
                    "pooled_recall": recall,
                    "pooled_f1": pooled_f1,
                    "mean_well_f1": float(wells["f1"].mean()),
                    "mean_well_f1_ci_low": low,
                    "mean_well_f1_ci_high": high,
                    "median_offset_m": float(np.median(offsets)),
                    "p95_offset_m": float(np.quantile(offsets, 0.95)),
                }
            )
    boundary_summary = pd.DataFrame(boundary_summary_frames)
    boundary_summary.to_csv(
        EVIDENCE_DIR / "boundary_metric_summary_class_aware.csv",
        index=False,
        encoding="utf-8-sig",
    )
    interval_summary.to_csv(
        EVIDENCE_DIR / "primary_interval_metric_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_report(
        model_decision,
        calibration_folds,
        calibration_summary,
        downsample_summary,
        boundary_summary,
    )

    checks = {
        "source_rows_101639": len(association) == 101_639,
        "analysis_rows_101615": len(data) == 101_615,
        "primary_is_gr_den": PRIMARY_SCENARIO == "NO_RT_GR_DEN",
        "primary_better_all_loowo_wells": bool(
            model_decision.loc[
                model_decision["validation_scheme"].eq("LEAVE_ONE_WELL_OUT"),
                "wells_gr_den_better",
            ].iloc[0]
            == 10
        ),
        "calibration_has_13_outer_folds": len(calibration_folds) == 13,
        "calibration_no_test_overlap": bool(
            calibration_folds["fit_test_well_overlap"].eq(0).all()
            and calibration_folds["calibration_test_well_overlap"].eq(0).all()
        ),
        "calibration_outer_target_not_claimed_as_guaranteed": bool(
            (~calibration_folds["outer_test_error_target_met"]).any()
        ),
        "rule_qc_segment_baseline_difference_reported": bool(
            downsample_summary.loc[
                downsample_summary["method"].eq(
                    "LEGACY_RULE_QC_SEGMENT_REEXECUTION"
                )
                & downsample_summary["stride"].eq(1),
                "mean_well_macro_f1",
            ].lt(1.0).all()
        ),
        "transition_aware_metric_present": bool(
            boundary_summary["metric_type"].eq("TRANSITION_AWARE").any()
        ),
    }
    manifest = {
        "stage": "NRR_MAJOR_REVISION_EVIDENCE_V1",
        "created_local_date": "2026-07-31",
        "primary_feature_set": PRIMARY_LABEL,
        "confidence_target": TARGET_ACCEPTED_ERROR,
        "minimum_calibration_coverage": MIN_CALIBRATION_COVERAGE,
        "boundary_tolerance_m": BOUNDARY_TOLERANCE_M,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "source_hashes": source_hashes(),
        "checks": checks,
    }
    (EVIDENCE_DIR / "analysis_validation.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Evidence validation failed: {failed}")
    build_manifest(EVIDENCE_DIR, "evidence_manifest.csv")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
