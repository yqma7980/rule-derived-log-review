from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import ALL_LABELS, ID_TO_CODE, align_probabilities, bootstrap_mean_ci
from nrr_stage_common import (
    OUTPUT_DIR,
    SCHEMES,
    apply_temperature,
    build_analysis_data,
    build_manifest,
    cache_path,
    choose_calibration_well,
    fit_temperature,
    frozen_random_forest,
    load_valid_prediction_cache,
    match_boundary_offsets,
    probability_metrics,
    reliability_rows,
    run_intervals,
    sample_support_weights,
    segment_cell_edges,
    source_hashes,
    validation_splits,
    write_json,
)


STAGE_DIR = OUTPUT_DIR / "stage2"
CALIBRATION_CACHE = OUTPUT_DIR / "stage2_cache"
CONFIDENCE_GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)
TARGET_ACCEPTED_ERROR = 0.02
MIN_CALIBRATION_COVERAGE = 0.50
BOUNDARY_TOLERANCE_M = 0.5
THIN_RUN_THRESHOLD_M = 0.5


def choose_confidence_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, bool, float, float]:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    candidates = []
    fallback = []
    for threshold in CONFIDENCE_GRID:
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        if coverage < MIN_CALIBRATION_COVERAGE or not np.any(accepted):
            continue
        error = float(np.mean(prediction[accepted] != y_true[accepted]))
        fallback.append((error, -coverage, float(threshold), coverage))
        if error <= TARGET_ACCEPTED_ERROR:
            candidates.append((float(threshold), coverage, error))
    if candidates:
        threshold, coverage, error = sorted(candidates, key=lambda item: item[0])[0]
        return threshold, True, coverage, error
    if not fallback:
        return 1.0, False, 0.0, math.nan
    error, negative_coverage, threshold, coverage = sorted(fallback)[0]
    return threshold, False, coverage, error


def confidence_result(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    accepted = confidence >= threshold
    errors = prediction != y_true
    accepted_error = float(errors[accepted].mean()) if np.any(accepted) else math.nan
    errors_referred = float((errors & ~accepted).sum() / errors.sum()) if np.any(errors) else 0.0
    return {
        "coverage": float(accepted.mean()),
        "accepted_error_rate": accepted_error,
        "errors_referred_fraction": errors_referred,
    }


def calibration_cache_path(scheme: str) -> Path:
    CALIBRATION_CACHE.mkdir(parents=True, exist_ok=True)
    return CALIBRATION_CACHE / f"{scheme.lower()}__rf_group_calibrated.npz"


def load_calibration_cache(
    path: Path, source_row: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as cached:
            rows = cached["source_row"].astype(int)
            raw = cached["raw_probability"].astype(float)
            calibrated = cached["calibrated_probability"].astype(float)
    except (KeyError, OSError, ValueError):
        return None
    valid = (
        np.array_equal(rows, source_row)
        and raw.shape == (len(source_row), len(ALL_LABELS))
        and calibrated.shape == raw.shape
        and np.isfinite(raw).all()
        and np.isfinite(calibrated).all()
        and np.allclose(raw.sum(axis=1), 1.0, atol=1e-5)
        and np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-5)
    )
    return (raw, calibrated) if valid else None


def run_group_calibration(
    data: pd.DataFrame,
    full_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    well_rows = []
    summary_rows = []
    reliability = []
    curves = []
    y_true = data["facies_id"].to_numpy(int)
    source_row = data["source_row"].to_numpy(int)
    for scheme in SCHEMES:
        path = calibration_cache_path(scheme)
        cached = load_calibration_cache(path, source_row)
        raw_oof = np.full((len(data), len(ALL_LABELS)), np.nan, dtype=np.float32)
        calibrated_oof = np.full_like(raw_oof, np.nan)
        if cached is not None:
            raw_oof[:] = cached[0]
            calibrated_oof[:] = cached[1]
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
                raise RuntimeError("Calibration base fit overlaps outer test well")
            if set(data.iloc[calibration_index]["well_name"]) & set(test["well_name"]):
                raise RuntimeError("Calibration well overlaps outer test well")

            start = time.perf_counter()
            estimator = frozen_random_forest()
            estimator.fit(data.iloc[fit_index][full_features], y_true[fit_index])
            calibration_probability = align_probabilities(
                np.asarray(estimator.classes_),
                estimator.predict_proba(data.iloc[calibration_index][full_features]),
            )
            test_raw_probability = align_probabilities(
                np.asarray(estimator.classes_),
                estimator.predict_proba(test[full_features]),
            )
            temperature, calibration_nll = fit_temperature(
                y_true[calibration_index], calibration_probability
            )
            calibration_scaled = apply_temperature(calibration_probability, temperature)
            test_calibrated_probability = apply_temperature(test_raw_probability, temperature)
            threshold, target_met, calibration_coverage, calibration_error = (
                choose_confidence_threshold(
                    y_true[calibration_index], calibration_scaled
                )
            )
            elapsed = time.perf_counter() - start
            raw_oof[test_index] = test_raw_probability.astype(np.float32)
            calibrated_oof[test_index] = test_calibrated_probability.astype(np.float32)
            test_confidence = confidence_result(
                y_true[test_index], test_calibrated_probability, threshold
            )
            fold_rows.append(
                {
                    "validation_scheme": scheme,
                    "fold": fold,
                    "test_wells": ";".join(sorted(test["well_name"].unique())),
                    "calibration_well": calibration_well,
                    "fit_wells": ";".join(sorted(data.iloc[fit_index]["well_name"].unique())),
                    "fit_rows": int(len(fit_index)),
                    "calibration_rows": int(len(calibration_index)),
                    "test_rows": int(len(test_index)),
                    "fit_test_well_overlap": int(
                        len(set(data.iloc[fit_index]["well_name"]) & set(test["well_name"]))
                    ),
                    "calibration_test_well_overlap": int(
                        len(set(data.iloc[calibration_index]["well_name"]) & set(test["well_name"]))
                    ),
                    "temperature": temperature,
                    "calibration_negative_log_loss": calibration_nll,
                    "selected_confidence_threshold": threshold,
                    "calibration_target_met": target_met,
                    "calibration_threshold_coverage": calibration_coverage,
                    "calibration_threshold_error": calibration_error,
                    "test_threshold_coverage": test_confidence["coverage"],
                    "test_threshold_accepted_error_rate": test_confidence[
                        "accepted_error_rate"
                    ],
                    "test_threshold_errors_referred_fraction": test_confidence[
                        "errors_referred_fraction"
                    ],
                    "fit_calibrate_predict_seconds": float(elapsed),
                    "cache_was_present_before_run": cached is not None,
                }
            )
            print(
                f"stage2 calibration {scheme} fold={fold} cal={calibration_well} "
                f"T={temperature:.3f} seconds={elapsed:.1f}",
                flush=True,
            )
        if not np.isfinite(raw_oof).all() or not np.isfinite(calibrated_oof).all():
            raise RuntimeError(f"Incomplete calibrated OOF probabilities for {scheme}")
        np.savez_compressed(
            path,
            source_row=source_row.astype(np.int32),
            raw_probability=raw_oof,
            calibrated_probability=calibrated_oof,
        )
        for well_name, positions in data.groupby("well_name", sort=True).indices.items():
            positions = np.asarray(positions, dtype=int)
            for probability_type, probability in (
                ("UNCALIBRATED_REDUCED_TRAIN", raw_oof),
                ("GROUP_CALIBRATED", calibrated_oof),
            ):
                well_rows.append(
                    {
                        "validation_scheme": scheme,
                        "well_name": str(well_name),
                        "probability_type": probability_type,
                        "rows": int(len(positions)),
                        **probability_metrics(y_true[positions], probability[positions]),
                    }
                )
        well_frame = pd.DataFrame(
            row for row in well_rows if row["validation_scheme"] == scheme
        )
        for probability_type, probability in (
            ("UNCALIBRATED_REDUCED_TRAIN", raw_oof),
            ("GROUP_CALIBRATED", calibrated_oof),
        ):
            subset = well_frame.loc[well_frame["probability_type"].eq(probability_type)]
            row = {
                "validation_scheme": scheme,
                "probability_type": probability_type,
                **probability_metrics(y_true, probability),
            }
            for metric in ("negative_log_loss", "multiclass_brier", "ece_15bin"):
                low, high = bootstrap_mean_ci(subset[metric].to_numpy(float))
                row[f"mean_well_{metric}"] = float(subset[metric].mean())
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
                        "accepted_accuracy": float(
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


def distance_to_operational_boundary(data: pd.DataFrame) -> np.ndarray:
    distance = np.full(len(data), np.inf, dtype=float)
    labels = data["facies_id"].to_numpy(int)
    for positions in data.groupby(["well_name", "segment_id"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        depth = data.iloc[positions]["depth"].to_numpy(float)
        changes = labels[positions][1:] != labels[positions][:-1]
        boundaries = 0.5 * (depth[:-1][changes] + depth[1:][changes])
        if len(boundaries):
            distance[positions] = np.min(
                np.abs(depth[:, None] - boundaries[None, :]), axis=1
            )
    return distance


def calibration_boundary_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    y_true = data["facies_id"].to_numpy(int)
    boundary_distance = distance_to_operational_boundary(data)
    bins = [-np.inf, 0.10, 0.25, 0.50, np.inf]
    labels = ["LE_0P10", "GT_0P10_LE_0P25", "GT_0P25_LE_0P50", "GT_0P50"]
    rows = []
    for scheme in SCHEMES:
        cached = load_calibration_cache(
            calibration_cache_path(scheme), data["source_row"].to_numpy(int)
        )
        if cached is None:
            raise RuntimeError(f"Missing calibration cache for {scheme}")
        for probability_type, probability in (
            ("UNCALIBRATED_REDUCED_TRAIN", cached[0]),
            ("GROUP_CALIBRATED", cached[1]),
        ):
            confidence = probability.max(axis=1)
            prediction = probability.argmax(axis=1)
            categories = pd.cut(boundary_distance, bins=bins, labels=labels)
            for category in labels:
                mask = np.asarray(categories == category)
                rows.append(
                    {
                        "validation_scheme": scheme,
                        "probability_type": probability_type,
                        "distance_bin": category,
                        "rows": int(mask.sum()),
                        "mean_confidence": float(confidence[mask].mean()),
                        "accuracy": float(np.mean(prediction[mask] == y_true[mask])),
                    }
                )
    return pd.DataFrame(rows)


def observed_run_matches(
    observed_runs: pd.DataFrame,
    predicted_runs: pd.DataFrame,
    scheme: str,
) -> pd.DataFrame:
    predicted_groups = {
        key: group
        for key, group in predicted_runs.groupby(
            ["well_name", "segment_id", "facies_id"], sort=False
        )
    }
    rows = []
    for run in observed_runs.itertuples(index=False):
        candidates = predicted_groups.get((run.well_name, run.segment_id, run.facies_id))
        best_iou = 0.0
        best_overlap = 0.0
        if candidates is not None:
            for predicted in candidates.itertuples(index=False):
                overlap = max(
                    0.0,
                    min(run.base_depth_m, predicted.base_depth_m)
                    - max(run.top_depth_m, predicted.top_depth_m),
                )
                union = (
                    max(run.base_depth_m, predicted.base_depth_m)
                    - min(run.top_depth_m, predicted.top_depth_m)
                )
                iou = overlap / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_overlap = overlap
        rows.append(
            {
                "validation_scheme": scheme,
                "well_name": run.well_name,
                "segment_id": int(run.segment_id),
                "observed_run_number": int(run.run_number),
                "facies_id": int(run.facies_id),
                "facies_code": ID_TO_CODE[int(run.facies_id)],
                "coal_seam_unit": run.coal_seam_unit,
                "top_depth_m": float(run.top_depth_m),
                "base_depth_m": float(run.base_depth_m),
                "observed_thickness_m": float(run.thickness_m),
                "best_same_class_overlap_m": float(best_overlap),
                "best_same_class_iou": float(best_iou),
                "detected_any_overlap": bool(best_overlap > 0),
                "matched_iou_ge_0p5": bool(best_iou >= 0.5),
                "is_thin_run_le_0p5m": bool(run.thickness_m <= THIN_RUN_THRESHOLD_M + 1e-9),
            }
        )
    return pd.DataFrame(rows)


def boundary_offset_tables(
    data: pd.DataFrame,
    prediction: np.ndarray,
    scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = data["facies_id"].to_numpy(int)
    match_rows = []
    segment_rows = []
    for (well_name, segment_id), positions in data.groupby(
        ["well_name", "segment_id"], sort=False
    ).indices.items():
        positions = np.asarray(positions, dtype=int)
        depth = data.iloc[positions]["depth"].to_numpy(float)
        true_changes = y_true[positions][1:] != y_true[positions][:-1]
        pred_changes = prediction[positions][1:] != prediction[positions][:-1]
        observed = 0.5 * (depth[:-1][true_changes] + depth[1:][true_changes])
        predicted = 0.5 * (depth[:-1][pred_changes] + depth[1:][pred_changes])
        matches = match_boundary_offsets(observed, predicted, BOUNDARY_TOLERANCE_M)
        segment_rows.append(
            {
                "validation_scheme": scheme,
                "well_name": str(well_name),
                "segment_id": int(segment_id),
                "observed_boundaries": int(len(observed)),
                "predicted_boundaries": int(len(predicted)),
                "matched_boundaries": int(len(matches)),
                "unmatched_observed": int(len(observed) - len(matches)),
                "unmatched_predicted": int(len(predicted) - len(matches)),
            }
        )
        for observed_depth, predicted_depth, offset in matches:
            nearest = positions[int(np.argmin(np.abs(depth - observed_depth)))]
            match_rows.append(
                {
                    "validation_scheme": scheme,
                    "well_name": str(well_name),
                    "segment_id": int(segment_id),
                    "coal_seam_unit": str(data.iloc[nearest]["coal_seam_unit"]),
                    "observed_boundary_depth_m": observed_depth,
                    "predicted_boundary_depth_m": predicted_depth,
                    "absolute_offset_m": offset,
                }
            )
    return pd.DataFrame(match_rows), pd.DataFrame(segment_rows)


def thickness_metrics(
    data: pd.DataFrame,
    prediction: np.ndarray,
    weights: np.ndarray,
    scheme: str,
) -> pd.DataFrame:
    frame = data[["well_name", "coal_seam_unit", "facies_id"]].copy()
    frame["predicted_id"] = prediction
    frame["support_m"] = weights
    rows = []
    groups = frame.groupby(["well_name", "coal_seam_unit"], sort=True)
    for (well_name, seam), group in groups:
        for facies_id in ALL_LABELS:
            observed = float(group.loc[group["facies_id"].eq(facies_id), "support_m"].sum())
            predicted = float(group.loc[group["predicted_id"].eq(facies_id), "support_m"].sum())
            rows.append(
                {
                    "validation_scheme": scheme,
                    "well_name": str(well_name),
                    "coal_seam_unit": str(seam),
                    "facies_id": int(facies_id),
                    "facies_code": ID_TO_CODE[int(facies_id)],
                    "observed_thickness_m": observed,
                    "predicted_thickness_m": predicted,
                    "bias_m": predicted - observed,
                    "absolute_bias_m": abs(predicted - observed),
                    "relative_bias": (predicted - observed) / observed if observed > 0 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def sample_cell_bounds(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    top = np.zeros(len(data), dtype=float)
    base = np.zeros(len(data), dtype=float)
    for positions in data.groupby(["well_name", "segment_id"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        edges = segment_cell_edges(data.iloc[positions]["depth"].to_numpy(float))
        top[positions] = edges[:-1]
        base[positions] = edges[1:]
    return top, base


def target_interval_facies_occupancy(
    data: pd.DataFrame,
    picks: pd.DataFrame,
    prediction: np.ndarray,
    scheme: str,
) -> pd.DataFrame:
    y_true = data["facies_id"].to_numpy(int)
    cell_top, cell_base = sample_cell_bounds(data)
    rows = []
    for pick in picks.itertuples(index=False):
        top_depth = float(pick.selected_top_depth_m)
        base_depth = float(pick.base_depth_m)
        well_mask = data["well_name"].eq(pick.well_id).to_numpy()
        overlap = np.maximum(
            0.0,
            np.minimum(cell_base, base_depth) - np.maximum(cell_top, top_depth),
        )
        overlap[~well_mask] = 0.0
        observed_sw = float(overlap[y_true == 0].sum())
        predicted_sw = float(overlap[prediction == 0].sum())
        covered = float(overlap.sum())
        gross = base_depth - top_depth
        rows.append(
            {
                "validation_scheme": scheme,
                "well_name": str(pick.well_id),
                "coal_seam_unit": str(pick.seam),
                "selected_top_depth_m": top_depth,
                "base_depth_m": base_depth,
                "gross_interval_m": gross,
                "logged_coverage_m": covered,
                "logged_coverage_fraction": covered / gross if gross > 0 else math.nan,
                "operational_swamp_facies_occupancy_m": observed_sw,
                "predicted_swamp_facies_occupancy_m": predicted_sw,
                "predicted_minus_operational_swamp_occupancy_m": predicted_sw - observed_sw,
                "predicted_vs_operational_swamp_occupancy_relative_bias": (
                    (predicted_sw - observed_sw) / observed_sw if observed_sw > 0 else math.nan
                ),
                "quantity_definition": "DEPOSITIONAL_SUBFACIES_CLASS_1_OCCUPANCY_WITHIN_AUTHORIZED_GROSS_INTERVAL",
                "comparison_scope": "PREDICTION_VS_OPERATIONAL_LABEL_ONLY",
                "pick_evidence_role": str(pick.v2_input_role),
            }
        )
    return pd.DataFrame(rows)


def island_metrics(runs: pd.DataFrame, scheme: str, label_source: str) -> pd.DataFrame:
    rows = []
    for (well_name, segment_id), group in runs.groupby(
        ["well_name", "segment_id"], sort=False
    ):
        ordered = group.sort_values("run_number")
        labels = ordered["facies_id"].to_numpy(int)
        thickness = ordered["thickness_m"].to_numpy(float)
        islands = 0
        island_thickness = 0.0
        for index in range(1, len(ordered) - 1):
            if labels[index - 1] == labels[index + 1] and labels[index] != labels[index - 1] and thickness[index] <= THIN_RUN_THRESHOLD_M + 1e-9:
                islands += 1
                island_thickness += float(thickness[index])
        rows.append(
            {
                "validation_scheme": scheme,
                "label_source": label_source,
                "well_name": str(well_name),
                "segment_id": int(segment_id),
                "logged_thickness_m": float(ordered["thickness_m"].sum()),
                "short_aba_islands": int(islands),
                "short_aba_island_thickness_m": island_thickness,
            }
        )
    return pd.DataFrame(rows)


def run_interval_analysis(
    data: pd.DataFrame,
    picks: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    weights = sample_support_weights(data)
    y_true = data["facies_id"].to_numpy(int)
    observed_runs = run_intervals(data, y_true)
    run_matches = []
    boundary_matches = []
    boundary_segments = []
    thickness = []
    facies_occupancy = []
    islands = []
    summaries = []
    for scheme in SCHEMES:
        cached = load_valid_prediction_cache(
            cache_path("1", scheme, "RF_MULTISCALE", "FULL_RT_GR_DEN"),
            data["source_row"].to_numpy(int),
        )
        if cached is None:
            raise RuntimeError(f"Missing stage1 primary cache for {scheme}")
        prediction = cached[0]
        predicted_runs = run_intervals(data, prediction)
        matches = observed_run_matches(observed_runs, predicted_runs, scheme)
        boundary_match, boundary_segment = boundary_offset_tables(data, prediction, scheme)
        thickness_frame = thickness_metrics(data, prediction, weights, scheme)
        occupancy_frame = target_interval_facies_occupancy(data, picks, prediction, scheme)
        observed_islands = island_metrics(observed_runs, scheme, "OPERATIONAL_LABEL")
        predicted_islands = island_metrics(predicted_runs, scheme, "PREDICTION")
        run_matches.append(matches)
        boundary_matches.append(boundary_match)
        boundary_segments.append(boundary_segment)
        thickness.append(thickness_frame)
        facies_occupancy.append(occupancy_frame)
        islands.extend([observed_islands, predicted_islands])

        well_summary = []
        for well_name, group in matches.groupby("well_name", sort=True):
            offsets = boundary_match.loc[
                boundary_match["well_name"].eq(well_name), "absolute_offset_m"
            ].to_numpy(float)
            well_summary.append(
                {
                    "well_name": str(well_name),
                    "median_run_iou": float(group["best_same_class_iou"].median()),
                    "thin_run_iou_ge_0p5_fraction": float(
                        group.loc[group["is_thin_run_le_0p5m"], "matched_iou_ge_0p5"].mean()
                    ),
                    "median_boundary_offset_m": float(np.median(offsets)) if len(offsets) else math.nan,
                    "p95_boundary_offset_m": float(np.quantile(offsets, 0.95)) if len(offsets) else math.nan,
                }
            )
        well_summary = pd.DataFrame(well_summary)
        row = {
            "validation_scheme": scheme,
            "observed_runs": int(len(matches)),
            "median_best_same_class_iou": float(matches["best_same_class_iou"].median()),
            "run_iou_ge_0p5_fraction": float(matches["matched_iou_ge_0p5"].mean()),
            "thin_runs": int(matches["is_thin_run_le_0p5m"].sum()),
            "thin_run_iou_ge_0p5_fraction": float(
                matches.loc[matches["is_thin_run_le_0p5m"], "matched_iou_ge_0p5"].mean()
            ),
            "matched_boundaries": int(len(boundary_match)),
            "median_boundary_offset_m": float(boundary_match["absolute_offset_m"].median()),
            "p90_boundary_offset_m": float(boundary_match["absolute_offset_m"].quantile(0.90)),
            "p95_boundary_offset_m": float(boundary_match["absolute_offset_m"].quantile(0.95)),
            "mean_abs_class_thickness_bias_m": float(thickness_frame["absolute_bias_m"].mean()),
            "mean_abs_predicted_vs_operational_swamp_occupancy_m": float(
                occupancy_frame[
                    "predicted_minus_operational_swamp_occupancy_m"
                ].abs().mean()
            ),
        }
        for metric in (
            "median_run_iou",
            "thin_run_iou_ge_0p5_fraction",
            "median_boundary_offset_m",
            "p95_boundary_offset_m",
        ):
            low, high = bootstrap_mean_ci(well_summary[metric].dropna().to_numpy(float))
            row[f"mean_well_{metric}"] = float(well_summary[metric].mean())
            row[f"mean_well_{metric}_ci_low"] = low
            row[f"mean_well_{metric}_ci_high"] = high
        summaries.append(row)
    return (
        pd.concat(run_matches, ignore_index=True),
        pd.concat(boundary_matches, ignore_index=True),
        pd.concat(boundary_segments, ignore_index=True),
        pd.concat(thickness, ignore_index=True),
        pd.concat(facies_occupancy, ignore_index=True),
        pd.concat(islands, ignore_index=True),
        pd.DataFrame(summaries),
    )


def build_report(calibration: pd.DataFrame, intervals: pd.DataFrame) -> str:
    lines = [
        "# NRR Stage 2: Interval metrics and group-held-out calibration",
        "",
        "## Calibration",
        "",
        "| Validation | Probability | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in calibration.itertuples(index=False):
        lines.append(
            f"| {row.validation_scheme} | {row.probability_type} | "
            f"{row.negative_log_loss:.4f} | {row.multiclass_brier:.4f} | {row.ece_15bin:.4f} |"
        )
    lines.extend(
        [
            "",
            "The calibrator for each outer fold used one complete training well that was excluded from base-model fitting. Outer test labels were not used for temperature or threshold selection.",
            "",
            "## Interval metrics",
            "",
            "| Validation | Median run IoU | Thin-run IoU>=0.5 | Median boundary offset (m) | P95 boundary offset (m) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in intervals.itertuples(index=False):
        lines.append(
            f"| {row.validation_scheme} | {row.median_best_same_class_iou:.4f} | "
            f"{row.thin_run_iou_ge_0p5_fraction:.4f} | {row.median_boundary_offset_m:.4f} | "
            f"{row.p95_boundary_offset_m:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "All interval metrics compare predictions with deterministic operational labels. Swamp-facies occupancy is a depositional-subfacies quantity and is not compared with net-coal thickness.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    association, data, wells, picks, multiscale, _ = build_analysis_data()
    calibration_folds, calibration_wells, calibration_summary, reliability, curves = (
        run_group_calibration(data, multiscale["FULL_RT_GR_DEN"])
    )
    boundary_diagnostics = calibration_boundary_diagnostics(data)
    (
        run_matches,
        boundary_matches,
        boundary_segments,
        thickness,
        facies_occupancy,
        islands,
        interval_summary,
    ) = run_interval_analysis(data, picks)

    calibration_folds.to_csv(STAGE_DIR / "calibration_fold_audit.csv", index=False, encoding="utf-8-sig")
    calibration_wells.to_csv(STAGE_DIR / "calibration_per_well_metrics.csv", index=False, encoding="utf-8-sig")
    calibration_summary.to_csv(STAGE_DIR / "calibration_summary.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(STAGE_DIR / "calibration_reliability_bins.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(STAGE_DIR / "confidence_coverage_curves.csv", index=False, encoding="utf-8-sig")
    boundary_diagnostics.to_csv(STAGE_DIR / "confidence_by_boundary_distance.csv", index=False, encoding="utf-8-sig")
    run_matches.to_csv(STAGE_DIR / "observed_run_iou_metrics.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    boundary_matches.to_csv(STAGE_DIR / "matched_boundary_offsets.csv", index=False, encoding="utf-8-sig")
    boundary_segments.to_csv(STAGE_DIR / "boundary_count_audit.csv", index=False, encoding="utf-8-sig")
    thickness.to_csv(STAGE_DIR / "class_thickness_metrics.csv", index=False, encoding="utf-8-sig")
    facies_occupancy.to_csv(
        STAGE_DIR / "target_interval_facies_occupancy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (STAGE_DIR / "target_interval_resource_consistency.csv").unlink(missing_ok=True)
    islands.to_csv(STAGE_DIR / "short_aba_island_metrics.csv", index=False, encoding="utf-8-sig")
    interval_summary.to_csv(STAGE_DIR / "interval_metric_summary.csv", index=False, encoding="utf-8-sig")

    checks = {
        "source_rows_101639": int(len(association)) == 101_639,
        "analysis_rows_101615": int(len(data)) == 101_615,
        "calibration_folds_13": int(len(calibration_folds)) == 13,
        "calibration_split_has_no_test_overlap": bool(
            calibration_folds["fit_test_well_overlap"].eq(0).all()
            and calibration_folds["calibration_test_well_overlap"].eq(0).all()
        ),
        "four_calibration_summaries": int(len(calibration_summary)) == 4,
        "two_interval_summaries": int(len(interval_summary)) == 2,
        "eighty_target_interval_facies_rows": int(len(facies_occupancy)) == 80,
        "all_interval_thickness_nonnegative": bool(
            run_matches["observed_thickness_m"].ge(0).all()
            and thickness["observed_thickness_m"].ge(0).all()
            and thickness["predicted_thickness_m"].ge(0).all()
        ),
        "all_boundary_offsets_within_tolerance": bool(
            boundary_matches["absolute_offset_m"].le(BOUNDARY_TOLERANCE_M + 1e-12).all()
        ),
        "no_cross_quantity_net_coal_columns": not any(
            "net_coal" in column.lower() for column in facies_occupancy.columns
        ),
    }
    state = {
        "package": "nrr_stage2",
        "generated_on": str(pd.Timestamp.now(tz="Asia/Shanghai")),
        "overall_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_hashes": source_hashes(),
        "counts": {
            "calibration_folds": int(len(calibration_folds)),
            "observed_run_metrics": int(len(run_matches)),
            "matched_boundaries": int(len(boundary_matches)),
            "target_interval_facies_rows": int(len(facies_occupancy)),
            "authorized_target_intervals": int(len(picks)),
        },
        "claim_boundary": [
            "Calibration uses a held-out training well and never outer-test labels.",
            "Interval metrics compare against operational rule labels, not independent facies truth.",
            "Stage 2 does not compare swamp-facies occupancy with net-coal thickness or estimate resources.",
        ],
    }
    write_json(STAGE_DIR / "stage2_quality_gate.json", state)
    (STAGE_DIR / "STAGE2_REPORT.md").write_text(
        build_report(calibration_summary, interval_summary), encoding="utf-8"
    )
    build_manifest(STAGE_DIR)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
