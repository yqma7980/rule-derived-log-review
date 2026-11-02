from __future__ import annotations

import json
import math
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from common import (
    ALL_LABELS,
    ID_TO_CODE,
    bootstrap_mean_ci,
    per_class_metrics,
    pooled_metrics,
)
from nrr_stage_common import (
    LABEL_AUDIT,
    OUTPUT_DIR,
    RANDOM_SEED,
    SCENARIOS,
    SCHEMES,
    build_analysis_data,
    build_manifest,
    cache_path,
    contiguous_gap_mask,
    degrade_logs_by_stride,
    frozen_random_forest,
    load_valid_prediction_cache,
    probability_metrics,
    raw_logistic_regression,
    save_prediction_cache,
    source_hashes,
    split_audit_rows,
    stable_seed,
    validation_splits,
    write_json,
)
from retained_metrics import evaluate_well


STAGE_DIR = OUTPUT_DIR / "stage1"
MODEL_NAMES = ("RF_MULTISCALE", "LOGISTIC_RAW")
DOWNSAMPLE_STRIDES = (2, 4, 5)
GAP_COVERAGES = (0.05, 0.10, 0.20)
GAP_LENGTHS_M = (0.5, 1.0, 2.0)
GAP_REPEATS = 10


def minority_labels(data: pd.DataFrame) -> np.ndarray:
    fraction = data["facies_id"].value_counts().reindex(ALL_LABELS, fill_value=0) / len(data)
    return fraction.loc[fraction < 0.10].index.to_numpy(int)


def summarize_prediction(
    data: pd.DataFrame,
    prediction: np.ndarray,
    probability: np.ndarray,
    scheme: str,
    model_name: str,
    scenario: str,
    minority: np.ndarray,
    context: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    y_true = data["facies_id"].to_numpy(int)
    well_rows = []
    for well_name, positions in data.groupby("well_name", sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        metrics = evaluate_well(
            data.iloc[positions].reset_index(drop=True),
            prediction[positions],
            minority,
        )
        probability_values = probability_metrics(y_true[positions], probability[positions])
        well_rows.append(
            {
                "validation_scheme": scheme,
                "model": model_name,
                "scenario": scenario,
                "context": context,
                "well_name": str(well_name),
                "rows": int(len(positions)),
                **metrics,
                **probability_values,
            }
        )
    well_frame = pd.DataFrame(well_rows)
    pooled = pooled_metrics(y_true, prediction, probability)
    pooled.update(probability_metrics(y_true, probability))
    macro_low, macro_high = bootstrap_mean_ci(well_frame["macro_f1_present"].to_numpy())
    minority_low, minority_high = bootstrap_mean_ci(
        well_frame["minority_mean_recall"].to_numpy()
    )
    boundary_low, boundary_high = bootstrap_mean_ci(
        well_frame["boundary_f1_0p5m"].to_numpy()
    )
    summary = {
        "validation_scheme": scheme,
        "model": model_name,
        "scenario": scenario,
        "context": context,
        "rows": int(len(data)),
        "wells": int(data["well_name"].nunique()),
        "mean_well_macro_f1": float(well_frame["macro_f1_present"].mean()),
        "macro_f1_ci_low": macro_low,
        "macro_f1_ci_high": macro_high,
        "mean_well_minority_recall": float(well_frame["minority_mean_recall"].mean()),
        "minority_recall_ci_low": minority_low,
        "minority_recall_ci_high": minority_high,
        "mean_well_boundary_f1_0p5m": float(well_frame["boundary_f1_0p5m"].mean()),
        "boundary_f1_ci_low": boundary_low,
        "boundary_f1_ci_high": boundary_high,
        **{f"pooled_{key}": value for key, value in pooled.items()},
    }
    class_frame = per_class_metrics(y_true, prediction)
    class_frame.insert(0, "context", context)
    class_frame.insert(0, "scenario", scenario)
    class_frame.insert(0, "model", model_name)
    class_frame.insert(0, "validation_scheme", scheme)
    return summary, well_rows, class_frame.to_dict(orient="records")


def fit_predict_oof(
    data: pd.DataFrame,
    columns: list[str],
    scheme: str,
    model_name: str,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    path = cache_path("1", scheme, model_name, scenario)
    source_row = data["source_row"].to_numpy(int)
    cached = load_valid_prediction_cache(path, source_row)
    if cached is not None:
        return cached[0], cached[1], [
            {
                "validation_scheme": scheme,
                "model": model_name,
                "scenario": scenario,
                "fold": 0,
                "fit_predict_seconds": 0.0,
                "cache_reused": True,
            }
        ]
    y_true = data["facies_id"].to_numpy(int)
    prediction = np.full(len(data), -1, dtype=np.int16)
    probability = np.full((len(data), len(ALL_LABELS)), np.nan, dtype=np.float32)
    timing_rows = []
    prototype = frozen_random_forest() if model_name == "RF_MULTISCALE" else raw_logistic_regression()
    for fold, (train_index, test_index) in enumerate(validation_splits(data, scheme), start=1):
        estimator = clone(prototype)
        start = time.perf_counter()
        estimator.fit(data.iloc[train_index][columns], y_true[train_index])
        fold_prediction = estimator.predict(data.iloc[test_index][columns]).astype(int)
        fold_probability = estimator.predict_proba(data.iloc[test_index][columns])
        from common import align_probabilities

        fold_probability = align_probabilities(np.asarray(estimator.classes_), fold_probability)
        elapsed = time.perf_counter() - start
        prediction[test_index] = fold_prediction
        probability[test_index] = fold_probability.astype(np.float32)
        timing_rows.append(
            {
                "validation_scheme": scheme,
                "model": model_name,
                "scenario": scenario,
                "fold": fold,
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "fit_predict_seconds": float(elapsed),
                "cache_reused": False,
            }
        )
        print(
            f"stage1 {scheme} {model_name} {scenario} fold={fold} "
            f"rows={len(test_index):,} seconds={elapsed:.1f}",
            flush=True,
        )
    if np.any(prediction < 0) or not np.isfinite(probability).all():
        raise RuntimeError(f"Incomplete OOF prediction for {scheme}/{model_name}/{scenario}")
    save_prediction_cache(path, source_row, prediction, probability)
    return prediction.astype(int), probability.astype(float), timing_rows


def run_primary_experiments(
    data: pd.DataFrame,
    multiscale: dict[str, list[str]],
    raw: dict[str, list[str]],
    minority: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    well_rows = []
    class_rows = []
    timing_rows = []
    for scheme in SCHEMES:
        for model_name in MODEL_NAMES:
            for scenario in SCENARIOS:
                columns = multiscale[scenario] if model_name == "RF_MULTISCALE" else raw[scenario]
                prediction, probability, timings = fit_predict_oof(
                    data, columns, scheme, model_name, scenario
                )
                summary, wells, classes = summarize_prediction(
                    data,
                    prediction,
                    probability,
                    scheme,
                    model_name,
                    scenario,
                    minority,
                    "STRICT_GROUPED_OOF",
                )
                summary["feature_count"] = int(len(columns))
                summary["available_logs"] = ";".join(SCENARIOS[scenario])
                summary["legacy_rule_executable"] = bool("GR" in SCENARIOS[scenario] and "DEN" in SCENARIOS[scenario])
                summaries.append(summary)
                well_rows.extend(wells)
                class_rows.extend(classes)
                timing_rows.extend(timings)
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(well_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(timing_rows),
    )


def run_downsample_stress(
    data: pd.DataFrame,
    full_features: list[str],
    minority: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    well_rows = []
    timing_rows = []
    y_true = data["facies_id"].to_numpy(int)
    source_row = data["source_row"].to_numpy(int)
    for scheme in SCHEMES:
        missing = []
        cache_records: dict[int, tuple[np.ndarray, np.ndarray, Path]] = {}
        for stride in DOWNSAMPLE_STRIDES:
            scenario = f"TEST_DOWNSAMPLE_X{stride}_INTERPOLATED"
            path = cache_path("1", scheme, "RF_MULTISCALE", scenario)
            cached = load_valid_prediction_cache(path, source_row)
            if cached is None:
                missing.append(stride)
                cache_records[stride] = (
                    np.full(len(data), -1, dtype=np.int16),
                    np.full((len(data), len(ALL_LABELS)), np.nan, dtype=np.float32),
                    path,
                )
            else:
                cache_records[stride] = (cached[0], cached[1], path)
        if missing:
            for fold, (train_index, test_index) in enumerate(
                validation_splits(data, scheme), start=1
            ):
                estimator = frozen_random_forest()
                start = time.perf_counter()
                estimator.fit(data.iloc[train_index][full_features], y_true[train_index])
                fit_seconds = time.perf_counter() - start
                test_frame = data.iloc[test_index].reset_index(drop=True)
                for stride in missing:
                    degraded = degrade_logs_by_stride(test_frame, stride)
                    fold_probability = estimator.predict_proba(degraded[full_features])
                    from common import align_probabilities

                    fold_probability = align_probabilities(
                        np.asarray(estimator.classes_), fold_probability
                    )
                    fold_prediction = fold_probability.argmax(axis=1)
                    cache_records[stride][0][test_index] = fold_prediction.astype(np.int16)
                    cache_records[stride][1][test_index] = fold_probability.astype(np.float32)
                elapsed = time.perf_counter() - start
                timing_rows.append(
                    {
                        "validation_scheme": scheme,
                        "model": "RF_MULTISCALE",
                        "scenario": "TEST_DOWNSAMPLE_SHARED_FIT",
                        "fold": fold,
                        "train_rows": int(len(train_index)),
                        "test_rows": int(len(test_index)),
                        "fit_seconds": float(fit_seconds),
                        "fit_predict_seconds": float(elapsed),
                        "cache_reused": False,
                    }
                )
                print(
                    f"stage1 downsample {scheme} fold={fold} "
                    f"strides={missing} seconds={elapsed:.1f}",
                    flush=True,
                )
            for stride in missing:
                prediction, probability, path = cache_records[stride]
                if np.any(prediction < 0) or not np.isfinite(probability).all():
                    raise RuntimeError(f"Incomplete downsample prediction: {scheme}/x{stride}")
                save_prediction_cache(path, source_row, prediction, probability)
        for stride in DOWNSAMPLE_STRIDES:
            prediction, probability, _ = cache_records[stride]
            scenario = f"TEST_DOWNSAMPLE_X{stride}_INTERPOLATED"
            summary, wells, _ = summarize_prediction(
                data,
                np.asarray(prediction, dtype=int),
                np.asarray(probability, dtype=float),
                scheme,
                "RF_MULTISCALE_FULL_RESOLUTION_TRAIN",
                scenario,
                minority,
                "TEST_MEASUREMENT_SENSITIVITY",
            )
            summary["effective_anchor_step_m"] = float(0.05 * stride)
            summaries.append(summary)
            well_rows.extend(wells)
    return pd.DataFrame(summaries), pd.DataFrame(well_rows), pd.DataFrame(timing_rows)


def run_naive_row_split(data: pd.DataFrame, full_features: list[str]) -> pd.DataFrame:
    train_index, test_index = train_test_split(
        np.arange(len(data)),
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=data["facies_id"].to_numpy(int),
    )
    estimator = frozen_random_forest()
    estimator.fit(data.iloc[train_index][full_features], data.iloc[train_index]["facies_id"])
    probability = estimator.predict_proba(data.iloc[test_index][full_features])
    from common import align_probabilities

    probability = align_probabilities(np.asarray(estimator.classes_), probability)
    prediction = probability.argmax(axis=1)
    metrics = pooled_metrics(data.iloc[test_index]["facies_id"].to_numpy(int), prediction, probability)
    metrics.update(probability_metrics(data.iloc[test_index]["facies_id"].to_numpy(int), probability))
    train_wells = set(data.iloc[train_index]["well_name"])
    test_wells = set(data.iloc[test_index]["well_name"])
    return pd.DataFrame(
        [
            {
                "validation_scheme": "NAIVE_STRATIFIED_ROW_SPLIT_80_20",
                "model": "RF_MULTISCALE",
                "scenario": "FULL_RT_GR_DEN",
                "context": "LEAKAGE_DIAGNOSTIC_NOT_PRIMARY",
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "well_overlap_count": int(len(train_wells & test_wells)),
                "window_features_precomputed_before_split": True,
                **metrics,
            }
        ]
    )


def run_gap_sensitivity(data: pd.DataFrame, minority: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_rows = []
    summary_rows = []
    for scheme in SCHEMES:
        full_path = cache_path("1", scheme, "RF_MULTISCALE", "FULL_RT_GR_DEN")
        full = load_valid_prediction_cache(full_path, data["source_row"].to_numpy(int))
        if full is None:
            raise RuntimeError(f"Missing full-log cache for {scheme}")
        for missing_log, reduced_scenario in (
            ("DEN", "NO_DEN_RT_GR"),
            ("GR", "NO_GR_RT_DEN"),
        ):
            reduced = load_valid_prediction_cache(
                cache_path("1", scheme, "RF_MULTISCALE", reduced_scenario),
                data["source_row"].to_numpy(int),
            )
            if reduced is None:
                raise RuntimeError(f"Missing reduced-log cache for {scheme}/{reduced_scenario}")
            for coverage in GAP_COVERAGES:
                for length_m in GAP_LENGTHS_M:
                    for repeat in range(1, GAP_REPEATS + 1):
                        mask = contiguous_gap_mask(
                            data,
                            coverage,
                            length_m,
                            stable_seed(scheme, missing_log, coverage, length_m, repeat),
                        )
                        hybrid_prediction = full[0].copy()
                        hybrid_probability = full[1].copy()
                        hybrid_prediction[mask] = reduced[0][mask]
                        hybrid_probability[mask] = reduced[1][mask]
                        summary, _, _ = summarize_prediction(
                            data,
                            hybrid_prediction,
                            hybrid_probability,
                            scheme,
                            "RF_HYBRID_DEPLOYMENT",
                            f"CONTIGUOUS_{missing_log}_GAPS",
                            minority,
                            "LABEL_INDEPENDENT_GAP_SENSITIVITY",
                        )
                        repeat_rows.append(
                            {
                                "validation_scheme": scheme,
                                "missing_log": missing_log,
                                "requested_coverage": coverage,
                                "actual_coverage": float(mask.mean()),
                                "gap_length_m": length_m,
                                "repeat": repeat,
                                "mean_well_macro_f1": summary["mean_well_macro_f1"],
                                "mean_well_minority_recall": summary["mean_well_minority_recall"],
                                "mean_well_boundary_f1_0p5m": summary["mean_well_boundary_f1_0p5m"],
                                "pooled_accuracy": summary["pooled_accuracy"],
                            }
                        )
    repeats = pd.DataFrame(repeat_rows)
    group_columns = [
        "validation_scheme",
        "missing_log",
        "requested_coverage",
        "gap_length_m",
    ]
    for keys, group in repeats.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        row["repeats"] = int(len(group))
        row["mean_actual_coverage"] = float(group["actual_coverage"].mean())
        for column in (
            "mean_well_macro_f1",
            "mean_well_minority_recall",
            "mean_well_boundary_f1_0p5m",
            "pooled_accuracy",
        ):
            values = group[column].to_numpy(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_p025"] = float(np.quantile(values, 0.025))
            row[f"{column}_p975"] = float(np.quantile(values, 0.975))
        summary_rows.append(row)
    return repeats, pd.DataFrame(summary_rows)


def rule_baseline() -> pd.DataFrame:
    audit = json.loads(LABEL_AUDIT.read_text(encoding="utf-8"))
    raw = audit["raw_to_saved_classification"]
    mapping = audit["saved_to_training_csv"]
    return pd.DataFrame(
        [
            {
                "baseline": "LEGACY_DETERMINISTIC_RULE_REEXECUTION",
                "rows": int(raw["total_rows"]),
                "exact_rows": int(raw["exact_rows_all_columns"]),
                "exact_fraction": float(raw["exact_fraction"]),
                "training_rows": int(mapping["training_rows"]),
                "one_to_one_saved_to_training_label_mapping": bool(mapping["one_to_one_mapping"]),
                "requires_GR": True,
                "requires_DEN": True,
                "requires_RT": False,
                "interpretation": "COMPLETE_DATA_CEILING_NOT_A_TRAINED_MODEL",
            }
        ]
    )


def build_report(
    rule: pd.DataFrame,
    primary: pd.DataFrame,
    downsample: pd.DataFrame,
    naive: pd.DataFrame,
    gaps: pd.DataFrame,
) -> str:
    primary_rf = primary.loc[primary["model"].eq("RF_MULTISCALE")]
    lines = [
        "# NRR Stage 1: Rule baseline and strict generalization",
        "",
        "## Evidence boundary",
        "",
        "The targets are deterministic operational well-log labels. Scores quantify reproducibility or sensitivity, not independent depositional-facies validity.",
        "",
        "## Deterministic ceiling",
        "",
        f"The legacy rule reproduced {int(rule.iloc[0].exact_rows):,}/{int(rule.iloc[0].rows):,} saved rows exactly (fraction {rule.iloc[0].exact_fraction:.6f}).",
        "",
        "## Strict grouped random-forest results",
        "",
        "| Validation | Scenario | Mean well macro F1 | Minority recall | Boundary F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in primary_rf.itertuples(index=False):
        lines.append(
            f"| {row.validation_scheme} | {row.scenario} | {row.mean_well_macro_f1:.4f} | "
            f"{row.mean_well_minority_recall:.4f} | {row.mean_well_boundary_f1_0p5m:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Naive row-split diagnostic",
            "",
            f"The intentionally naive split has well overlap count {int(naive.iloc[0].well_overlap_count)} and pooled macro F1 {naive.iloc[0].macro_f1:.4f}. It is not used for a generalization claim.",
            "",
            "## Measurement sensitivity",
            "",
            f"Downsampling produced {len(downsample)} strict grouped summaries. Contiguous-gap sensitivity produced {len(gaps)} scenario summaries.",
            "",
            "## Claim gate",
            "",
            "Machine learning may be discussed as a workflow-continuity mechanism only where the required rule inputs are unavailable or degraded. With complete GR and DEN, the deterministic rule remains the exact operational baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    association, data, wells, picks, multiscale, raw = build_analysis_data()
    minority = minority_labels(data)
    rule = rule_baseline()
    split_audit = pd.DataFrame(split_audit_rows(data))
    primary, per_well, per_class, timings = run_primary_experiments(
        data, multiscale, raw, minority
    )
    downsample, downsample_wells, downsample_timings = run_downsample_stress(
        data, multiscale["FULL_RT_GR_DEN"], minority
    )
    naive = run_naive_row_split(data, multiscale["FULL_RT_GR_DEN"])
    gap_repeats, gap_summary = run_gap_sensitivity(data, minority)

    rule.to_csv(STAGE_DIR / "deterministic_rule_baseline.csv", index=False, encoding="utf-8-sig")
    split_audit.to_csv(STAGE_DIR / "strict_split_audit.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(STAGE_DIR / "strict_experiment_summary.csv", index=False, encoding="utf-8-sig")
    per_well.to_csv(STAGE_DIR / "strict_per_well_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(STAGE_DIR / "strict_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    timings.to_csv(STAGE_DIR / "strict_fit_timing.csv", index=False, encoding="utf-8-sig")
    downsample.to_csv(STAGE_DIR / "downsample_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    downsample_wells.to_csv(STAGE_DIR / "downsample_sensitivity_per_well.csv", index=False, encoding="utf-8-sig")
    downsample_timings.to_csv(STAGE_DIR / "downsample_fit_timing.csv", index=False, encoding="utf-8-sig")
    naive.to_csv(STAGE_DIR / "naive_row_split_diagnostic.csv", index=False, encoding="utf-8-sig")
    gap_repeats.to_csv(STAGE_DIR / "contiguous_gap_repeats.csv", index=False, encoding="utf-8-sig")
    gap_summary.to_csv(STAGE_DIR / "contiguous_gap_summary.csv", index=False, encoding="utf-8-sig")
    feature_manifest = []
    for scenario in SCENARIOS:
        feature_manifest.append(
            {
                "scenario": scenario,
                "available_logs": ";".join(SCENARIOS[scenario]),
                "rf_feature_count": len(multiscale[scenario]),
                "rf_features": ";".join(multiscale[scenario]),
                "logistic_feature_count": len(raw[scenario]),
                "logistic_features": ";".join(raw[scenario]),
                "legacy_rule_executable": bool("GR" in SCENARIOS[scenario] and "DEN" in SCENARIOS[scenario]),
            }
        )
    pd.DataFrame(feature_manifest).to_csv(
        STAGE_DIR / "feature_scenario_manifest.csv", index=False, encoding="utf-8-sig"
    )

    checks = {
        "source_rows_101639": int(len(association)) == 101_639,
        "analysis_rows_101615": int(len(data)) == 101_615,
        "ten_wells": int(data["well_name"].nunique()) == 10,
        "three_spatial_blocks": int(data["spatial_block"].nunique()) == 3,
        "rule_exact_reexecution": bool(math.isclose(float(rule.iloc[0].exact_fraction), 1.0)),
        "strict_splits_disjoint": bool(
            split_audit["well_overlap_count"].eq(0).all()
            and split_audit["group_overlap_count"].eq(0).all()
        ),
        "twenty_primary_experiments": int(len(primary)) == 20,
        "six_downsample_experiments": int(len(downsample)) == 6,
        "naive_split_explicitly_overlaps_wells": int(naive.iloc[0].well_overlap_count) == 10,
        "gap_sensitivity_complete": int(len(gap_summary))
        == len(SCHEMES) * 2 * len(GAP_COVERAGES) * len(GAP_LENGTHS_M),
    }
    state = {
        "package": "nrr_stage1",
        "generated_on": str(pd.Timestamp.now(tz="Asia/Shanghai")),
        "overall_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_hashes": source_hashes(),
        "counts": {
            "source_rows": int(len(association)),
            "analysis_rows": int(len(data)),
            "wells": int(data["well_name"].nunique()),
            "spatial_blocks": int(data["spatial_block"].nunique()),
            "authorized_target_intervals": int(len(picks)),
            "primary_experiments": int(len(primary)),
            "downsample_experiments": int(len(downsample)),
            "gap_summaries": int(len(gap_summary)),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "claim_boundary": [
            "The deterministic rule is the complete-GR-and-DEN operational ceiling.",
            "Strict grouped metrics quantify operational-label reproducibility, not independent facies truth.",
            "Downsampling and contiguous gaps are sensitivity tests, not documented tool-error distributions.",
        ],
    }
    write_json(STAGE_DIR / "stage1_quality_gate.json", state)
    (STAGE_DIR / "STAGE1_REPORT.md").write_text(
        build_report(rule, primary, downsample, naive, gap_summary), encoding="utf-8"
    )
    build_manifest(STAGE_DIR)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
