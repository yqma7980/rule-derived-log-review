from __future__ import annotations

import importlib.util
import json
import math
import os
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
ANALYSIS_DIR = REPO / "IJCG投稿" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from common import ALL_LABELS, ID_TO_CODE, add_window_features, bootstrap_mean_ci  # noqa: E402
from nrr_stage_common import (  # noqa: E402
    SCHEMES,
    build_analysis_data,
    cache_path,
    frozen_random_forest,
    load_valid_prediction_cache,
    run_intervals,
    source_hashes,
    validation_splits,
)
from run_nrr_stage1 import minority_labels  # noqa: E402


EVIDENCE = STAGE_DIR / "evidence"
RULE_RESOURCE = STAGE_DIR / "online_resources" / "Online_Resource_2_rule_code"
SOURCE_STAGE1 = REPO / "IJCG投稿" / "results" / "nrr_stage1_stage2" / "stage1"
PRIMARY_SCENARIO = "NO_RT_GR_DEN"
PRIMARY_MODEL = "RF_MULTISCALE"
PRIMARY_LABEL = "GR_DEN_MULTISCALE_POST_HOC"
RF_JOBS = int(os.environ.get("NRR_RF_JOBS", "1"))
BOUNDARY_TOLERANCES_M = (0.05, 0.10, 0.25, 0.50)
DOWNSAMPLE_STRIDES = (1, 2, 4, 5)
CANDIDATES = (
    ("RF_MULTISCALE", "NO_RT_GR_DEN"),
    ("RF_MULTISCALE", "FULL_RT_GR_DEN"),
    ("RF_MULTISCALE", "NO_DEN_RT_GR"),
    ("RF_MULTISCALE", "NO_GR_RT_DEN"),
    ("RF_MULTISCALE", "RT_ONLY"),
    ("LOGISTIC_RAW", "FULL_RT_GR_DEN"),
)


def load_rule_module():
    path = RULE_RESOURCE / "operational_rule.py"
    spec = importlib.util.spec_from_file_location("online_resource_operational_rule", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load rule resource: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_rule_by_group(
    frame: pd.DataFrame,
    group_columns: list[str],
    rule,
) -> np.ndarray:
    prediction = np.full(len(frame), -1, dtype=np.int16)
    for positions in frame.groupby(group_columns, sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        local = frame.iloc[positions][["depth", "GR", "DEN"]].reset_index(drop=True)
        # Preserve the historical row-indexed calculation for source replay.
        interpreted = rule.apply_operational_rule(
            local,
            enforce_nominal_grid=False,
        )
        prediction[positions] = interpreted["Subfacies"].to_numpy(np.int16) - 1
    if np.any(prediction < 0):
        raise RuntimeError("Operational rule did not return every row")
    return prediction


def write_rule_specification(rule, source_replay: dict[str, object]) -> None:
    rows = [
        ("Input", "RT", "ohm m", "Recorded but not used by the deterministic rule"),
        ("Input", "GR", "API", "Required natural-gamma measurement"),
        ("Input", "DEN", "g/cm^3", "Required bulk-density measurement"),
        ("QC", "RT", "ohm m", "Exclude RT <= 0 from model evaluation"),
        ("QC", "GR", "API", "Exclude GR <= 0 from model evaluation"),
        ("QC", "DEN", "g/cm^3", "Exclude DEN <= 0 or DEN > 5 from model evaluation"),
        ("Lithology priority 1", "Coal", "", "DEN < 1.75 and GR < 70"),
        ("Lithology priority 2", "Mudstone", "", "GR >= 80 and DEN >= 2.30"),
        ("Lithology priority 3", "Siltstone", "", "60 <= GR < 80"),
        ("Lithology priority 4", "Fine sandstone", "", "40 <= GR < 60"),
        ("Lithology priority 5", "Medium sandstone", "", "20 <= GR < 40"),
        ("Lithology priority 6", "Coarse sandstone/conglomerate", "", "GR < 20"),
        ("Lithology fallback", "Mudstone", "", "All remaining nonmissing combinations"),
        ("Thickness", "Run thickness", "m", "Consecutive identical lithology rows x 0.05 m"),
        ("Context", "Window", "rows", "[i-10, i+10), clipped at segment limits"),
        ("Subfacies priority 1", "SW", "", "Lithology is coal"),
        ("Subfacies priority 2", "DLM", "", "Lithology is mudstone and GR > 90"),
        ("Subfacies priority 2 fallback", "SLM", "", "Other mudstone rows"),
        ("Subfacies priority 3", "DCH", "", "Medium/coarse lithology and thickness > 3.0 m"),
        (
            "Subfacies priority 4",
            "MB",
            "",
            "Fine sandstone; next five deeper rows contain mudstone/siltstone and previous five shallower rows contain medium/coarse sandstone",
        ),
        (
            "Subfacies priority 5",
            "SS",
            "",
            "Siltstone/fine sandstone; thickness < 2.0 m; 20-row window has >=3 lithologies and contains both siltstone and fine sandstone",
        ),
        ("Subfacies fallback", "SS/MB/DCH/SLM", "", "Map lithology 2/3/4-5/other to class 4/3/2/5"),
    ]
    pd.DataFrame(rows, columns=["stage", "quantity_or_class", "unit", "exact_definition"]).to_csv(
        EVIDENCE / "deterministic_rule_specification.csv", index=False, encoding="utf-8-sig"
    )
    (EVIDENCE / "deterministic_rule_replay.json").write_text(
        json.dumps(source_replay, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def degrade_gr_den_by_phase(frame: pd.DataFrame, stride: int, phase: int) -> pd.DataFrame:
    if stride == 1:
        if phase != 0:
            raise ValueError("Stride 1 only permits phase 0")
        return frame.copy()
    if not 0 <= phase < stride:
        raise ValueError(f"Phase {phase} is invalid for stride {stride}")
    out = frame.copy().reset_index(drop=True)
    for positions in out.groupby(["well_name", "segment_id"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        if len(positions) < 2:
            continue
        depth = out.iloc[positions]["depth"].to_numpy(float)
        phase_anchors = np.arange(phase, len(positions), stride, dtype=int)
        anchors = np.unique(np.concatenate(([0], phase_anchors, [len(positions) - 1]))).astype(int)
        for log_name in ("GR", "DEN"):
            values = out.iloc[positions][log_name].to_numpy(float)
            out.loc[positions, log_name] = np.interp(depth, depth[anchors], values[anchors])
    degraded, _ = add_window_features(out)
    return degraded


def per_well_classification(
    data: pd.DataFrame,
    prediction: np.ndarray,
    scheme: str,
    method: str,
    stride: int,
    phase: int,
    minority: np.ndarray,
) -> list[dict[str, object]]:
    y_true = data["facies_id"].to_numpy(int)
    rows: list[dict[str, object]] = []
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
                "phase": phase,
                "effective_anchor_spacing_m": 0.05 * stride,
                "well_name": str(well_name),
                "accuracy": float(np.mean(truth == predicted)),
                "macro_f1_present": float(
                    f1_score(truth, predicted, labels=present, average="macro", zero_division=0)
                ),
                "minority_mean_recall": float(
                    recall_score(truth, predicted, labels=minority, average="macro", zero_division=0)
                ),
            }
        )
    return rows


def run_phase_complete_downsampling(
    data: pd.DataFrame,
    features: list[str],
    rule,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = [(1, 0)] + [
        (stride, phase)
        for stride in DOWNSAMPLE_STRIDES[1:]
        for phase in range(stride)
    ]
    y_true = data["facies_id"].to_numpy(int)
    source_row = data["source_row"].to_numpy(int)
    minority = minority_labels(data)
    all_rows: list[dict[str, object]] = []

    fitted_folds: dict[str, list[tuple[np.ndarray, object]]] = {}
    for scheme in SCHEMES:
        fitted_folds[scheme] = []
        for fold, (train_index, test_index) in enumerate(validation_splits(data, scheme), start=1):
            estimator = frozen_random_forest()
            estimator.set_params(n_jobs=RF_JOBS)
            estimator.fit(data.iloc[train_index][features], y_true[train_index])
            fitted_folds[scheme].append((test_index, estimator))
            print(f"Fitted downsampling reference {scheme} fold {fold}", flush=True)

    for scenario_number, (stride, phase) in enumerate(scenarios, start=1):
        degraded = degrade_gr_den_by_phase(data, stride, phase)
        rule_prediction = apply_rule_by_group(
            degraded, ["well_name", "segment_id"], rule
        )
        for scheme in SCHEMES:
            if (stride, phase) == (1, 0):
                cached = load_valid_prediction_cache(
                    cache_path("1", scheme, PRIMARY_MODEL, PRIMARY_SCENARIO), source_row
                )
                if cached is None:
                    raise RuntimeError(f"Missing primary prediction cache for {scheme}")
                prediction = cached[0]
            else:
                prediction = np.full(len(data), -1, dtype=np.int16)
                for test_index, estimator in fitted_folds[scheme]:
                    prediction[test_index] = estimator.predict(
                        degraded.iloc[test_index][features]
                    ).astype(np.int16)
                if np.any(prediction < 0):
                    raise RuntimeError(
                        f"Incomplete prediction for {scheme}, stride={stride}, phase={phase}"
                    )
            all_rows.extend(
                per_well_classification(
                    data,
                    prediction,
                    scheme,
                    "RF_GR_DEN_MULTISCALE",
                    stride,
                    phase,
                    minority,
                )
            )
            all_rows.extend(
                per_well_classification(
                    data,
                    rule_prediction,
                    scheme,
                    "DETERMINISTIC_RULE_QC_SEGMENT_REEXECUTION",
                    stride,
                    phase,
                    minority,
                )
            )
        print(
            f"Completed resolution scenario {scenario_number}/{len(scenarios)}: "
            f"stride={stride}, phase={phase}",
            flush=True,
        )

    per_well = pd.DataFrame(all_rows)
    phase_rows: list[dict[str, object]] = []
    for keys, group in per_well.groupby(
        ["validation_scheme", "method", "stride", "phase", "effective_anchor_spacing_m"],
        sort=True,
    ):
        scheme, method, stride, phase, spacing = keys
        phase_rows.append(
            {
                "validation_scheme": scheme,
                "method": method,
                "stride": int(stride),
                "phase": int(phase),
                "effective_anchor_spacing_m": float(spacing),
                "mean_per_well_accuracy": float(group["accuracy"].mean()),
                "mean_per_well_macro_f1": float(group["macro_f1_present"].mean()),
                "mean_per_well_minority_recall": float(group["minority_mean_recall"].mean()),
            }
        )
    by_phase = pd.DataFrame(phase_rows)

    aggregate_rows: list[dict[str, object]] = []
    for keys, group in by_phase.groupby(
        ["validation_scheme", "method", "stride", "effective_anchor_spacing_m"], sort=True
    ):
        scheme, method, stride, spacing = keys
        row: dict[str, object] = {
            "validation_scheme": scheme,
            "method": method,
            "stride": int(stride),
            "effective_anchor_spacing_m": float(spacing),
            "phase_count": int(group["phase"].nunique()),
        }
        for metric in (
            "mean_per_well_accuracy",
            "mean_per_well_macro_f1",
            "mean_per_well_minority_recall",
        ):
            row[f"phase_mean_{metric}"] = float(group[metric].mean())
            row[f"phase_min_{metric}"] = float(group[metric].min())
            row[f"phase_max_{metric}"] = float(group[metric].max())
        aggregate_rows.append(row)
    return per_well, by_phase, pd.DataFrame(aggregate_rows)


def transition_records(depth: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    changes = np.flatnonzero(labels[1:] != labels[:-1])
    return pd.DataFrame(
        {
            "depth_m": 0.5 * (depth[changes] + depth[changes + 1]),
            "from_id": labels[changes].astype(int),
            "to_id": labels[changes + 1].astype(int),
        }
    )


def nearest_first_one_to_one_matches(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    tolerance_m: float,
    transition_aware: bool,
) -> list[tuple[int, int, float]]:
    """Greedily accept globally nearest compatible pairs with deterministic ties."""
    candidates: list[tuple[float, int, int]] = []
    predicted_depth = predicted["depth_m"].to_numpy(float)
    predicted_from = predicted["from_id"].to_numpy(int)
    predicted_to = predicted["to_id"].to_numpy(int)
    for observed_index, row in enumerate(observed.itertuples(index=False)):
        distance = np.abs(predicted_depth - float(row.depth_m))
        compatible = distance <= tolerance_m + 1e-12
        if transition_aware:
            compatible &= predicted_from == int(row.from_id)
            compatible &= predicted_to == int(row.to_id)
        for predicted_index in np.flatnonzero(compatible):
            candidates.append((float(distance[predicted_index]), observed_index, int(predicted_index)))
    used_observed: set[int] = set()
    used_predicted: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for distance, observed_index, predicted_index in sorted(candidates):
        if observed_index in used_observed or predicted_index in used_predicted:
            continue
        used_observed.add(observed_index)
        used_predicted.add(predicted_index)
        matches.append((observed_index, predicted_index, distance))
    return matches


def boundary_evaluation(
    data: pd.DataFrame,
    prediction: np.ndarray,
    scheme: str,
    tolerances: tuple[float, ...],
    metric_types: tuple[str, ...] = ("CLASS_AGNOSTIC", "TRANSITION_AWARE"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = data["facies_id"].to_numpy(int)
    count_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    for (well_name, segment_id), positions in data.groupby(
        ["well_name", "segment_id"], sort=False
    ).indices.items():
        positions = np.asarray(positions, dtype=int)
        depth = data.iloc[positions]["depth"].to_numpy(float)
        observed = transition_records(depth, y_true[positions])
        predicted = transition_records(depth, prediction[positions])
        for tolerance in tolerances:
            for metric_type in metric_types:
                matches = nearest_first_one_to_one_matches(
                    observed,
                    predicted,
                    tolerance,
                    transition_aware=metric_type == "TRANSITION_AWARE",
                )
                count_rows.append(
                    {
                        "validation_scheme": scheme,
                        "metric_type": metric_type,
                        "tolerance_m": tolerance,
                        "well_name": str(well_name),
                        "segment_id": int(segment_id),
                        "observed_boundaries": int(len(observed)),
                        "predicted_boundaries": int(len(predicted)),
                        "matched_boundaries": int(len(matches)),
                    }
                )
                for observed_index, predicted_index, offset in matches:
                    observed_row = observed.iloc[observed_index]
                    predicted_row = predicted.iloc[predicted_index]
                    match_rows.append(
                        {
                            "validation_scheme": scheme,
                            "metric_type": metric_type,
                            "tolerance_m": tolerance,
                            "well_name": str(well_name),
                            "segment_id": int(segment_id),
                            "observed_boundary_depth_m": float(observed_row["depth_m"]),
                            "predicted_boundary_depth_m": float(predicted_row["depth_m"]),
                            "absolute_offset_m": offset,
                            "observed_from_code": ID_TO_CODE[int(observed_row["from_id"])],
                            "observed_to_code": ID_TO_CODE[int(observed_row["to_id"])],
                            "predicted_from_code": ID_TO_CODE[int(predicted_row["from_id"])],
                            "predicted_to_code": ID_TO_CODE[int(predicted_row["to_id"])],
                        }
                    )

    counts = pd.DataFrame(count_rows)
    matches = pd.DataFrame(match_rows)
    summary_rows: list[dict[str, object]] = []
    for (metric_type, tolerance), group in counts.groupby(
        ["metric_type", "tolerance_m"], sort=True
    ):
        well_rows: list[dict[str, object]] = []
        for well_name, well in group.groupby("well_name", sort=True):
            observed_count = int(well["observed_boundaries"].sum())
            predicted_count = int(well["predicted_boundaries"].sum())
            matched_count = int(well["matched_boundaries"].sum())
            precision = matched_count / predicted_count if predicted_count else math.nan
            recall = matched_count / observed_count if observed_count else math.nan
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            well_rows.append({"well_name": str(well_name), "f1": f1})
        wells = pd.DataFrame(well_rows)
        observed_count = int(group["observed_boundaries"].sum())
        predicted_count = int(group["predicted_boundaries"].sum())
        matched_count = int(group["matched_boundaries"].sum())
        precision = matched_count / predicted_count
        recall = matched_count / observed_count
        pooled_f1 = 2 * precision * recall / (precision + recall)
        offsets = matches.loc[
            matches["metric_type"].eq(metric_type)
            & np.isclose(matches["tolerance_m"], tolerance),
            "absolute_offset_m",
        ].to_numpy(float)
        low, high = bootstrap_mean_ci(wells["f1"].to_numpy(float))
        summary_rows.append(
            {
                "validation_scheme": scheme,
                "metric_type": metric_type,
                "tolerance_m": float(tolerance),
                "observed_boundaries": observed_count,
                "predicted_boundaries": predicted_count,
                "matched_boundaries": matched_count,
                "pooled_precision": precision,
                "pooled_recall": recall,
                "pooled_f1": pooled_f1,
                "mean_per_well_f1": float(wells["f1"].mean()),
                "mean_per_well_f1_ci_low": low,
                "mean_per_well_f1_ci_high": high,
                "median_offset_m": float(np.median(offsets)) if len(offsets) else math.nan,
                "p95_offset_m": float(np.quantile(offsets, 0.95)) if len(offsets) else math.nan,
            }
        )
    return counts, matches, pd.DataFrame(summary_rows)


def recompute_boundary_tables(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_row = data["source_row"].to_numpy(int)
    sensitivity_counts: list[pd.DataFrame] = []
    sensitivity_matches: list[pd.DataFrame] = []
    sensitivity_summary: list[pd.DataFrame] = []
    strict_rows: list[dict[str, object]] = []
    strict_source = pd.read_csv(SOURCE_STAGE1 / "strict_experiment_summary.csv")

    for scheme in SCHEMES:
        for model, scenario in CANDIDATES:
            cached = load_valid_prediction_cache(cache_path("1", scheme, model, scenario), source_row)
            if cached is None:
                raise RuntimeError(f"Missing prediction cache for {scheme}/{model}/{scenario}")
            prediction = cached[0]
            if model == PRIMARY_MODEL and scenario == PRIMARY_SCENARIO:
                counts, matches, summary = boundary_evaluation(
                    data, prediction, scheme, BOUNDARY_TOLERANCES_M
                )
                sensitivity_counts.append(counts)
                sensitivity_matches.append(matches)
                sensitivity_summary.append(summary)
                agnostic_050 = summary.loc[
                    summary["metric_type"].eq("CLASS_AGNOSTIC")
                    & np.isclose(summary["tolerance_m"], 0.50)
                ].iloc[0]
                agnostic_005 = summary.loc[
                    summary["metric_type"].eq("CLASS_AGNOSTIC")
                    & np.isclose(summary["tolerance_m"], 0.05)
                ].iloc[0]
            else:
                _, _, summary = boundary_evaluation(
                    data,
                    prediction,
                    scheme,
                    (0.05, 0.50),
                    metric_types=("CLASS_AGNOSTIC",),
                )
                agnostic_005 = summary.loc[np.isclose(summary["tolerance_m"], 0.05)].iloc[0]
                agnostic_050 = summary.loc[np.isclose(summary["tolerance_m"], 0.50)].iloc[0]
            source = strict_source.loc[
                strict_source["validation_scheme"].eq(scheme)
                & strict_source["model"].eq(model)
                & strict_source["scenario"].eq(scenario)
            ].iloc[0]
            strict_rows.append(
                {
                    "validation_scheme": scheme,
                    "model": model,
                    "scenario": scenario,
                    "mean_per_well_macro_f1": float(source["mean_well_macro_f1"]),
                    "mean_per_well_minority_recall": float(source["mean_well_minority_recall"]),
                    "unified_class_agnostic_boundary_f1_0p05m": float(
                        agnostic_005["mean_per_well_f1"]
                    ),
                    "unified_class_agnostic_boundary_f1_0p50m": float(
                        agnostic_050["mean_per_well_f1"]
                    ),
                    "matching_algorithm": "GLOBAL_NEAREST_FIRST_ONE_TO_ONE",
                }
            )
            print(f"Boundary audit {scheme} {model} {scenario}", flush=True)

    return (
        pd.concat(sensitivity_counts, ignore_index=True),
        pd.concat(sensitivity_matches, ignore_index=True),
        pd.concat(sensitivity_summary, ignore_index=True),
        pd.DataFrame(strict_rows),
    )


def run_thickness_distribution(data: pd.DataFrame) -> pd.DataFrame:
    runs = run_intervals(data, data["facies_id"].to_numpy(int))
    rows = []
    for threshold in BOUNDARY_TOLERANCES_M:
        count = int(runs["thickness_m"].le(threshold + 1e-12).sum())
        rows.append(
            {
                "threshold_m": threshold,
                "runs_at_or_below_threshold": count,
                "total_runs": int(len(runs)),
                "fraction": count / len(runs),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    source_replay: dict[str, object],
    boundary_summary: pd.DataFrame,
    downsample_aggregate: pd.DataFrame,
) -> None:
    lines = [
        "# NRR Stage 6 submission-readiness evidence",
        "",
        "## Deterministic rule",
        "",
        f"- Source replay: {source_replay['matched_rows']}/{source_replay['source_rows']} rows.",
        "- Full thresholds, units, inequality directions, and window indexing are in deterministic_rule_specification.csv.",
        "- The executable rule is Online Resource 2 and contains no controlled measurements or coordinates.",
        "",
        "## Boundary tolerance sensitivity",
        "",
        "A single globally nearest-first one-to-one matcher was used for both class-agnostic and transition-aware metrics.",
        "",
    ]
    for row in boundary_summary.itertuples(index=False):
        lines.append(
            f"- {row.validation_scheme}, {row.metric_type}, {row.tolerance_m:.2f} m: "
            f"mean per-well F1={row.mean_per_well_f1:.4f}; pooled F1={row.pooled_f1:.4f}."
        )
    lines.extend(["", "## Downsampling phase sensitivity", ""])
    for row in downsample_aggregate.itertuples(index=False):
        lines.append(
            f"- {row.validation_scheme}, {row.method}, {row.effective_anchor_spacing_m:.2f} m, "
            f"{row.phase_count} phase(s): macro F1 mean="
            f"{row.phase_mean_mean_per_well_macro_f1:.4f}, range "
            f"{row.phase_min_mean_per_well_macro_f1:.4f}--"
            f"{row.phase_max_mean_per_well_macro_f1:.4f}."
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "The GR+DEN analysis model was selected post hoc from the same grouped results used for reporting. Its score and paired delta are exploratory, not independently confirmed model-selection estimates.",
            "",
            "All targets remain deterministic operational interpretations. The calculations do not validate true facies, net coal, resources, reserves, permeability, or production.",
            "",
        ]
    )
    (EVIDENCE / "STAGE6_EVIDENCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    association, data, _, _, multiscale, _ = build_analysis_data()
    if len(association) != 101_639 or len(data) != 101_615:
        raise RuntimeError("Unexpected source or eligible population")
    rule = load_rule_module()

    full_rule = apply_rule_by_group(association, ["well_name"], rule)
    source_truth = association["facies_id"].to_numpy(int)
    source_replay = {
        "source_rows": int(len(association)),
        "matched_rows": int(np.sum(full_rule == source_truth)),
        "mismatched_rows": int(np.sum(full_rule != source_truth)),
        "grouping": "complete well",
        "row_order": "increasing measured depth",
    }
    write_rule_specification(rule, source_replay)

    per_well, by_phase, downsample_aggregate = run_phase_complete_downsampling(
        data, multiscale[PRIMARY_SCENARIO], rule
    )
    per_well.to_csv(EVIDENCE / "downsample_all_phases_per_well.csv", index=False, encoding="utf-8-sig")
    by_phase.to_csv(EVIDENCE / "downsample_all_phases_summary.csv", index=False, encoding="utf-8-sig")
    downsample_aggregate.to_csv(
        EVIDENCE / "downsample_phase_aggregate.csv", index=False, encoding="utf-8-sig"
    )

    boundary_counts, boundary_matches, boundary_summary, strict_unified = recompute_boundary_tables(data)
    boundary_counts.to_csv(
        EVIDENCE / "boundary_tolerance_count_audit.csv", index=False, encoding="utf-8-sig"
    )
    boundary_matches.to_csv(
        EVIDENCE / "boundary_tolerance_matches.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    boundary_summary.to_csv(
        EVIDENCE / "boundary_tolerance_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    strict_unified.to_csv(
        EVIDENCE / "strict_model_unified_boundary.csv", index=False, encoding="utf-8-sig"
    )
    run_thickness_distribution(data).to_csv(
        EVIDENCE / "operational_run_thickness_distribution.csv", index=False, encoding="utf-8-sig"
    )

    primary_050 = boundary_summary.loc[
        boundary_summary["metric_type"].eq("CLASS_AGNOSTIC")
        & np.isclose(boundary_summary["tolerance_m"], 0.50)
    ].set_index("validation_scheme")["mean_per_well_f1"]
    strict_050 = strict_unified.loc[
        strict_unified["model"].eq(PRIMARY_MODEL)
        & strict_unified["scenario"].eq(PRIMARY_SCENARIO)
    ].set_index("validation_scheme")["unified_class_agnostic_boundary_f1_0p50m"]
    expected_phase_counts = {1: 1, 2: 2, 4: 4, 5: 5}
    observed_phase_counts = (
        by_phase.loc[by_phase["method"].eq("RF_GR_DEN_MULTISCALE")]
        .groupby("stride")["phase"]
        .nunique()
        .to_dict()
    )
    checks = {
        "source_rule_replay_exact": source_replay["mismatched_rows"] == 0,
        "single_boundary_algorithm": strict_unified["matching_algorithm"].nunique() == 1,
        "primary_boundary_table_consistent": bool(
            np.allclose(primary_050.sort_index(), strict_050.sort_index(), atol=1e-12)
        ),
        "all_boundary_tolerances_present": set(np.round(boundary_summary["tolerance_m"], 2))
        == set(BOUNDARY_TOLERANCES_M),
        "both_boundary_types_present": set(boundary_summary["metric_type"])
        == {"CLASS_AGNOSTIC", "TRANSITION_AWARE"},
        "all_downsample_phases_present": observed_phase_counts == expected_phase_counts,
        "post_hoc_primary_explicit": PRIMARY_LABEL.endswith("POST_HOC"),
    }
    validation = {
        "stage": "NRR_SUBMISSION_READINESS_STAGE6",
        "created_local_date": "2026-07-31",
        "runtime_seconds": float(time.perf_counter() - started),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "random_forest_jobs": RF_JOBS,
        "source_hashes": source_hashes(),
        "checks": checks,
    }
    (EVIDENCE / "stage6_analysis_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(source_replay, boundary_summary, downsample_aggregate)
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Stage 6 analysis checks failed: {failed}")
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
