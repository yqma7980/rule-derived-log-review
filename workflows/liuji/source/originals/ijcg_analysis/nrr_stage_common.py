from __future__ import annotations

import hashlib
import json
import math
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    ALL_LABELS,
    EXPECTED_SAMPLE_STEP_M,
    RAW_FEATURES,
    add_segments,
    add_window_features,
    align_probabilities,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT_DIR = ROOT / "results" / "nrr_stage1_stage2"
EXISTING_GEOLOGY_OUTPUT = ROOT / "outputs" / "geology_aware_subfacies_validation_v1"
LABEL_AUDIT = ROOT / "results" / "project_audit" / "label_provenance_audit.json"
RANDOM_SEED = 42
SCHEMES = ("LEAVE_ONE_WELL_OUT", "LEAVE_ONE_SPATIAL_BLOCK_OUT")

SCENARIOS: dict[str, tuple[str, ...]] = {
    "FULL_RT_GR_DEN": ("RT", "GR", "DEN"),
    "NO_RT_GR_DEN": ("GR", "DEN"),
    "NO_DEN_RT_GR": ("RT", "GR"),
    "NO_GR_RT_DEN": ("RT", "DEN"),
    "RT_ONLY": ("RT",),
}


def stable_seed(*parts: object) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return (RANDOM_SEED + zlib.crc32(value)) % (2**32 - 1)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_analysis_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, list[str]],
    dict[str, list[str]],
]:
    # Historical geology-building branch; the public runner uses fixed groups.
    # Its CAD modules are outside this release's main-result replay scope.
    from run_geology_aware_subfacies_validation import build_association_table
    association, wells, picks = build_association_table()
    analysis = association.loc[association["analysis_eligible"]].copy()
    analysis = add_segments(analysis)
    data, feature_groups = add_window_features(analysis)
    full_features = feature_groups["raw_through_w100"]
    multiscale = {
        name: select_log_features(full_features, logs)
        for name, logs in SCENARIOS.items()
    }
    raw = {name: list(logs) for name, logs in SCENARIOS.items()}
    if data[full_features].isna().any().any():
        raise ValueError("Analysis features contain missing values")
    return association, data, wells, picks, multiscale, raw


def select_log_features(columns: Iterable[str], logs: Iterable[str]) -> list[str]:
    selected = []
    for column in columns:
        if any(column == log or column.startswith(f"{log}_") for log in logs):
            selected.append(column)
    if not selected:
        raise ValueError("No features selected for log scenario")
    return selected


def frozen_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def raw_logistic_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def validation_splits(
    data: pd.DataFrame, scheme: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown validation scheme: {scheme}")
    group_column = "well_name" if scheme == "LEAVE_ONE_WELL_OUT" else "spatial_block"
    splitter = LeaveOneGroupOut()
    groups = data[group_column].to_numpy()
    return list(splitter.split(data, data["facies_id"].to_numpy(int), groups))


def split_audit_rows(data: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for scheme in SCHEMES:
        group_column = "well_name" if scheme == "LEAVE_ONE_WELL_OUT" else "spatial_block"
        for fold, (train_index, test_index) in enumerate(
            validation_splits(data, scheme), start=1
        ):
            train = data.iloc[train_index]
            test = data.iloc[test_index]
            train_wells = set(train["well_name"])
            test_wells = set(test["well_name"])
            train_groups = set(train[group_column])
            test_groups = set(test[group_column])
            rows.append(
                {
                    "validation_scheme": scheme,
                    "fold": fold,
                    "group_column": group_column,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_wells": ";".join(sorted(train_wells)),
                    "test_wells": ";".join(sorted(test_wells)),
                    "train_groups": ";".join(map(str, sorted(train_groups))),
                    "test_groups": ";".join(map(str, sorted(test_groups))),
                    "well_overlap_count": int(len(train_wells & test_wells)),
                    "group_overlap_count": int(len(train_groups & test_groups)),
                }
            )
    return rows


def cache_path(stage: str, scheme: str, model: str, scenario: str) -> Path:
    directory = OUTPUT_DIR / f"stage{stage}_cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{scheme.lower()}__{model.lower()}__{scenario.lower()}.npz"


def save_prediction_cache(
    path: Path,
    source_row: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        source_row=np.asarray(source_row, dtype=np.int32),
        prediction=np.asarray(prediction, dtype=np.int16),
        probability=np.asarray(probability, dtype=np.float32),
    )


def load_valid_prediction_cache(path: Path, source_row: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as cached:
            cached_rows = cached["source_row"].astype(int)
            prediction = cached["prediction"].astype(int)
            probability = cached["probability"].astype(float)
    except (KeyError, OSError, ValueError):
        return None
    valid = (
        np.array_equal(cached_rows, np.asarray(source_row, dtype=int))
        and prediction.shape == (len(source_row),)
        and probability.shape == (len(source_row), len(ALL_LABELS))
        and np.all(prediction >= 0)
        and np.isfinite(probability).all()
        and np.allclose(probability.sum(axis=1), 1.0, atol=1e-5)
    )
    return (prediction, probability) if valid else None


def multiclass_brier(y_true: np.ndarray, probability: np.ndarray) -> float:
    one_hot = np.eye(len(ALL_LABELS), dtype=float)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 15
) -> float:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece if total else math.nan)


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "negative_log_loss": float(log_loss(y_true, probability, labels=ALL_LABELS)),
        "multiclass_brier": multiclass_brier(y_true, probability),
        "ece_15bin": expected_calibration_error(y_true, probability, bins=15),
        "mean_confidence": float(probability.max(axis=1).mean()),
    }


def reliability_rows(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 15,
) -> list[dict[str, object]]:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        rows.append(
            {
                "bin": index + 1,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "rows": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if np.any(mask) else math.nan,
                "accuracy": float(correct[mask].mean()) if np.any(mask) else math.nan,
            }
        )
    return rows


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        calibrated = apply_temperature(probability, temperature)
        return float(log_loss(y_true, calibrated, labels=ALL_LABELS))

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    return temperature, float(result.fun)


def choose_calibration_well(training: pd.DataFrame) -> str:
    counts = training.groupby("well_name", sort=True).size()
    class_counts = training.groupby("well_name", sort=True)["facies_id"].nunique()
    candidates = counts.loc[class_counts.eq(len(ALL_LABELS))]
    if candidates.empty:
        candidates = counts
    median = float(candidates.median())
    ordered = sorted(candidates.index, key=lambda name: (abs(int(candidates[name]) - median), name))
    return str(ordered[0])


def degrade_logs_by_stride(frame: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride < 2:
        return frame.copy()
    out = frame.copy().reset_index(drop=True)
    for positions in out.groupby(["well_name", "segment_id"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        if len(positions) < 2:
            continue
        depth = out.iloc[positions]["depth"].to_numpy(float)
        anchors = np.arange(0, len(positions), stride, dtype=int)
        if anchors[-1] != len(positions) - 1:
            anchors = np.append(anchors, len(positions) - 1)
        for log_name in RAW_FEATURES:
            values = out.iloc[positions][log_name].to_numpy(float)
            interpolated = np.interp(depth, depth[anchors], values[anchors])
            out.loc[positions, log_name] = interpolated
    degraded, _ = add_window_features(out)
    return degraded


def contiguous_gap_mask(
    data: pd.DataFrame,
    coverage: float,
    length_m: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(data), dtype=bool)
    run_rows = max(1, int(round(length_m / EXPECTED_SAMPLE_STEP_M)))
    for _, well_positions in data.groupby("well_name", sort=True).indices.items():
        well_positions = np.asarray(well_positions, dtype=int)
        target = int(math.ceil(coverage * len(well_positions)))
        segment_positions = []
        well_frame = data.iloc[well_positions]
        for local_positions in well_frame.groupby("segment_id", sort=False).indices.values():
            segment_positions.append(well_positions[np.asarray(local_positions, dtype=int)])
        attempts = 0
        while int(mask[well_positions].sum()) < target and attempts < 10000:
            segment = segment_positions[int(rng.integers(0, len(segment_positions)))]
            if len(segment) <= run_rows:
                start = 0
            else:
                start = int(rng.integers(0, len(segment) - run_rows + 1))
            mask[segment[start : start + run_rows]] = True
            attempts += 1
        if int(mask[well_positions].sum()) < target:
            raise RuntimeError("Unable to construct requested contiguous-gap coverage")
    return mask


def segment_cell_edges(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=float)
    if len(depth) == 1:
        half = EXPECTED_SAMPLE_STEP_M / 2.0
        return np.array([depth[0] - half, depth[0] + half], dtype=float)
    midpoint = 0.5 * (depth[:-1] + depth[1:])
    first = depth[0] - 0.5 * (depth[1] - depth[0])
    last = depth[-1] + 0.5 * (depth[-1] - depth[-2])
    return np.concatenate(([first], midpoint, [last]))


def sample_support_weights(data: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(data), dtype=float)
    for positions in data.groupby(["well_name", "segment_id"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        edges = segment_cell_edges(data.iloc[positions]["depth"].to_numpy(float))
        weights[positions] = np.diff(edges)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Invalid sample support weights")
    return weights


def run_intervals(data: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    rows = []
    for (well_name, segment_id), positions in data.groupby(
        ["well_name", "segment_id"], sort=False
    ).indices.items():
        positions = np.asarray(positions, dtype=int)
        depth = data.iloc[positions]["depth"].to_numpy(float)
        segment_labels = labels[positions]
        edges = segment_cell_edges(depth)
        starts = np.r_[0, np.flatnonzero(segment_labels[1:] != segment_labels[:-1]) + 1]
        stops = np.r_[starts[1:], len(segment_labels)]
        for run_number, (start, stop) in enumerate(zip(starts, stops, strict=True), start=1):
            global_positions = positions[start:stop]
            seam_values = data.iloc[global_positions]["coal_seam_unit"].astype(str)
            seam = seam_values.mode().iloc[0] if len(seam_values) else "OUTSIDE_TARGET"
            rows.append(
                {
                    "well_name": str(well_name),
                    "segment_id": int(segment_id),
                    "run_number": run_number,
                    "facies_id": int(segment_labels[start]),
                    "start_position": int(global_positions[0]),
                    "stop_position_exclusive": int(global_positions[-1] + 1),
                    "top_depth_m": float(edges[start]),
                    "base_depth_m": float(edges[stop]),
                    "thickness_m": float(edges[stop] - edges[start]),
                    "rows": int(stop - start),
                    "coal_seam_unit": str(seam),
                }
            )
    return pd.DataFrame(rows)


def match_boundary_offsets(
    observed: np.ndarray,
    predicted: np.ndarray,
    tolerance_m: float = 0.5,
) -> list[tuple[float, float, float]]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    candidates = [
        (abs(float(left - right)), left_index, right_index)
        for left_index, left in enumerate(observed)
        for right_index, right in enumerate(predicted)
        if abs(float(left - right)) <= tolerance_m
    ]
    used_observed: set[int] = set()
    used_predicted: set[int] = set()
    matches = []
    for distance, left_index, right_index in sorted(candidates):
        if left_index in used_observed or right_index in used_predicted:
            continue
        used_observed.add(left_index)
        used_predicted.add(right_index)
        matches.append(
            (
                float(observed[left_index]),
                float(predicted[right_index]),
                float(distance),
            )
        )
    return matches


def build_manifest(directory: Path, manifest_name: str = "package_manifest.csv") -> pd.DataFrame:
    rows = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == manifest_name:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / manifest_name, index=False, encoding="utf-8-sig")
    return frame


def source_hashes() -> dict[str, str]:
    return {
        "measurement_data": sha256_file(ROOT / "data" / "measurement_data.csv"),
        "well_inventory": sha256_file(ROOT / "data" / "well_inventory_route_a.csv"),
        "authorized_picks": sha256_file(
            ROOT
            / "outputs"
            / "cad_fault_constrained_3d_model_v2"
            / "author_authorized_top_inputs_v2.csv"
        ),
        "label_provenance_audit": sha256_file(LABEL_AUDIT),
    }
