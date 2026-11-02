"""Unmodified metric functions extracted from the legacy geology module.

These legacy boundary metrics are not the later maximum-cardinality analysis.
The public main-result runner does not recreate that separate historical branch.
"""
import numpy as np
import pandas as pd
from common import well_metrics


def minority_recall(
    y_true: np.ndarray, y_pred: np.ndarray, minority_labels: np.ndarray
) -> float:
    recalls = []
    for label in minority_labels:
        mask = y_true == label
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def boundary_positions(depth: np.ndarray, labels: np.ndarray) -> np.ndarray:
    changes = labels[1:] != labels[:-1]
    return 0.5 * (depth[:-1][changes] + depth[1:][changes])


def match_boundaries(
    observed: np.ndarray, predicted: np.ndarray, tolerance_m: float
) -> int:
    if len(observed) == 0 or len(predicted) == 0:
        return 0
    used = np.zeros(len(predicted), dtype=bool)
    matches = 0
    for value in observed:
        distance = np.abs(predicted - value)
        distance[used] = np.inf
        index = int(np.argmin(distance))
        if distance[index] <= tolerance_m:
            used[index] = True
            matches += 1
    return matches


def vertical_consistency_metrics(
    well_frame: pd.DataFrame,
    prediction: np.ndarray,
    tolerance_m: float = 0.5,
) -> dict[str, float]:
    observed_count = 0
    predicted_count = 0
    matched_count = 0
    for _, positions in well_frame.groupby("segment_id", sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        depth = well_frame.iloc[positions]["depth"].to_numpy(float)
        observed = boundary_positions(
            depth, well_frame.iloc[positions]["facies_id"].to_numpy(int)
        )
        predicted = boundary_positions(depth, prediction[positions])
        observed_count += len(observed)
        predicted_count += len(predicted)
        matched_count += match_boundaries(observed, predicted, tolerance_m)
    precision = matched_count / predicted_count if predicted_count else 0.0
    recall = matched_count / observed_count if observed_count else 0.0
    boundary_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    transition_ratio = (
        predicted_count / observed_count if observed_count else float("nan")
    )
    return {
        "observed_boundary_count": int(observed_count),
        "predicted_boundary_count": int(predicted_count),
        "matched_boundary_count_0p5m": int(matched_count),
        "boundary_precision_0p5m": float(precision),
        "boundary_recall_0p5m": float(recall),
        "boundary_f1_0p5m": float(boundary_f1),
        "transition_count_ratio": float(transition_ratio),
    }


def evaluate_well(
    well_frame: pd.DataFrame,
    prediction: np.ndarray,
    minority_labels: np.ndarray,
) -> dict[str, float]:
    y_true = well_frame["facies_id"].to_numpy(int)
    metrics = well_metrics(y_true, prediction)
    metrics["minority_mean_recall"] = minority_recall(
        y_true, prediction, minority_labels
    )
    metrics.update(vertical_consistency_metrics(well_frame, prediction))
    return metrics
