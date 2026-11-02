"""Deterministic operational interpretation rule used in the Liujia study.

This module reproduces the historical six-class labels. It does not provide an
independent sedimentological interpretation. Each call accepts one complete
well or one quality-control-contiguous segment sampled on the 0.05 m grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LITHOLOGY_THRESHOLDS = {
    "coal": {"DEN_max": 1.75, "GR_max": 70.0},
    "shale": {"GR_min": 80.0, "DEN_min": 2.30},
    "siltstone": {"GR_min": 60.0, "GR_max": 80.0},
    "fine_sand": {"GR_min": 40.0, "GR_max": 60.0},
    "medium_sand": {"GR_min": 20.0, "GR_max": 40.0},
    "coarse_sand": {"GR_max": 20.0},
}

SUBFACIES_CONFIG = {
    "thick_threshold_m": 3.0,
    "thin_threshold_m": 2.0,
    "window_size_rows": 20,
    "deep_lake_GR_API": 90.0,
    "depth_step_m": 0.05,
}

SUBFACIES_CODE = {
    1: "SW",
    2: "DCH",
    3: "MB",
    4: "SS",
    5: "SLM",
    6: "DLM",
}

EXPECTED_DEPTH_STEP_M = 0.05
DEPTH_GRID_TOLERANCE_M = 0.0051


def validate_input(
    frame: pd.DataFrame,
    enforce_nominal_grid: bool = True,
) -> None:
    """Enforce the exact input contract used by the operational rule."""
    required = {"depth", "GR", "DEN"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Input must contain at least one row")

    values = frame[["depth", "GR", "DEN"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("depth, GR, and DEN must contain only finite values")

    depth = values[:, 0]
    if len(depth) > 1:
        increments = np.diff(depth)
        if np.any(increments <= 0):
            raise ValueError("depth must be strictly increasing within one well or segment")
        expected_depth = depth[0] + np.arange(len(depth)) * EXPECTED_DEPTH_STEP_M
        if enforce_nominal_grid and not np.allclose(
            depth,
            expected_depth,
            rtol=0.0,
            atol=DEPTH_GRID_TOLERANCE_M,
        ):
            raise ValueError(
                "depth must follow the nominal 0.05 m grid within 0.0051 m; "
                "split the input at gaps or explicitly use the documented "
                "historical row-index replay mode"
            )


def classify_lithology(row: pd.Series) -> int:
    """Return historical lithology code 0-5 from GR and DEN."""
    gr = float(row["GR"])
    den = float(row["DEN"])
    if not np.isfinite(gr) or not np.isfinite(den):
        raise ValueError("GR and DEN must be finite")
    if den < LITHOLOGY_THRESHOLDS["coal"]["DEN_max"] and gr < LITHOLOGY_THRESHOLDS["coal"]["GR_max"]:
        return 0
    if gr >= LITHOLOGY_THRESHOLDS["shale"]["GR_min"] and den >= LITHOLOGY_THRESHOLDS["shale"]["DEN_min"]:
        return 1
    if LITHOLOGY_THRESHOLDS["siltstone"]["GR_min"] <= gr < LITHOLOGY_THRESHOLDS["siltstone"]["GR_max"]:
        return 2
    if LITHOLOGY_THRESHOLDS["fine_sand"]["GR_min"] <= gr < LITHOLOGY_THRESHOLDS["fine_sand"]["GR_max"]:
        return 3
    if LITHOLOGY_THRESHOLDS["medium_sand"]["GR_min"] <= gr < LITHOLOGY_THRESHOLDS["medium_sand"]["GR_max"]:
        return 4
    if gr < LITHOLOGY_THRESHOLDS["coarse_sand"]["GR_max"]:
        return 5
    return 1


def add_layer_thickness(frame: pd.DataFrame) -> pd.DataFrame:
    """Add run identifiers and sample-count thicknesses at the 0.05 m grid."""
    out = frame.copy()
    out["layer_id"] = out["Lithology"].ne(out["Lithology"].shift()).cumsum()
    thickness = out.groupby("layer_id", sort=False).size() * SUBFACIES_CONFIG["depth_step_m"]
    out["layer_thickness"] = out["layer_id"].map(thickness)
    return out


def add_cycle_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain the historical coal-entry cycle index (not used by later rules)."""
    out = frame.copy()
    out["is_coal"] = out["Lithology"].eq(0).astype(int)
    out["cycle_id"] = out["is_coal"].diff().gt(0).cumsum().ffill().astype(int)
    return out


def classify_subfacies(frame: pd.DataFrame) -> np.ndarray:
    """Apply the historical 20-row local-context rule.

    For row i, the slice is [i-10, i+10), clipped at segment limits. Because
    rows increase with depth, indices less than i are shallower and indices
    greater than i are deeper. The original source named these two subsets
    ``below`` and ``above``, respectively; the implementation below uses the
    physical names while preserving the exact index tests.
    """
    labels: list[int] = []
    n_rows = len(frame)
    half_window = SUBFACIES_CONFIG["window_size_rows"] // 2
    lithology_values = frame["Lithology"].to_numpy(int)
    gr_values = frame["GR"].to_numpy(float)
    thickness_values = frame["layer_thickness"].to_numpy(float)
    for i in range(n_rows):
        start = max(0, i - half_window)
        stop = min(n_rows, i + half_window)
        window_lithology = lithology_values[start:stop]
        lithology = int(lithology_values[i])
        gr = float(gr_values[i])
        thickness = float(thickness_values[i])

        if lithology == 0:
            labels.append(1)
            continue
        if lithology == 1:
            labels.append(6 if gr > SUBFACIES_CONFIG["deep_lake_GR_API"] else 5)
            continue
        if lithology in (4, 5) and thickness > SUBFACIES_CONFIG["thick_threshold_m"]:
            labels.append(2)
            continue
        if lithology == 3:
            deeper = lithology_values[i + 1 : min(stop, i + 6)]
            shallower = lithology_values[max(start, i - 5) : i]
            if np.isin(deeper, [1, 2]).any() and np.isin(shallower, [4, 5]).any():
                labels.append(3)
                continue
        if lithology in (2, 3) and thickness < SUBFACIES_CONFIG["thin_threshold_m"]:
            unique = np.unique(window_lithology)
            if len(unique) >= 3 and np.any(window_lithology == 2) and np.any(window_lithology == 3):
                labels.append(4)
                continue

        labels.append({2: 4, 3: 3, 4: 2, 5: 2}.get(lithology, 5))
    return np.asarray(labels, dtype=np.int16)


def apply_operational_rule(
    frame: pd.DataFrame,
    *,
    enforce_nominal_grid: bool = True,
) -> pd.DataFrame:
    """Return operational labels after validating one well or segment.

    ``enforce_nominal_grid=False`` exists only to reproduce the historical
    complete-well, row-indexed calculation documented in the accompanying
    README. New applications should retain the default strict grid check.
    """
    validate_input(frame, enforce_nominal_grid=enforce_nominal_grid)
    out = frame.reset_index(drop=True).copy()
    gr = out["GR"].to_numpy(float)
    den = out["DEN"].to_numpy(float)
    lithology = np.ones(len(out), dtype=np.int16)
    unresolved = np.ones(len(out), dtype=bool)
    coal = unresolved & (den < 1.75) & (gr < 70.0)
    lithology[coal] = 0
    unresolved &= ~coal
    shale = unresolved & (gr >= 80.0) & (den >= 2.30)
    lithology[shale] = 1
    unresolved &= ~shale
    siltstone = unresolved & (gr >= 60.0) & (gr < 80.0)
    lithology[siltstone] = 2
    unresolved &= ~siltstone
    fine_sand = unresolved & (gr >= 40.0) & (gr < 60.0)
    lithology[fine_sand] = 3
    unresolved &= ~fine_sand
    medium_sand = unresolved & (gr >= 20.0) & (gr < 40.0)
    lithology[medium_sand] = 4
    unresolved &= ~medium_sand
    coarse_sand = unresolved & (gr < 20.0)
    lithology[coarse_sand] = 5
    out["Lithology"] = lithology
    out = add_layer_thickness(out)
    out = add_cycle_id(out)
    out["Subfacies"] = classify_subfacies(out)
    return out
