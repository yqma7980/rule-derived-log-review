from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


REQUIRED_COLUMNS = ["well_name", "depth", "RT", "GR", "DEN", "Subfacies"]
RAW_FEATURES = ["RT", "GR", "DEN"]
EXPECTED_SAMPLE_STEP_M = 0.05
GAP_THRESHOLD_M = 0.075

# This order follows the geological table in the manuscript, not the
# alphabetical order produced by sklearn.preprocessing.LabelEncoder.
FACIES = [
    {"id": 0, "code": "SW", "cn": "沼泽", "en": "Swamp"},
    {"id": 1, "code": "DCH", "cn": "分流河道", "en": "Distributary channel"},
    {"id": 2, "code": "MB", "cn": "河口坝", "en": "Mouth bar"},
    {"id": 3, "code": "SS", "cn": "席状砂", "en": "Sheet sand"},
    {"id": 4, "code": "SLM", "cn": "浅湖泥", "en": "Shallow lacustrine mud"},
    {"id": 5, "code": "DLM", "cn": "深湖泥", "en": "Deep lacustrine mud"},
]
CN_TO_ID = {item["cn"]: item["id"] for item in FACIES}
ID_TO_CN = {item["id"]: item["cn"] for item in FACIES}
ID_TO_CODE = {item["id"]: item["code"] for item in FACIES}
ID_TO_EN = {item["id"]: item["en"] for item in FACIES}
ALL_LABELS = np.arange(len(FACIES), dtype=int)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    unexpected_labels = sorted(set(df["Subfacies"].dropna()) - set(CN_TO_ID))
    if unexpected_labels:
        raise ValueError(f"Unexpected Subfacies labels: {unexpected_labels}")


def qc_reasons(df: pd.DataFrame) -> pd.Series:
    reasons = pd.Series("", index=df.index, dtype="object")
    rules = [
        (df["RT"] <= 0, "RT<=0"),
        (df["GR"] <= 0, "GR<=0"),
        (df["DEN"] <= 0, "DEN<=0"),
        (df["DEN"] > 5, "DEN>5"),
    ]
    for mask, reason in rules:
        reasons.loc[mask & reasons.eq("")] = reason
        reasons.loc[mask & reasons.ne("")] = reasons.loc[mask & reasons.ne("")].where(
            reasons.loc[mask & reasons.ne("")].str.contains(reason, regex=False),
            reasons.loc[mask & reasons.ne("")] + ";" + reason,
        )
    return reasons


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["well_name", "depth"], kind="mergesort").reset_index(drop=True)
    well_change = out["well_name"].ne(out["well_name"].shift())
    depth_step = out.groupby("well_name", sort=False)["depth"].diff()
    segment_break = well_change | depth_step.isna() | (depth_step <= 0) | (depth_step > GAP_THRESHOLD_M)
    out["segment_id"] = segment_break.groupby(out["well_name"]).cumsum().astype(int)
    return out


def load_data(path: Path, apply_qc: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    validate_columns(df)
    df = df.copy()
    df["source_row"] = np.arange(2, len(df) + 2, dtype=int)
    reasons = qc_reasons(df)
    rejected = df.loc[reasons.ne("")].copy()
    rejected["qc_reason"] = reasons.loc[reasons.ne("")]
    if apply_qc:
        df = df.loc[reasons.eq("")].copy()
    df["facies_id"] = df["Subfacies"].map(CN_TO_ID).astype(int)
    return add_segments(df), rejected


def _window_sample_count(width_m: float) -> int:
    count = max(3, int(round(width_m / EXPECTED_SAMPLE_STEP_M)))
    if count % 2 == 0:
        count += 1
    return count


def add_window_features(
    df: pd.DataFrame,
    windows_m: Iterable[float] = (0.25, 0.50, 1.00),
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = df.copy()
    feature_groups: dict[str, list[str]] = {"raw": list(RAW_FEATURES)}
    segment_positions = list(out.groupby(["well_name", "segment_id"], sort=False).indices.values())

    accumulated = list(RAW_FEATURES)
    for width_m in windows_m:
        tag = f"w{int(round(width_m * 100)):03d}"
        sample_count = _window_sample_count(width_m)
        half = sample_count // 2
        created: list[str] = []
        for log_name in RAW_FEATURES:
            names = {
                "mean": f"{log_name}_{tag}_mean",
                "std": f"{log_name}_{tag}_std",
                "range": f"{log_name}_{tag}_range",
                "slope": f"{log_name}_{tag}_slope",
            }
            arrays = {name: np.empty(len(out), dtype=float) for name in names.values()}
            for positions in segment_positions:
                positions = np.asarray(positions, dtype=int)
                values = out.iloc[positions][log_name].to_numpy(dtype=float)
                depths = out.iloc[positions]["depth"].to_numpy(dtype=float)
                series = pd.Series(values)
                rolling = series.rolling(sample_count, center=True, min_periods=1)
                arrays[names["mean"]][positions] = rolling.mean().to_numpy()
                arrays[names["std"]][positions] = rolling.std(ddof=0).fillna(0.0).to_numpy()
                arrays[names["range"]][positions] = (rolling.max() - rolling.min()).to_numpy()

                if len(values) == 1:
                    slope = np.zeros(1, dtype=float)
                else:
                    index = np.arange(len(values))
                    left = np.maximum(index - half, 0)
                    right = np.minimum(index + half, len(values) - 1)
                    denominator = depths[right] - depths[left]
                    slope = np.divide(
                        values[right] - values[left],
                        denominator,
                        out=np.zeros_like(values, dtype=float),
                        where=np.abs(denominator) > 1e-12,
                    )
                arrays[names["slope"]][positions] = slope

            for name, values in arrays.items():
                out[name] = values
                created.append(name)

        accumulated.extend(created)
        feature_groups[f"raw_through_{tag}"] = list(accumulated)

    if out[accumulated].isna().any().any():
        raise ValueError("Window feature generation produced missing values")
    return out, feature_groups


def align_probabilities(model_classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(ALL_LABELS)), dtype=float)
    for source_column, label in enumerate(model_classes.astype(int)):
        aligned[:, label] = probabilities[:, source_column]
    return aligned


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ALL_LABELS, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if probabilities is not None:
        binary = label_binarize(y_true, classes=ALL_LABELS)
        metrics["macro_roc_auc_ovr"] = float(
            roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")
        )
        metrics["macro_average_precision"] = float(
            average_precision_score(binary, probabilities, average="macro")
        )
    return metrics


def well_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    present = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1_present": float(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=ALL_LABELS,
        zero_division=0,
    )
    rows = []
    for label, p_value, r_value, f_value, n_value in zip(
        ALL_LABELS, precision, recall, f1, support, strict=True
    ):
        rows.append(
            {
                "facies_id": int(label),
                "facies_code": ID_TO_CODE[int(label)],
                "facies_cn": ID_TO_CN[int(label)],
                "facies_en": ID_TO_EN[int(label)],
                "precision": float(p_value),
                "recall": float(r_value),
                "f1": float(f_value),
                "support": int(n_value),
            }
        )
    return pd.DataFrame(rows)


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    return pd.DataFrame(
        matrix,
        index=[f"true_{ID_TO_CODE[int(label)]}" for label in ALL_LABELS],
        columns=[f"pred_{ID_TO_CODE[int(label)]}" for label in ALL_LABELS],
    )


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42, repeats: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return float(lower), float(upper)
