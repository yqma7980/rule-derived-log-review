"""Generated controls and paired-state/budget demonstration; no fitting or data I/O."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows/kgs/code"))
from external_core import features, interpolate, partition, rule
from external_review_trial import physical_blocks
from review_priority import evaluate_blocks


def run(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Keep synthetic outputs inside this local candidate")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory; reference outputs are preserved")
    output.mkdir(parents=True, exist_ok=True)
    z = np.arange(200) * .152
    gr = 55 + 33 * np.sin(z * 2.5)
    den = 2.32 + .16 * np.cos(z * .9)
    fine = rule(gr, den, z, .152)[0]
    cg, cd = (interpolate(a, z, 5, 0) for a in (gr, den))
    coarse = rule(cg, cd, z, .152)[0]
    comparator = coarse.copy()
    comparator[::17] = (comparator[::17] + 1) % 6
    parts = partition(fine, coarse, comparator)
    risk = np.zeros(len(z))
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a or b:
                risk += rule(cg + a, cd + b * .01, z, .152)[0] != coarse
    risk /= 8
    frame = pd.DataFrame({"depth": z, "segment": 0})
    starts, sizes = physical_blocks(frame)
    scores = np.array([risk[start:start + size].mean() for start, size in zip(starts, sizes)])
    target = coarse != fine
    budgets = {str(b): evaluate_blocks(scores, starts, sizes, target, b, well="SYNTHETIC")
               for b in (0., .10, .13, 1.)}
    zero = rule(interpolate(gr, z, 1, 0), interpolate(den, z, 1, 0), z, .152)[0]
    np.testing.assert_array_equal(zero, fine)
    controls = []
    for g, d, expected in [(50., 2.4, 2), (50., 1.5, 0), (95., 2.4, 5)]:
        actual = rule(np.full(80, g), np.full(80, d), np.arange(80) * .05, .05)[0]
        assert np.all(actual == expected)
        controls.append({"GR_API": g, "DEN_g_cm3": d, "expected_state": expected, "rows": 80})
    counts = dict(zip("ABCDE", parts.sum(axis=1).tolist()))
    rule_changed = int(target.sum())
    comparator_differs = int((comparator != fine).sum())
    assert comparator_differs - rule_changed == counts["B"] - counts["D"]
    x, spans = features(cg, cd, z, .152)
    summary = {
        "kind": "GENERATED SYNTHETIC DEMONSTRATION; NOT MANUSCRIPT EVIDENCE",
        "rows": len(z), "native_step_m": .152, "stride": 5, "phase": 0,
        "comparator": "processed rule with every 17th label rotated; not a trained model",
        "ranking": "processed-curve perturbation fraction only; no native target in ranking",
        "paired_counts": counts, "rule_changed": rule_changed,
        "comparator_vs_native": comparator_differs,
        "identity_difference": comparator_differs - rule_changed,
        "zero_stride_changed_rows": int((zero != fine).sum()), "constant_controls": controls,
        "feature_shape": list(x.shape), "actual_feature_spans_m": spans,
        "budget_results": budgets, "models_fitted": 0, "network_access_required": False,
    }
    pd.DataFrame({"synthetic_row": np.arange(len(z)), "relative_depth_m": z,
                  "synthetic_GR_API": gr, "synthetic_DEN_g_cm3": den,
                  "processed_GR_API": cg, "processed_DEN_g_cm3": cd,
                  "native_rule": fine, "processed_rule": coarse,
                  "constructed_comparator": comparator, "processed_only_risk": risk}).to_csv(
                      output / "synthetic_points.csv", index=False)
    pd.DataFrame({"start": starts, "size": sizes, "score": scores}).to_csv(
        output / "synthetic_blocks.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/output_local")
    run(parser.parse_args().output)
