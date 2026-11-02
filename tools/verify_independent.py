"""Stdlib-only output readback; no imports of production scientific functions."""
import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from independent_oracles import interpolate_scalar, paired_counts, review_scalar, rule_scalar


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b, tolerance=1e-10):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"Mismatch: {a} versus {b}")


def verify_synthetic(output):
    points = rows(output / "synthetic_points.csv")
    blocks = rows(output / "synthetic_blocks.csv")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(points) == summary["rows"] == 200
    z = [float(p["relative_depth_m"]) for p in points]
    gr = [float(p["synthetic_GR_API"]) for p in points]
    den = [float(p["synthetic_DEN_g_cm3"]) for p in points]
    for i in range(200):
        close(z[i], i * .152)
        close(gr[i], 55 + 33 * math.sin(z[i] * 2.5))
        close(den[i], 2.32 + .16 * math.cos(z[i] * .9))
    cg, cd = (interpolate_scalar(a, z, 5, 0) for a in (gr, den))
    for p, g, d in zip(points, cg, cd):
        close(p["processed_GR_API"], g)
        close(p["processed_DEN_g_cm3"], d)
    fine = rule_scalar(gr, den, z, .152)
    coarse = rule_scalar(cg, cd, z, .152)
    prediction = [(v + 1) % 6 if i % 17 == 0 else v for i, v in enumerate(coarse)]
    assert fine == [int(p["native_rule"]) for p in points]
    assert coarse == [int(p["processed_rule"]) for p in points]
    assert prediction == [int(p["constructed_comparator"]) for p in points]
    assert paired_counts(fine, coarse, prediction) == summary["paired_counts"]
    target = [f != c for f, c in zip(fine, coarse)]
    assert sum(target) == summary["rule_changed"]
    assert sum(p != f for p, f in zip(prediction, fine)) == summary["comparator_vs_native"]
    assert summary["comparator_vs_native"] - summary["rule_changed"] == summary["identity_difference"]
    assert summary["identity_difference"] == summary["paired_counts"]["B"] - summary["paired_counts"]["D"]
    risk = [0.] * len(z)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a or b:
                perturbed = rule_scalar([v + a for v in cg], [v + b * .01 for v in cd], z, .152)
                risk = [r + (c != p) / 8 for r, c, p in zip(risk, coarse, perturbed)]
    for p, r in zip(points, risk):
        close(p["processed_only_risk"], r)
    groups = defaultdict(list)
    for i, depth in enumerate(z):
        groups[math.floor((depth - z[0] + 1e-8) / 1.)].append(i)
    expected = list(groups.values())
    starts, sizes, scores = [], [], []
    assert len(blocks) == len(expected)
    for row, indices in zip(blocks, expected):
        start, size = int(row["start"]), int(row["size"])
        assert start == indices[0] and size == len(indices)
        score = statistics.mean(risk[i] for i in indices)
        close(score, row["score"])
        starts.append(start)
        sizes.append(size)
        scores.append(score)
    for budget, reported in summary["budget_results"].items():
        assert reported == review_scalar(scores, starts, sizes, target, float(budget))
    assert summary["zero_stride_changed_rows"] == 0
    assert rule_scalar(interpolate_scalar(gr, z, 1, 0), interpolate_scalar(den, z, 1, 0), z, .152) == fine
    for control in summary["constant_controls"]:
        n = control["rows"]
        assert rule_scalar([control["GR_API"]] * n, [control["DEN_g_cm3"]] * n,
                           [i * .05 for i in range(n)], .05) == [control["expected_state"]] * n
    return {"synthetic_rows": len(points), "independent_scalar_rule_evaluations": 11,
            "budget_cases": len(summary["budget_results"]), "constant_controls": 3,
            "generated_measurements_verified": True, "paired_state_identity": "PASS"}


def verify_force():
    base = ROOT / "workflows/force/results"
    counts = rows(base / "force_paired_counts.csv")
    expected = rows(base / "force_paired_summary.csv")
    metrics = list("ABCDE") + ["rule_changed", "RF_vs_fine", "RF_minus_rule"]
    phases = defaultdict(list)
    unique = set()
    for row in counts:
        key = (int(row["stride"]), row["well"])
        phase_key = (*key, int(row["phase"]))
        assert phase_key not in unique
        unique.add(phase_key)
        n = int(row["n"])
        a, b, c, d, e = (int(row[k]) for k in "ABCDE")
        assert a + b + c + d + e == n
        assert c + d + e == int(row["rule_changed"])
        assert b + c + e == int(row["RF_vs_fine"])
        assert b - d == int(row["RF_minus_rule"])
        phases[key].append({m: float(row[m]) / n for m in metrics})
    perwell = defaultdict(list)
    for (stride, well), values in phases.items():
        assert len(values) == stride
        perwell[stride].append({m: statistics.mean(v[m] for v in values) for m in metrics})
    for row in expected:
        vals = perwell[int(row["stride"])]
        assert int(row["n_wells"]) == len(vals) == 98
        close(statistics.mean(v[row["metric"]] for v in vals), row["mean"], 1e-12)
    assert len(counts) == 1176
    return {"well_phase_rows": len(counts), "summary_mean_cells": len(expected),
            "aggregation": "phase means within well, then equal-well mean",
            "raw_inputs_replayed": False, "RF_predictions_recomputed": False,
            "bootstrap_intervals_recomputed": False}


def verify_kgs():
    base = ROOT / "reference_results/kgs"
    paired = rows(base / "paired_differences.csv")
    original = {int(r["stride"]): r for r in rows(base / "summary_original.csv")}
    independent = {int(r["stride"]): r for r in rows(base / "summary_independent.csv")}
    grouped = defaultdict(list)
    methods = ["rule_stability", "category_support", "change_prediction"]
    for row in paired:
        grouped[int(row["stride"])].append(row)
    checks = []
    for stride, values in sorted(grouped.items()):
        assert len(values) == len({v["well"] for v in values}) == 49
        defined = [v for v in values if v["change_minus_rule_pp"]]
        assert len(defined) == 48
        assert int(independent[stride]["tested_wells"]) == 49
        assert int(independent[stride]["assessable_wells"]) == 48
        for row in values:
            if not row["change_minus_rule_pp"]:
                assert all(not row[m] for m in methods)
        for method in methods:
            mean = 100 * statistics.mean(float(v[method]) for v in defined)
            close(mean, original[stride][method + "_capture_pct"])
            close(mean, independent[stride][method + "_capture_pct"])
        for baseline, short in (("rule_stability", "rule"), ("category_support", "category")):
            name = f"change_minus_{short}_pp"
            differences = [float(v[name]) for v in defined]
            for v, delta in zip(defined, differences):
                close(delta, 100 * (float(v["change_prediction"]) - float(v[baseline])))
            close(statistics.mean(differences), original[stride][name])
            close(statistics.mean(differences), independent[stride][name])
            signs = {"positive": sum(x > 0 for x in differences),
                     "negative": sum(x < 0 for x in differences), "tied": sum(x == 0 for x in differences)}
            for sign, count in signs.items():
                assert count == int(independent[stride][name + "_" + sign])
                if sign != "tied":
                    assert count == int(original[stride][name + "_" + sign + "_wells"])
        checks.append({"stride": stride, "tested_wells": 49, "assessable_wells": 48})
    assert set(grouped) == {2, 4, 5} and len(paired) == 147
    return {"well_stride_rows": len(paired), "denominators": checks,
            "mean_and_sign_readback": "PASS", "bootstrap_intervals_recomputed": False,
            "raw_input_replay_or_refit": False, "point_predictions_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-output", type=Path, default=ROOT / "examples/output_local")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = {"status": "PASS", "implementation": "Python standard library, separate scalar/list oracles",
              "synthetic": verify_synthetic(args.synthetic_output), "force": verify_force(), "kgs": verify_kgs(),
              "scope": "Arithmetic and generated-control readback only; not full scientific reproduction or public approval"}
    if args.report:
        report = args.report.resolve()
        if not report.is_relative_to(ROOT):
            raise ValueError("Report must remain inside candidate")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
