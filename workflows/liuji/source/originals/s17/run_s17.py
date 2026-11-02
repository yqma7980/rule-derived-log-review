"""Exploratory paired-discrepancy and rule-intervention analysis; no model fitting."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "code" / "inherited"))
import audit_core as core
import legacy_common as common

SCHEMES = ("leave_one_well_out", "leave_one_spatial_block_out")
STATES = ("A_all_same", "B_rule_stable_RF_differs", "C_RF_follows_changed_rule",
          "D_RF_matches_fine_rule", "E_three_different")
SCENARIOS = [(1, 0)] + [(s, p) for s in (2, 4, 5) for p in range(s)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition(fine, coarse, prediction):
    equal = fine == coarse
    masks = (equal & (prediction == fine), equal & (prediction != fine),
             ~equal & (prediction == coarse), ~equal & (prediction == fine),
             ~equal & (prediction != fine) & (prediction != coarse))
    assert np.all(np.sum(masks, axis=0) == 1)
    return masks


def diagnose(frame):
    result = np.empty(len(frame), np.int16)
    base = np.empty_like(result)
    run_length = np.empty(len(frame), float)
    for idx in frame.groupby(["well_name", "segment_id"], sort=False).indices.values():
        part = frame.iloc[idx]
        lab, info = core.simplified_rule(part.GR, part.DEN, True)
        result[idx] = lab
        base[idx] = np.where(info["fine_context_changes_default"], 2, lab)
        starts = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1]
        lens = np.diff(np.r_[starts, len(lab)])
        run_length[idx] = np.repeat(lens, lens) * .05
    return result, base, run_length


def reconstruct(frame, stride, phase, channels=("GR", "DEN")):
    out = frame.copy()
    if stride == 1:
        return out
    for idx in frame.groupby(["well_name", "segment_id"], sort=False).indices.values():
        part = frame.iloc[idx]
        anchors = np.unique(np.r_[0, np.arange(phase, len(idx), stride), len(idx)-1])
        for channel in channels:
            out.loc[idx, channel] = np.interp(part.depth, part.depth.iloc[anchors],
                                             part[channel].iloc[anchors])
    return out


def rates_summary(frame, keys, columns):
    averaged = frame.groupby(keys + ["well"], as_index=False)[columns].mean()
    rows = []
    for key, group in averaged.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        for col in columns:
            values = group[col].dropna()
            rows.append(dict(zip(keys, key)) | {"metric": col, "n_wells": len(values),
                        "mean": values.mean(), "min": values.min(), "max": values.max()})
    return averaged, pd.DataFrame(rows)


def unit_tests():
    triples = np.array([(f, c, p) for f in range(6) for c in range(6) for p in range(6)])
    masks = partition(*triples.T)
    for i, (f, c, p) in enumerate(triples):
        expected = 0 if f == c == p else 1 if f == c else 2 if p == c else 3 if p == f else 4
        assert masks[expected][i]
    f, c, p = triples.T
    np.testing.assert_array_equal((p != f).astype(int) - (c != f).astype(int),
                                   masks[1].astype(int) - masks[3].astype(int))
    gr = np.full(80, 50.)
    den = np.full(80, 2.4)
    assert np.all(core.simplified_rule(gr, den) == 2)
    assert np.all(core.simplified_rule(gr, np.full(80, 1.5)) == 0)
    assert np.all(core.simplified_rule(np.full(80, 95.), den) == 5)
    return {"enumerated_label_triples": len(triples), "partition_identity": "PASS",
            "three_constant_rule_examples": "PASS"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(unit_tests(), indent=2))
        return
    if args.data_root is None:
        parser.error("--data-root is required for field-data analysis")
    root = args.data_root
    s12 = root / "GAS投稿" / "stage12_substantive_revision_20260907"
    source = root / "IJCG投稿" / "data" / "measurement_data.csv"
    blockfile = root / "IJCG投稿" / "outputs" / "geology_aware_subfacies_validation_v1" / "well_spatial_blocks_v1.csv"
    results = HERE / "results"
    tests = unit_tests()
    raw, _ = common.load_data(source, False)
    data, rejected = common.load_data(source, True)
    blocks = pd.read_csv(blockfile, usecols=["well_id", "spatial_block"]).rename(columns={"well_id": "well_name"})
    data = data.merge(blocks, on="well_name", how="left", validate="many_to_one", sort=False)
    assert len(raw) == 101639 and len(data) == 101615 and len(rejected) == 24
    assert data.spatial_block.notna().all() and data.source_row.is_unique
    fine, finebase, runlen = diagnose(data)
    saved_delta = int(np.count_nonzero(fine != data.facies_id))
    assert saved_delta == 15
    wellgroups = data.groupby("well_name", sort=False).indices
    margin = np.minimum(np.abs(data.GR.to_numpy()[:, None] - [20, 40, 60, 70, 80, 90]).min(axis=1),
                        np.abs(data.DEN.to_numpy()[:, None] - [1.75, 2.30]).min(axis=1) / .01)
    marginbin = pd.cut(margin, [-np.inf, 1, 3, np.inf], labels=["<=1", "(1,3]", ">3"])
    pairs, interventions, strata, inputs = [], [], [], []
    inputs.append({"role": "compiled_measurements", "filename": source.name, "size": source.stat().st_size, "sha256": digest(source)})
    inputs.append({"role": "spatial_blocks", "filename": blockfile.name, "size": blockfile.stat().st_size, "sha256": digest(blockfile)})
    for stride, phase in SCENARIOS:
        coarse_data = reconstruct(data, stride, phase)
        coarse, coarsebase, _ = diagnose(coarse_data)
        changed = coarse != fine
        if stride == 1:
            assert not changed.any()
        for scheme in SCHEMES:
            path = s12 / "private" / f"triage_{scheme}_s{stride}_p{phase}.npz"
            with np.load(path) as cache:
                np.testing.assert_array_equal(data.source_row, cache["source_row"])
                np.testing.assert_array_equal(coarse, cache["rule_class"])
                np.testing.assert_array_equal(changed.astype(int), cache["target"])
                probs = cache["probability"]
                assert probs.shape == (len(data), 6) and np.isfinite(probs).all()
                assert (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1, atol=1e-6)
                prediction = probs.argmax(axis=1)
            inputs.append({"role": "held_out_RF_cache", "filename": path.name,
                           "size": path.stat().st_size, "sha256": digest(path)})
            masks = partition(fine, coarse, prediction)
            for well, idx in wellgroups.items():
                counts = {name: int(mask[idx].sum()) for name, mask in zip(STATES, masks)}
                n = len(idx)
                rulecount = counts[STATES[2]] + counts[STATES[3]] + counts[STATES[4]]
                rfcount = counts[STATES[1]] + counts[STATES[2]] + counts[STATES[4]]
                assert rfcount - rulecount == counts[STATES[1]] - counts[STATES[3]]
                assert sum(counts.values()) == n
                pairs.append({"scheme": scheme, "stride": stride, "spacing_m": .05*stride,
                              "phase": phase, "well": well, "block": str(data.iloc[idx[0]].spatial_block),
                              "n": n, **counts, "rule_changed": rulecount, "RF_vs_fine": rfcount,
                              "RF_minus_rule": rfcount-rulecount})
        supportbin = pd.cut(runlen / (.05*stride), [-np.inf, 1, 2, 4, np.inf],
                            labels=["<=1", "(1,2]", "(2,4]", ">4"])
        for channel, channels in [("GR_only", ("GR",)), ("DEN_only", ("DEN",)), ("GR_and_DEN", ("GR", "DEN"))]:
            if channel == "GR_and_DEN":
                c, b = coarse, coarsebase
            else:
                c, b, _ = diagnose(reconstruct(data, stride, phase, channels))
            flip, baseflip = c != fine, b != finebase
            for well, idx in wellgroups.items():
                interventions.append({"stride": stride, "spacing_m": .05*stride, "phase": phase,
                    "well": well, "channel": channel, "n": len(idx), "changed": int(flip[idx].sum()),
                    "base_stable_context_changed": int((flip & ~baseflip)[idx].sum()),
                    "base_changed_full_changed": int((flip & baseflip)[idx].sum()),
                    "base_changed_full_stable": int((~flip & baseflip)[idx].sum())})
            if channel == "GR_and_DEN":
                for descriptor, bins in [("literal_threshold_distance", marginbin), ("fine_run_support_ratio", supportbin)]:
                    for level in bins.categories:
                        selector = bins == level
                        for well, idx in wellgroups.items():
                            selected = idx[selector[idx]]
                            strata.append({"stride": stride, "spacing_m": .05*stride, "phase": phase,
                                           "well": well, "descriptor": descriptor, "stratum": str(level),
                                           "n": len(selected), "changed": int(flip[selected].sum())})
        print(f"Validated and analyzed stride={stride}, phase={phase}", flush=True)
    pair = pd.DataFrame(pairs)
    pair.to_csv(results / "paired_counts_by_well_phase.csv", index=False)
    countcols = list(STATES) + ["rule_changed", "RF_vs_fine", "RF_minus_rule"]
    rated = pair.copy()
    for col in countcols:
        rated[col] = rated[col] / rated.n
    meanwell, summary = rates_summary(rated, ["scheme", "spacing_m"], countcols)
    meanwell.to_csv(results / "paired_phase_mean_by_well.csv", index=False)
    summary.to_csv(results / "paired_equal_well_summary.csv", index=False)
    blocked = pair.groupby(["scheme", "spacing_m", "phase", "block"], as_index=False)[["n"] + countcols].sum()
    for col in countcols:
        blocked[col] = blocked[col] / blocked.n
    blocked = blocked.groupby(["scheme", "spacing_m", "block"], as_index=False)[countcols].mean()
    blocked.to_csv(results / "paired_phase_mean_by_block.csv", index=False)
    blocked.groupby(["scheme", "spacing_m"], as_index=False)[countcols].mean().to_csv(results / "paired_equal_block_summary.csv", index=False)
    intervention = pd.DataFrame(interventions)
    intervention.to_csv(results / "channel_context_counts_by_well_phase.csv", index=False)
    intcols = ["changed", "base_stable_context_changed", "base_changed_full_changed", "base_changed_full_stable"]
    assert np.array_equal(intervention.changed, intervention.base_stable_context_changed + intervention.base_changed_full_changed)
    for col in intcols:
        intervention[col] = intervention[col] / intervention.n
    intwell, intsum = rates_summary(intervention, ["spacing_m", "channel"], intcols)
    intwell.to_csv(results / "channel_context_phase_mean_by_well.csv", index=False)
    intsum.to_csv(results / "channel_context_summary.csv", index=False)
    strat = pd.DataFrame(strata)
    strat.to_csv(results / "stratum_counts_by_well_phase.csv", index=False)
    strat["change_rate"] = strat.changed / strat.n.replace(0, np.nan)
    sw, ss = rates_summary(strat, ["spacing_m", "descriptor", "stratum"], ["change_rate"])
    sw.to_csv(results / "stratum_phase_mean_by_well.csv", index=False)
    ss.to_csv(results / "stratum_equal_well_summary.csv", index=False)
    pd.DataFrame(inputs).to_csv(HERE / "qa" / "input_fingerprints.csv", index=False)
    manifest = pd.read_csv(HERE / "qa" / "baseline_manifest.csv")
    baseline_present = all((HERE / row.path).is_file() for row in manifest.itertuples())
    if baseline_present:
        assert all(digest(HERE / row.path).upper() == row.sha256.upper() for row in manifest.itertuples())
    qa = {"completed_utc": datetime.now(timezone.utc).isoformat(), "tests": tests,
          "raw_rows": len(raw), "eligible_rows": len(data), "wells": len(wellgroups),
          "spatial_blocks": int(data.spatial_block.nunique()), "QC_exclusions": len(rejected),
          "QC_rule_vs_saved_labels": saved_delta, "validated_cache_files": 24,
          "scenarios": len(SCENARIOS), "retrained_models": 0, "external_validation": "NOT_PERFORMED",
          "cache_alignment_rule_target_probability_checks": "PASS", "partition_and_count_identities": "PASS",
          "baseline_unchanged": "PASS" if baseline_present else "NOT_RECHECKED_BASELINE_NOT_BUNDLED",
          "protocol_sha256": digest(HERE / "ANALYSIS_PROTOCOL.md"),
          "versions": {x: importlib.metadata.version(x) for x in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]}}
    (HERE / "qa" / "ANALYSIS_QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
