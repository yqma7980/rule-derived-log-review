"""Frozen-context diagnostics and contiguous interpretation-change intervals.

Recomputes rule outputs and scores without fitting models.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.environ["M001_CONTROLLED_ROOT"]
S12 = ROOT + "/GAS投稿/stage12_substantive_revision_20260907"
S6 = ROOT + "/NRR投稿/stage6_submission_readiness"
sys.path.insert(0, os.environ["M001_RECOVERED_S12"] + "/code")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import audit_core as core            # noqa: E402
import legacy_common as common       # noqa: E402
_range_sum = core._range_sum

OUT = os.environ["M001_RECOVERY_OUTPUT"]
SCHEMES = ("LEAVE_ONE_WELL_OUT", "LEAVE_ONE_SPATIAL_BLOCK_OUT")
SCENARIOS = [(s, p) for s in (2, 4, 5) for p in range(s)]
BUDGET = 0.10


def rule_parts(gr, den):
    """Return (lithology state, context gate): thickness<2 & hetero & ~protected."""
    lith = np.select([(den < 1.75) & (gr < 70), (gr >= 80) & (den >= 2.30),
                      (gr >= 60) & (gr < 80), (gr >= 40) & (gr < 60),
                      (gr >= 20) & (gr < 40), gr < 20], [0, 1, 2, 3, 4, 5], default=1)
    n = len(gr)
    starts = np.r_[0, np.flatnonzero(lith[1:] != lith[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    thickness = np.repeat(lengths, lengths) * 0.05
    i = np.arange(n)
    left, right = np.maximum(0, i - 10), np.minimum(n, i + 10)
    unique_count = sum(_range_sum(lith == k, left, right) > 0 for k in range(6))
    hetero = (unique_count >= 3) & (_range_sum(lith == 2, left, right) > 0) & (_range_sum(lith == 3, left, right) > 0)
    deeper = _range_sum(np.isin(lith, [1, 2]), i + 1, np.minimum(n, i + 6)) > 0
    shallower = _range_sum(np.isin(lith, [4, 5]), np.maximum(0, i - 5), i) > 0
    gate = (thickness < 2.0) & hetero & ~(deeper & shallower)
    return lith, gate


def label_from(lith, gate, gr):
    ctx = (lith == 3) & gate
    return np.select([lith == 0, lith == 1, lith == 2, np.isin(lith, [4, 5]), ctx],
                     [0, np.where(gr > 90, 5, 4), 3, 1, 3], default=2).astype(np.int16)


def labels_frozen(frame, gate_base):
    """Perturb lithology/threshold responses while retaining baseline context gates."""
    out = np.full(len(frame), -1, np.int16)
    for key, idx in frame.groupby(["well_name", "segment_id"], sort=False).indices.items():
        part = frame.iloc[idx]
        lith, _ = rule_parts(part.GR.to_numpy(), part.DEN.to_numpy())
        out[idx] = label_from(lith, gate_base[key], part.GR.to_numpy())
    assert (out >= 0).all()
    return out


def degrade(data, stride, phase):
    out = data.copy()
    if stride > 1:
        for idx in out.groupby(["well_name", "segment_id"], sort=False).indices.values():
            part = out.iloc[idx]
            anchors = np.unique(np.r_[0, np.arange(phase, len(idx), stride), len(idx) - 1]).astype(int)
            for log in ("GR", "DEN"):
                out.loc[idx, log] = np.interp(part.depth, part.depth.iloc[anchors], part[log].iloc[anchors])
    return out


def baseline_gates(frame):
    out = {}
    for key, idx in frame.groupby(["well_name", "segment_id"], sort=False).indices.items():
        part = frame.iloc[idx]
        out[key] = rule_parts(part.GR.to_numpy(), part.DEN.to_numpy())[1]
    return out


def risks(frame, labels, gates, amplitude=1.0):
    full = np.zeros(len(frame), float)
    frozen = np.zeros(len(frame), float)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if not (a or b):
                continue
            ch = frame.copy()
            ch["GR"] += a * amplitude
            ch["DEN"] += b * amplitude * 0.01
            full += core.apply_grouped_rule(ch) != labels
            frozen += labels_frozen(ch, gates) != labels
    return full / 8, frozen / 8


def interval_ids(target, seg):
    new_seg = np.r_[True, seg[1:] != seg[:-1]]
    starts = target & (~np.r_[False, target[:-1]] | new_seg)
    ids = np.cumsum(starts) - 1
    return np.where(target, ids, -1), int(starts.sum())


def evaluate(risk, target, seg, well, interval_of, n_interval):
    blocks = core.review_blocks(seg)
    scores = np.array([risk[x].mean() for x in blocks])
    order = np.lexsort((core.stable_ties(well, len(blocks)), -scores))
    mask = core.select_review(blocks, order, int(np.floor(BUDGET * len(target))), len(target))
    hit = np.unique(interval_of[mask & target])
    full = [i for i in hit if (interval_of == i).sum() == (mask & (interval_of == i)).sum()]
    return {"capture": float((mask & target).sum() / target.sum()) if target.sum() else np.nan,
            "intervals_total": n_interval,
            "intervals_touched": int(len(hit)),
            "intervals_fully_covered": int(len(full)),
            "touched_rate": len(hit) / n_interval if n_interval else np.nan,
            "full_rate": len(full) / n_interval if n_interval else np.nan}


def main():
    data, _ = common.load_data(ROOT + "/IJCG投稿/data/measurement_data.csv", apply_qc=True)
    blocks = pd.read_csv(ROOT + "/IJCG投稿/outputs/geology_aware_subfacies_validation_v1/well_spatial_blocks_v1.csv",
                         usecols=["well_id", "spatial_block"]).rename(columns={"well_id": "well_name"})
    data = data.merge(blocks, on="well_name", how="left", validate="many_to_one", sort=False)
    folds = pd.read_csv(S12 + "/results/calibration_fold_details.csv")
    fine = core.apply_grouped_rule(data)

    rows = []
    for stride, phase in SCENARIOS:
        frame = degrade(data, stride, phase)
        labels = core.apply_grouped_rule(frame)
        target = labels != fine
        gates = baseline_gates(frame)
        r_full, r_frozen = risks(frame, labels, gates)
        ref = core.perturbation_risk(frame, 1.0, labels)
        assert np.allclose(r_full, ref), "全上下文扰动应复现既有实现"
        r_thr = core.threshold_risk(frame)
        for scheme in SCHEMES:
            z = np.load(S12 + f"/private/triage_{scheme.lower()}_s{stride}_p{phase}.npz")
            np.testing.assert_array_equal(z["rule_class"].astype(int), labels)
            np.testing.assert_array_equal(z["target"].astype(bool), target)
            q = z["probability"].astype(np.float64)
            well_t = {}
            for r in folds[folds.validation_scheme == scheme].itertuples():
                for w in str(r.test_wells).split(";"):
                    well_t[w.strip()] = float(r.temperature)
            T = np.array([well_t[w] for w in data.well_name])
            p = q ** T[:, None]
            p = p / p.sum(axis=1, keepdims=True)
            s_cal = 1.0 - q[np.arange(len(q)), labels]
            s_raw = 1.0 - p[np.arange(len(p)), labels]
            for w, idx in data.groupby("well_name", sort=True).indices.items():
                part = data.iloc[idx]
                seg = part.segment_id.to_numpy()
                t = target[idx]
                iid, nint = interval_ids(t, seg)
                for name, risk in (("STABILITY_1", r_full), ("STABILITY_1_FROZEN", r_frozen),
                                   ("THRESHOLD_DISTANCE", r_thr), ("RF_SUPPORT_CAL", s_cal),
                                   ("RF_SUPPORT_RAW", s_raw), ("RANDOM", np.zeros(len(data)))):
                    rows.append({"scheme": scheme, "stride": stride, "spacing_m": stride * .05,
                                 "phase": phase, "well": w, "block": str(part.spatial_block.iloc[0]),
                                 "method": name, **evaluate(risk[idx], t, seg, w, iid, nint)})
        print("scenario stride=%d phase=%d done" % (stride, phase), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT + "/mechanism_and_intervals_phases.csv", index=False, encoding="utf-8-sig")
    df["test_group"] = np.where(df.scheme == SCHEMES[0], df.well, df.block)
    g = df.groupby(["scheme", "stride", "spacing_m", "phase", "test_group", "method"], as_index=False).mean(numeric_only=True)
    s = g.groupby(["scheme", "stride", "spacing_m", "method"], as_index=False).mean(numeric_only=True)
    s.to_csv(OUT + "/mechanism_and_intervals_summary.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 260)
    for scheme in SCHEMES:
        sub = s[s.scheme == scheme]
        for c in ("capture", "touched_rate", "full_rate"):
            v = sub.pivot_table(index=["stride", "spacing_m"], columns="method", values=c) * 100
            print("\n== %s  %s (%%)" % (scheme, c))
            print(v.round(2).to_string())
        v = sub.pivot_table(index=["stride", "spacing_m"], columns="method", values="intervals_total")
        print("\n== %s  变化区间数" % scheme)
        print(v.round(1).to_string())


if __name__ == "__main__":
    main()
