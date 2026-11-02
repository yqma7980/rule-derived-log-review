"""Portable S12 audit functions. Inputs never include hidden test targets."""
import hashlib
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def _range_sum(mask, start, stop):
    prefix = np.r_[0, np.cumsum(mask, dtype=np.int64)]
    return prefix[stop] - prefix[start]


def simplified_rule(gr, den, return_diagnostics=False):
    """Equivalent historical return logic for ONE ordered well/segment, zero-based."""
    gr, den = np.asarray(gr, float), np.asarray(den, float)
    if gr.ndim != 1 or gr.shape != den.shape or not len(gr) or not np.isfinite([gr, den]).all():
        raise ValueError("Finite, nonempty, equal-length 1D GR/DEN required")
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
    protected_mb = deeper & shallower
    context_ss = (lith == 3) & (thickness < 2.0) & hetero & ~protected_mb
    label = np.select([lith == 0, lith == 1, lith == 2, np.isin(lith, [4, 5]), context_ss],
                      [0, np.where(gr > 90, 5, 4), 3, 1, 3], default=2).astype(np.int16)
    if return_diagnostics:
        return label, {"lithology": lith, "fine_context_changes_default": context_ss,
                       "fine_priority_mb": (lith == 3) & protected_mb,
                       "thickness": thickness}
    return label


def apply_grouped_rule(frame):
    labels = np.full(len(frame), -1, np.int16)
    for idx in frame.groupby(["well_name", "segment_id"], sort=False).indices.values():
        part = frame.iloc[idx]
        labels[idx] = simplified_rule(part.GR.to_numpy(), part.DEN.to_numpy())
    if np.any(labels < 0):
        raise ValueError("Incomplete grouped rule")
    return labels


def transitions(depth, labels):
    depth, labels = np.asarray(depth, float), np.asarray(labels, int)
    if depth.ndim != 1 or depth.shape != labels.shape or not np.isfinite(depth).all() or np.any(np.diff(depth) <= 0):
        raise ValueError("Depth and labels must be aligned and depth strictly increasing")
    pos = np.flatnonzero(labels[1:] != labels[:-1])
    return np.column_stack(((depth[pos] + depth[pos + 1]) / 2, labels[pos], labels[pos + 1]))


def matching_edges(observed, predicted, tolerance, aware):
    observed, predicted = np.asarray(observed, float).reshape(-1, 3), np.asarray(predicted, float).reshape(-1, 3)
    if not np.isfinite(tolerance) or tolerance < 0 or not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Invalid boundary records/tolerance")
    edges = []
    for i, obs in enumerate(observed):
        dist = np.abs(predicted[:, 0] - obs[0])
        valid = dist <= tolerance + 1e-12
        if aware:
            valid &= np.all(predicted[:, 1:] == obs[1:], axis=1)
        edges.extend((i, int(j), float(dist[j])) for j in np.flatnonzero(valid))
    return edges


def boundary_matches(observed, predicted, tolerance, aware=False, method="optimal"):
    """Max cardinality, then min offset, using positive sparse costs + private dummies.

    Every observed vertex gets a private dummy. P > n*(1+tol) makes one fewer
    real match more costly than all possible real-edge cost savings combined.
    For fixed cardinality the +1 shift is constant, leaving min total offset.
    """
    n, m = len(observed), len(predicted)
    edges = matching_edges(observed, predicted, tolerance, aware)
    if not edges:
        return []
    if method == "greedy":
        used_o, used_p, out = set(), set(), []
        for i, j, d in sorted(edges, key=lambda x: (x[2], x[0], x[1])):
            if i not in used_o and j not in used_p:
                out.append((i, j, d)); used_o.add(i); used_p.add(j)
        return sorted(out)
    if method != "optimal":
        raise ValueError(method)
    penalty = (n + 1) * (tolerance + 2)
    rows = [x[0] for x in edges] + list(range(n))
    cols = [x[1] for x in edges] + list(range(m, m + n))
    weights = [1 + x[2] for x in edges] + [penalty] * n
    graph = csr_matrix((weights, (rows, cols)), shape=(n, m + n))
    rid, cid = min_weight_full_bipartite_matching(graph)
    distance = {(i, j): d for i, j, d in edges}
    return sorted((int(i), int(j), distance[(i, j)]) for i, j in zip(rid, cid) if j < m)


def boundary_f1(n_observed, n_predicted, n_matched):
    return 2.0 * n_matched / (n_observed + n_predicted) if n_observed + n_predicted else math.nan


def run_records(labels):
    labels = np.asarray(labels, int)
    if not len(labels):
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    stops = np.r_[starts[1:], len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(starts, stops)]


def directional_run_iou(labels_a, labels_b):
    """Best overlap per source run; explicitly not one-to-one matching."""
    a, b = run_records(labels_a), run_records(labels_b)
    scores = []
    for left, right, label in a:
        best = 0.0
        for bl, br, bc in b:
            if bl >= right:
                break
            if br <= left or bc != label:
                continue
            overlap = min(right, br) - max(left, bl)
            best = max(best, overlap / (right - left + br - bl - overlap))
        scores.append(best)
    return np.asarray(scores)


def temperature_scale(probability, temperature):
    p = np.asarray(probability, float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Invalid class probability")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid temperature")
    z = np.log(np.clip(p, 1e-12, 1)) / temperature
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def threshold_risk(frame):
    gr = np.min(np.abs(frame.GR.to_numpy()[:, None] - np.array([20, 40, 60, 70, 80, 90])), axis=1)
    den = np.min(np.abs(frame.DEN.to_numpy()[:, None] - np.array([1.75, 2.30])), axis=1) / 0.01
    return -np.minimum(gr, den)


def perturbation_risk(frame, amplitude, baseline):
    risks = np.zeros(len(frame), float)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if not (a or b):
                continue
            changed = frame.copy()
            changed["GR"] += a * amplitude
            changed["DEN"] += b * amplitude * 0.01
            risks += apply_grouped_rule(changed) != baseline
    return risks / 8


def review_blocks(segment_ids, block_rows=20):
    seg = np.asarray(segment_ids)
    starts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
    stops = np.r_[starts[1:], len(seg)]
    return [np.arange(a, min(a + block_rows, stop)) for start, stop in zip(starts, stops) for a in range(start, stop, block_rows)]


def stable_ties(well_name, n):
    return np.array([int.from_bytes(hashlib.sha256(f"20260907|{well_name}|{i}".encode()).digest()[:8], "big") for i in range(n)], dtype=np.uint64)


def select_review(blocks, order, budget_rows, n):
    mask = np.zeros(n, bool)
    remaining = budget_rows
    for b in order:
        use = blocks[int(b)][:remaining]
        mask[use] = True
        remaining -= len(use)
        if remaining == 0:
            break
    if remaining != 0 or mask.sum() != budget_rows:
        raise ValueError("Review budget accounting failed")
    return mask


def review_scores(risk, target, segment_ids, well_name, budget, random_seed=None):
    risk, target = np.asarray(risk, float), np.asarray(target, bool)
    n = len(target)
    if risk.shape != target.shape or not np.isfinite(risk).all() or not 0 <= budget <= 1:
        raise ValueError("Invalid review input")
    blocks = review_blocks(segment_ids)
    scores = np.array([risk[x].mean() for x in blocks])
    order = np.lexsort((stable_ties(well_name, len(blocks)), -scores)) if random_seed is None else np.random.default_rng(random_seed).permutation(len(blocks))
    mask = select_review(blocks, order, int(np.floor(budget * n)), n)
    segment_start = np.r_[True, np.asarray(segment_ids)[1:] != np.asarray(segment_ids)[:-1]]
    interval_count = np.count_nonzero(mask & (segment_start | ~np.r_[False, mask[:-1]]))
    captured = int(np.count_nonzero(mask & target))
    errors, reviewed = int(target.sum()), int(mask.sum())
    return {"target_rows": errors, "reviewed_rows": reviewed, "captured_rows": captured,
            "capture": captured / errors if errors else math.nan,
            "residual_error": (errors - captured) / (n - reviewed) if n > reviewed else math.nan,
            "reviewed_length_m": reviewed * 0.05, "target_length_m": errors * 0.05,
            "residual_length_m": (errors - captured) * 0.05, "merged_intervals": int(interval_count)}
