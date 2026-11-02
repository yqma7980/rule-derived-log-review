"""Scalar/list reference calculations, deliberately independent of NumPy code."""
import bisect
import hashlib
import math


def rule_scalar(gr, den, z, step):
    lith = []
    for g, d in zip(gr, den):
        if d < 1.75 and g < 70:
            lith.append(0)
        elif g >= 80 and d >= 2.30:
            lith.append(1)
        elif 60 <= g < 80:
            lith.append(2)
        elif 40 <= g < 60:
            lith.append(3)
        elif 20 <= g < 40:
            lith.append(4)
        elif g < 20:
            lith.append(5)
        else:
            lith.append(1)
    answer = []
    for i, kind in enumerate(lith):
        if kind != 3:
            answer.append({0: 0, 1: 5 if gr[i] > 90 else 4, 2: 3, 4: 1, 5: 1}[kind])
            continue
        left = right = i
        while left > 0 and lith[left - 1] == kind:
            left -= 1
        while right + 1 < len(lith) and lith[right + 1] == kind:
            right += 1
        neighborhood = {lith[j] for j in range(len(z)) if -.5 - 1e-8 <= z[j] - z[i] < .5 - 1e-8}
        deeper = any(lith[j] in (1, 2) and z[j] - z[i] <= .25 + 1e-8 for j in range(i + 1, len(z)))
        shallower = any(lith[j] in (4, 5) and z[i] - z[j] <= .25 + 1e-8 for j in range(i))
        contextual = (z[right] - z[left] + step < 2. - 1e-8 and len(neighborhood) >= 3
                      and {2, 3} <= neighborhood and not (deeper and shallower))
        answer.append(3 if contextual else 2)
    return answer


def interpolate_scalar(values, z, stride, phase):
    anchors = sorted({0, len(z) - 1, *range(phase, len(z), stride)})
    za = [z[i] for i in anchors]
    result = []
    for depth in z:
        pos = bisect.bisect_left(za, depth)
        if pos < len(za) and za[pos] == depth:
            result.append(values[anchors[pos]])
        else:
            lo, hi = anchors[pos - 1], anchors[pos]
            weight = (depth - z[lo]) / (z[hi] - z[lo])
            result.append(values[lo] + weight * (values[hi] - values[lo]))
    return result


def paired_counts(fine, coarse, prediction):
    counts = dict.fromkeys("ABCDE", 0)
    mapping = {7: "A", 4: "B", 2: "C", 1: "D", 0: "E"}
    for f, c, p in zip(fine, coarse, prediction):
        bits = 4 * int(f == c) + 2 * int(c == p) + int(f == p)
        counts[mapping[bits]] += 1
    return counts


def review_scalar(scores, starts, sizes, target, budget, well="SYNTHETIC"):
    keys = [int.from_bytes(hashlib.sha256(f"20260907|{well}|{i}".encode()).digest()[:8], "big")
            for i in range(len(scores))]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], keys[i]))
    need = math.floor(budget * len(target))
    ordered_rows = [j for i in order for j in range(starts[i], starts[i] + sizes[i])]
    selected = ordered_rows[:need]
    if len(selected) != need or len(set(ordered_rows)) != len(target):
        raise AssertionError("Invalid independent selection")
    captured = sum(bool(target[j]) for j in selected)
    total = sum(bool(v) for v in target)
    return {"budget_rows": need, "reviewed_rows": len(selected), "target_rows": total,
            "captured_rows": captured, "capture_rate": captured / total if total else None,
            "review_yield": captured / need if need else None}
