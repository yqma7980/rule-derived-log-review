"""Bounded tests: generated inputs only, no forest fitting or network calls."""
import itertools
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows/kgs/code"))
sys.path.insert(0, str(ROOT / "workflows/force/code/inherited"))
from external_core import features, interpolate, partition, rule
from external_review_trial import physical_blocks
from review_priority import evaluate_blocks, self_test
from audit_core import simplified_rule
from independent_oracles import paired_counts, review_scalar, rule_scalar


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.fit_guard = patch("sklearn.ensemble.RandomForestClassifier.fit", side_effect=AssertionError("Training forbidden"))
        self.net_guard = patch("socket.socket.connect", side_effect=AssertionError("Network forbidden"))
        self.fit_guard.start()
        self.net_guard.start()
        self.addCleanup(self.fit_guard.stop)
        self.addCleanup(self.net_guard.stop)

    def test_all_216_paired_states_and_identity(self):
        triples = np.array(list(itertools.product(range(6), repeat=3)))
        masks = partition(*triples.T)
        counts = paired_counts(*triples.T)
        self.assertEqual(masks.sum(axis=1).tolist(), list(counts.values()))
        self.assertEqual(counts, {"A": 6, "B": 30, "C": 30, "D": 30, "E": 120})
        f, c, p = triples.T
        np.testing.assert_array_equal((p != f).astype(int) - (c != f).astype(int),
                                      masks[1].astype(int) - masks[3].astype(int))

    def test_physical_rule_against_scalar_three_grids(self):
        rng = np.random.default_rng(20261101)
        for step in (.05, .152, .3048):
            gr, den = rng.uniform(1, 140, 101), rng.uniform(1.3, 3., 101)
            z = np.arange(101) * step
            np.testing.assert_array_equal(rule(gr, den, z, step)[0], rule_scalar(gr, den, z, step))

    def test_thresholds_and_two_metre_edges(self):
        for length in (39, 40, 41):
            gr = np.r_[np.repeat([10., 70.], 5), np.full(length, 50.), np.repeat([70., 95.], 5)]
            den = np.full(len(gr), 2.4)
            z = np.arange(len(gr)) * .05
            expected = rule_scalar(gr, den, z, .05)
            np.testing.assert_array_equal(rule(gr, den, z, .05)[0], expected)
            np.testing.assert_array_equal(simplified_rule(gr, den), expected)
        gr = np.array([20., 40., 60., 70., 80., 90., 90.0001])
        for d in (1.749999, 1.75, 2.299999, 2.30):
            z = np.arange(len(gr)) * .05
            den = np.full(len(gr), d)
            np.testing.assert_array_equal(rule(gr, den, z, .05)[0], rule_scalar(gr, den, z, .05))

    def test_constant_controls(self):
        for g, d, label in ((50., 2.4, 2), (50., 1.5, 0), (95., 2.4, 5)):
            np.testing.assert_array_equal(rule(np.full(80, g), np.full(80, d), np.arange(80) * .05, .05)[0],
                                          np.full(80, label))

    def test_interpolation_all_phases_and_endpoints(self):
        z = np.arange(31) * .152
        for stride in (1, 2, 4, 5):
            for phase in range(stride):
                x = 3 * z + 5
                np.testing.assert_allclose(interpolate(x, z, stride, phase), x, atol=1e-12)
                nonlinear = np.sin(z)
                got = interpolate(nonlinear, z, stride, phase)
                self.assertEqual(got[0], nonlinear[0])
                self.assertEqual(got[-1], nonlinear[-1])

    def test_zero_stride_control(self):
        z = np.arange(81) * .152
        gr, den = 55 + 33 * np.sin(z), 2.32 + .16 * np.cos(z)
        np.testing.assert_array_equal(rule(gr, den, z, .152)[0],
                                      rule(interpolate(gr, z, 1, 0), interpolate(den, z, 1, 0), z, .152)[0])

    def test_finite_26_features(self):
        z = np.arange(37) * .152
        x, spans = features(50 + z, 2.3 + z / 100, z, .152)
        self.assertEqual(x.shape, (37, 26))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(spans, [.304, .608, .912])

    def test_budget_inherited_selftests(self):
        self.assertEqual(self_test()["fixed_budget_denominator_tests"], 5)

    def test_budget_partial_block_capture_not_yield(self):
        target = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
        got = evaluate_blocks([2, 1], [0, 5], [5, 5], target, .3)
        self.assertEqual(got["reviewed_rows"], 3)
        self.assertEqual(got["capture_rate"], .5)
        self.assertEqual(got["review_yield"], 2 / 3)

    def test_budget_ties_against_independent_selector(self):
        for fraction in (0., .01, .1, .13, .3, .99, 1.):
            for scores in ([1., 1., 1.], [.1, .8, .4]):
                target = [i % 3 == 0 for i in range(23)]
                got = evaluate_blocks(scores, [0, 5, 15], [5, 10, 8], target, fraction, well="SYNTHETIC")
                self.assertEqual(got, review_scalar(scores, [0, 5, 15], [5, 10, 8], target, fraction))

    def test_ranking_is_independent_of_target(self):
        chosen = []
        for i in range(23):
            target = [j == i for j in range(23)]
            got = evaluate_blocks([1, 1, 1], [0, 5, 15], [5, 10, 8], target, .3, well="SYNTHETIC")
            if got["captured_rows"]:
                chosen.append(i)
        self.assertEqual(len(chosen), 6)
        for i in range(23):
            anti = [j != i for j in range(23)]
            got = evaluate_blocks([1, 1, 1], [0, 5, 15], [5, 10, 8], anti, .3, well="SYNTHETIC")
            self.assertEqual(got["captured_rows"], 6 - int(i in chosen))

    def test_zero_targets_are_undefined_not_zero_capture(self):
        got = evaluate_blocks([1], [0], [23], [False] * 23, .1)
        self.assertIsNone(got["capture_rate"])
        self.assertEqual(got["review_yield"], 0)

    def test_invalid_budget_blocks_and_nonfinite_scores(self):
        cases = [([1, 2], [0, 4], [5, 6], .3), ([1, 2], [0, 6], [5, 4], .3),
                 ([float("nan"), 2], [0, 5], [5, 5], .3), ([1, 2], [0, 5], [5, 5], -1),
                 ([1, 2], [0, 5], [5, 5], 1.1), ([1, 2], [-1, 5], [6, 5], .3)]
        for scores, starts, sizes, budget in cases:
            with self.assertRaises(ValueError):
                evaluate_blocks(scores, starts, sizes, [False] * 10, budget)

    def test_physical_blocks_do_not_bridge_segments(self):
        part = pd.DataFrame({"segment": [0] * 12 + [1] * 9,
                             "depth": [i * .152 for i in range(12)] + [10 + i * .152 for i in range(9)]})
        starts, sizes = physical_blocks(part)
        self.assertEqual(starts.tolist(), [0, 7, 12, 19])
        self.assertEqual(sizes.tolist(), [7, 5, 7, 2])
        self.assertEqual(sum(sizes), len(part))

    def test_rule_rejects_bad_depth_and_nonfinite(self):
        for z in ([0, 0, 1], [0, 2, 1], [0, float("nan"), 2]):
            with self.assertRaises(ValueError):
                rule([50] * 3, [2.4] * 3, z, .05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
