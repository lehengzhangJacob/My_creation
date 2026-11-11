#!/usr/bin/env python3
"""Unit tests for reviewer-experiment helpers. No GPU and no checkpoints."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import numpy as np

from analyze_nonlanguage_primary import summarize_groups, uncertain_subset
from corrupt_inputs_on_split import (
    apply_corruption,
    confidence_delta,
    corruption_mask,
    force_primary,
    modalities_to_corrupt,
    window_bounds,
)
from define_conflict_subset import (
    conflict_from_probe_signs,
    conflict_from_unimodal_labels,
    fit_unimodal_probe,
    predict_probe,
)
from parse_routing_diag import parse_routing_diag


LOG = """\
Epoch 1/3  [stage 1]
  Diag  w=[a:0.333 l:0.334 v:0.333]  target=[a:0.000 l:0.000 v:0.000]  primary=[a:0.34 l:0.33 v:0.33]
Epoch 2/3  [stage 2]
  Diag  w=[a:0.200 l:0.600 v:0.200]  target=[a:0.100 l:0.800 v:0.100]  primary=[a:0.20 l:0.59 v:0.21]
noise Diag ignored because it is incomplete
"""


class RoutingDiagTests(unittest.TestCase):
    def test_parses_epoch_stage_and_triples(self):
        rows = parse_routing_diag(LOG, "mosei", "full")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["epoch"], "2")
        self.assertEqual(rows[1]["stage"], "2")
        self.assertEqual(rows[1]["primary_l"], "0.59")
        self.assertEqual(rows[0]["w_a"], "0.333")


class CorruptionTests(unittest.TestCase):
    def test_window_stays_inside_and_mask_matches(self):
        start, end = window_bounds(10, 0.4, start=3)
        mask = corruption_mask(10, start, end)
        self.assertEqual(int(mask.sum()), end - start)
        self.assertTrue(mask[start:end].all())
        self.assertFalse(mask[:start].any())
        self.assertFalse(mask[end:].any())

    def test_zero_and_shuffle_touch_only_the_window(self):
        values = np.arange(8, dtype=float)[:, None]
        rng = np.random.default_rng(0)
        zeroed = apply_corruption(values, 2, 5, "zero", 1.0, rng)
        self.assertTrue(np.all(zeroed[2:5] == 0))
        self.assertTrue(np.array_equal(zeroed[:2], values[:2]))
        self.assertTrue(np.array_equal(zeroed[5:], values[5:]))
        shuffled = apply_corruption(values, 2, 5, "shuffle", 1.0, np.random.default_rng(1))
        self.assertTrue(np.array_equal(shuffled[:2], values[:2]))
        self.assertCountEqual(shuffled[2:5, 0].tolist(), values[2:5, 0].tolist())

    def test_perturb_scope_and_fixed_clean_route(self):
        self.assertEqual(
            modalities_to_corrupt("language", "auxiliary"),
            {"acoustic", "visual"},
        )
        self.assertEqual(modalities_to_corrupt("acoustic", "primary"), {"acoustic"})
        weights = np.array([[0.2, 0.7, 0.1], [0.5, 0.1, 0.4]])
        forced = force_primary(weights, clean_primary=1)
        self.assertTrue(np.array_equal(forced[:, 1], np.ones(2)))
        self.assertTrue(np.all(forced[:, [0, 2]] == 0))

    def test_confidence_delta_splits_positions(self):
        clean = np.array([0.2, 0.2, 0.2, 0.2])
        corrupt = np.array([0.2, 0.0, 0.0, 0.2])
        mask = np.array([False, True, True, False])
        delta = confidence_delta(clean, corrupt, mask)
        self.assertAlmostEqual(delta["delta_corrupted"], -0.2)
        self.assertAlmostEqual(delta["delta_clean"], 0.0)


class ConflictTests(unittest.TestCase):
    def test_opposite_sign_is_conflict_and_is_reproducible(self):
        text = np.array([0.8, 0.8, 0.05, -0.4])
        audio = np.array([-0.5, 0.2, -0.9, -0.3])
        visual = np.array([0.4, -0.6, 0.2, 0.0])
        flags, reason = conflict_from_unimodal_labels(text, audio, visual, min_abs=0.1)
        again, reason2 = conflict_from_unimodal_labels(text, audio, visual, min_abs=0.1)
        np.testing.assert_array_equal(flags, np.array([True, True, False, False]))
        np.testing.assert_array_equal(reason, np.array([1, 2, 0, 0]))
        np.testing.assert_array_equal(flags, again)
        np.testing.assert_array_equal(reason, reason2)

    def test_probe_signs_match_the_label_rule(self):
        flags, reason = conflict_from_probe_signs(
            np.array([1, 1, 0]), np.array([-1, 1, -1]), np.array([1, -1, 1])
        )
        np.testing.assert_array_equal(flags, np.array([True, True, False]))
        self.assertEqual(reason.tolist(), [1, 2, 0])

    def test_probe_fits_a_linear_feature(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = 0.5 * x[:, 0] - 0.25
        weights = fit_unimodal_probe(x, y)
        pred = predict_probe(x, weights)
        self.assertLess(np.max(np.abs(pred - y)), 1e-8)


class NonlanguageTests(unittest.TestCase):
    def test_groups_and_uncertain_slice(self):
        rows = [
            {"primary_name": "language", "label": "1", "abs_err": "0.1",
             "w_max": "0.9", "ib_conf_t": "0.8", "ib_conf_a": "0.4", "ib_conf_v": "0.4"},
            {"primary_name": "acoustic", "label": "-1", "abs_err": "0.4",
             "w_max": "0.4", "ib_conf_t": "0.3", "ib_conf_a": "0.7", "ib_conf_v": "0.3"},
        ]
        groups = {row["group"]: row for row in summarize_groups(rows)}
        self.assertEqual(groups["acoustic"]["n"], "1")
        self.assertEqual(groups["visual"]["n"], "0")
        uncertain = uncertain_subset(rows, quantile=0.5)
        self.assertEqual(uncertain["n_uncertain"], "1")


class SubsetEvalTests(unittest.TestCase):
    def test_subset_metrics_and_bootstrap_file(self):
        from evaluate_on_subset import load_conflict_index, subset_metrics

        labels = np.array([1.0, -1.0, 0.5])
        preds = np.array([0.8, 0.2, 0.4])
        conflict = np.array([True, True, False])
        metrics = subset_metrics(labels, preds, conflict)
        self.assertEqual(metrics["conflict"]["n"], 2)
        self.assertAlmostEqual(metrics["conflict"]["mae"], 0.7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subset.csv"
            path.write_text(
                "global_index,is_conflict,reason,source\n"
                "0,1,text_vs_audio,test\n"
                "1,0,none,test\n",
                encoding="utf-8",
            )
            loaded = load_conflict_index(path)
            np.testing.assert_array_equal(loaded, np.array([True, False]))


if __name__ == "__main__":
    unittest.main()
