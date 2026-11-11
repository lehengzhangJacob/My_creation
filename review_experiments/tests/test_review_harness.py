from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HARNESS_DIR = Path(__file__).resolve().parents[1]
MY_CREATION_DIR = HARNESS_DIR.parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))
if str(MY_CREATION_DIR) not in sys.path:
    sys.path.insert(0, str(MY_CREATION_DIR))
if str(MY_CREATION_DIR / "ablation_study") not in sys.path:
    sys.path.insert(0, str(MY_CREATION_DIR / "ablation_study"))

import infogate_modules
from data_humor import features_to_dataset
from collect_results import main as collect_main
from collect_results import parse_dev_selected_log
from launch_review_matrix import build_command
from launch_review_matrix import main as launch_main
from paired_bootstrap import load_prediction_csv, paired_bootstrap, validate_pair
from review_common import checkpoint_dir, load_manifest


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.experiments = cls.manifest["experiments"]

    def test_matrix_counts_and_seed(self) -> None:
        self.assertEqual(len(self.experiments), 33)
        actions = Counter(exp["action"] for exp in self.experiments)
        self.assertEqual(
            actions, {"train": 18, "reuse": 13, "external_running": 2}
        )
        new_families = Counter(
            exp["family"] for exp in self.experiments if exp["action"] == "train"
        )
        self.assertEqual(
            new_families,
            {"backbone": 2, "confidence": 4, "routing": 8, "hcf": 4},
        )
        self.assertTrue(
            all(exp["seed"] == 128 for exp in self.experiments if exp["action"] == "train")
        )

    def test_existing_controls_are_reuse_only(self) -> None:
        protected = {
            "full_token_gumbel_deberta",
            "full_token_gumbel",
            "no_confidence",
            "fixed_text_primary",
            "no_infogate",
            "no_adaptive_gate",
        }
        matching = [
            exp for exp in self.experiments if exp.get("condition") in protected
        ]
        self.assertGreaterEqual(len(matching), 13)
        for exp in matching:
            self.assertEqual(exp["action"], "reuse", exp["id"])
            self.assertNotIn("launcher", exp)
            self.assertNotIn("overrides", exp)

    def test_new_commands_call_frozen_launchers(self) -> None:
        commands = []
        for exp in self.experiments:
            if exp["action"] != "train":
                continue
            command = build_command(exp, "/env/ITHP5090/bin/python")
            commands.append(command)
            self.assertIn("/ablation_study/train_fixed_", command[2])
            self.assertTrue(command[2].endswith(".py"))
            self.assertNotEqual(Path(command[2]).name, "train.py")
            self.assertIn("--checkpoint-dir", command)
            checkpoint_index = command.index("--checkpoint-dir") + 1
            self.assertEqual(Path(command[checkpoint_index]), checkpoint_dir(exp))
        self.assertEqual(len(commands), 18)

    def test_external_bert_is_structurally_non_runnable(self) -> None:
        bert_entries = [
            exp
            for exp in self.experiments
            if exp["id"] in {"mosi_prism_bert_s128", "mosei_prism_bert_s128"}
        ]
        self.assertEqual(len(bert_entries), 2)
        for exp in bert_entries:
            self.assertEqual(exp["action"], "external_running")
            self.assertNotIn("launcher", exp)
            self.assertNotIn("overrides", exp)
            self.assertIn("never-launch", exp["tags"])

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            return_code = launch_main(["--dry-run", "--all"])
        self.assertEqual(return_code, 0)
        output = stdout.getvalue()
        self.assertIn("mosi_prism_bert_s128 [NEVER LAUNCH]", output)
        self.assertIn("mosei_prism_bert_s128 [NEVER LAUNCH]", output)
        self.assertNotIn("bert-base-uncased", output)

    def test_dry_run_does_not_start_training(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return_code = launch_main(
                ["--dry-run", "--only", "mosei_router_argmax_s128"]
            )
        self.assertEqual(return_code, 0)
        output = stdout.getvalue()
        self.assertIn("train_fixed_mosei_phase1_trial70.py", output)
        self.assertIn("--routing-mode argmax", output)
        self.assertNotIn("START", output)

    def test_hcf_pairs_and_no_hcf_structure(self) -> None:
        by_id = {exp["id"]: exp for exp in self.experiments}
        for dataset in ("ur_funny", "mustard"):
            hcf = by_id[f"{dataset}_hcf_s128"]
            no_hcf = by_id[f"{dataset}_no_hcf_s128"]
            self.assertEqual(hcf["seed"], 128)
            self.assertEqual(no_hcf["seed"], 128)
            self.assertNotIn("--disable-hcf", hcf["overrides"])
            self.assertIn("--disable-hcf", no_hcf["overrides"])

        tiny_args = {
            "text_dim": 2,
            "acoustic_dim": 2,
            "visual_dim": 2,
            "hcf_dim": 0,
            "unified_dim": 4,
            "ib_hidden_dim": 4,
            "bottleneck_dim": 2,
            "num_heads": 1,
            "num_infogate_layers": 1,
            "dropout_prob": 0.0,
            "use_l_lib": False,
            "use_l_rib": False,
            "task_type": "binary",
        }
        model = infogate_modules.InfoGate(tiny_args)
        self.assertFalse(model.use_hcf)
        self.assertEqual(model.modalities, ("t", "a", "v"))
        self.assertEqual(model.mselector.num_modalities, 3)
        self.assertFalse(hasattr(model, "proj_h"))
        self.assertFalse(hasattr(model, "ib_enc_h"))
        self.assertFalse(hasattr(model.mselector, "W_hcf"))

        feature = {
            "input_ids": [1, 2],
            "visual": [[0.0, 0.0], [0.0, 0.0]],
            "acoustic": [[0.0, 0.0], [0.0, 0.0]],
            "hcf": [[0.0] * 4, [0.0] * 4],
            "label": 1.0,
        }
        self.assertEqual(len(features_to_dataset([feature], include_hcf=False).tensors), 4)
        self.assertEqual(len(features_to_dataset([feature], include_hcf=True).tensors), 5)


class ResultCollectorTests(unittest.TestCase):
    def test_parser_uses_dev_selected_block_not_oracle(self) -> None:
        text = """
InfoGate Training
Epoch 4/10  [stage 2]
  >> Best model saved (mae=0.4) to checkpoint.pt
Best Results (mae, epoch >= 3):
  Selection score: 0.400000
  Acc-2: 80.00%
  Acc-7: 50.00%
  MAE:   0.5000
  Corr:  0.7000
  F1:    0.8000

Last Epoch (10) Results:
  Acc-2: 70.00%
  MAE: 0.7000

Best Test MAE (oracle, for reference only):
  Epoch: 8
  Acc-2: 99.00%
  MAE: 0.1000
"""
        parsed = parse_dev_selected_log(text)
        self.assertEqual(parsed["selection_metric"], "mae")
        self.assertEqual(parsed["epoch"], 4)
        self.assertAlmostEqual(parsed["selection_score"], 0.4)
        self.assertAlmostEqual(parsed["test_results"]["acc2"], 0.8)
        self.assertAlmostEqual(parsed["test_results"]["mae"], 0.5)

    def test_collect_existing_log_to_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "results"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = collect_main(
                    [
                        "--only",
                        "mosei_full_deberta_s128_reuse",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(return_code, 0)
            document = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(document["counts"]["complete"], 1)
            result = document["results"][0]
            self.assertEqual(result["provenance"]["source"], "log")
            self.assertFalse(result["test_oracle"])
            self.assertAlmostEqual(result["test_results"]["mae"], 0.4994)
            self.assertTrue((output_dir / "summary.csv").is_file())

    def test_import_external_checkpoint_explicitly(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "external.pt"
            torch.save(
                {
                    "test_results": {
                        "acc2": 0.81,
                        "acc7": 0.49,
                        "mae": 0.61,
                        "corr": 0.82,
                        "f1": 0.80,
                    },
                    "selection_metric": "mae",
                    "selection_score": 0.60,
                    "epoch": 12,
                    "args": {"dataset": "mosi", "seed": 128},
                    "test_preds": [0.2, -0.4],
                    "test_labels": [0.0, -1.0],
                },
                checkpoint,
            )
            output_dir = root / "results"
            with contextlib.redirect_stdout(io.StringIO()):
                return_code = collect_main(
                    [
                        "--only",
                        "mosi_prism_bert_s128",
                        "--checkpoint",
                        f"mosi_prism_bert_s128={checkpoint}",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(return_code, 0)
            document = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            result = document["results"][0]
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["provenance"]["source"], "checkpoint")
            self.assertTrue(result["provenance"]["explicit_import"])
            self.assertEqual(result["epoch"], 12)
            self.assertTrue(result["predictions"]["available"])


class PairedBootstrapTests(unittest.TestCase):
    @staticmethod
    def _write_csv(
        path: Path,
        labels: list[float],
        predictions: list[float],
        *,
        binary: bool = False,
        sample_ids: list[int] | None = None,
    ) -> None:
        ids = sample_ids if sample_ids is not None else list(range(len(labels)))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_index", "true_label", "prediction"])
            for sample_id, label, prediction in zip(ids, labels, predictions):
                writer.writerow(
                    [
                        sample_id,
                        int(label) if binary else label,
                        int(prediction) if binary else prediction,
                    ]
                )

    def test_regression_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a_path = root / "a.csv"
            b_path = root / "b.csv"
            labels = [0.0, 1.0, 2.0, 3.0]
            self._write_csv(a_path, labels, labels)
            self._write_csv(b_path, labels, [1.0, 2.0, 3.0, 4.0])
            a = load_prediction_csv(a_path, "regression")
            b = load_prediction_csv(b_path, "regression")
            first = paired_bootstrap(
                a, b, metric="mae", n_bootstrap=200, seed=128, confidence=0.95
            )
            second = paired_bootstrap(
                a, b, metric="mae", n_bootstrap=200, seed=128, confidence=0.95
            )
            self.assertEqual(first, second)
            self.assertAlmostEqual(first["difference_a_minus_b"], -1.0)
            self.assertAlmostEqual(first["ci_lower"], -1.0)
            self.assertAlmostEqual(first["ci_upper"], -1.0)
            self.assertLess(first["p_value_two_sided"], 0.02)

    def test_binary_and_strict_label_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a_path = root / "a.csv"
            b_path = root / "b.csv"
            labels = [0, 1, 1, 0]
            self._write_csv(a_path, labels, labels, binary=True)
            self._write_csv(b_path, labels, [0, 0, 0, 0], binary=True)
            a = load_prediction_csv(a_path, "binary")
            b = load_prediction_csv(b_path, "binary")
            result = paired_bootstrap(
                a,
                b,
                metric="accuracy",
                n_bootstrap=200,
                seed=128,
                confidence=0.95,
            )
            self.assertAlmostEqual(result["estimate_a"], 1.0)
            self.assertAlmostEqual(result["estimate_b"], 0.5)

            mismatch_path = root / "mismatch.csv"
            self._write_csv(
                mismatch_path, [0, 0, 1, 0], [0, 0, 1, 0], binary=True
            )
            mismatch = load_prediction_csv(mismatch_path, "binary")
            with self.assertRaisesRegex(ValueError, "label order/value mismatch"):
                validate_pair(a, mismatch)

    def test_external_baseline_without_predictions_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "external_baseline_predictions.csv"
            with self.assertRaisesRegex(
                FileNotFoundError, "external baseline.*cannot be tested"
            ):
                load_prediction_csv(missing, "regression")


if __name__ == "__main__":
    unittest.main()
