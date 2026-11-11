#!/usr/bin/env python3
"""Evaluate confidence interventions over a complete MOSEI or SIMSv2 split.

The trained checkpoint and bottleneck representations remain fixed.  The five
supported modes only alter every confidence tensor consumed by InfoGate:

* ``original``: learned VTB confidence;
* ``modality``: one masked mean per sample/stream, broadcast to valid tokens;
* ``uniform``: exactly ``1 / L`` on each valid token and bottleneck dimension;
* ``shuffle``: deterministic within-sample permutation of valid token vectors;
* ``inverse``: ``1 - confidence`` on valid tokens.

Outputs are a long-form per-sample CSV plus split-level metrics in CSV and JSON.
Routing weights must be invariant across modes because DPR consumes bottleneck
representations rather than confidence; the script verifies this property.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
from types import MethodType, SimpleNamespace
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DebertaV2Tokenizer

_QE_DIR = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_QE_DIR)
if _MY not in sys.path:
    sys.path.insert(0, _MY)
if _QE_DIR not in sys.path:
    sys.path.insert(0, _QE_DIR)

import global_configs  # noqa: E402
from deberta_infogate import InfoGate_DeBertaForSequenceClassification  # noqa: E402
from global_configs import DEVICE  # noqa: E402
from simsv2_metrics import compute_simsv2_kuda_metrics  # noqa: E402
from dump_dpr_primary_on_split import (  # noqa: E402
    _apply_ckpt_arch,
    _weights_and_primary_from_mselector_output,
    build_tensor_dataset,
    convert_to_features,
)
from simsv2_qual_utils import (  # noqa: E402
    apply_ckpt_arch_simsv2,
    build_simsv2_loader,
    default_simsv2_cli,
    forward_simsv2,
    load_simsv2_model,
    simsv2_infogate,
    unpack_simsv2_batch,
)

MODES = ("original", "modality", "uniform", "shuffle", "inverse")
PRIMARY_NAMES = {0: "acoustic", 1: "language", 2: "visual"}


def _valid_mask(conf: torch.Tensor, tok_mask: torch.Tensor | None) -> torch.Tensor:
    if tok_mask is None:
        return torch.ones(
            conf.size(0), conf.size(1), 1,
            dtype=conf.dtype, device=conf.device,
        )
    return tok_mask.to(dtype=conf.dtype).unsqueeze(-1)


def transform_confidence(
    conf: torch.Tensor,
    tok_mask: torch.Tensor | None,
    mode: str,
    *,
    seed: int,
    stream_index: int,
    sample_offset: int,
) -> torch.Tensor:
    """Apply one intervention without changing valid-token alignment."""
    if mode == "original":
        return conf

    valid = _valid_mask(conf, tok_mask)
    if mode == "modality":
        denom = (
            valid.sum(dim=(1, 2), keepdim=True) * conf.size(-1)
        ).clamp_min(1.0)
        scalar = (conf * valid).sum(dim=(1, 2), keepdim=True) / denom
        return scalar.expand_as(conf) * valid

    if mode == "uniform":
        lengths = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.ones_like(conf) * valid / lengths

    if mode == "inverse":
        return (1.0 - conf).clamp(0.0, 1.0) * valid

    if mode == "shuffle":
        out = torch.zeros_like(conf)
        for batch_index in range(conf.size(0)):
            if tok_mask is None:
                valid_idx = torch.arange(conf.size(1), device=conf.device)
            else:
                valid_idx = torch.nonzero(
                    tok_mask[batch_index].bool(), as_tuple=False
                ).flatten()
            if valid_idx.numel() <= 1:
                out[batch_index, valid_idx] = conf[batch_index, valid_idx]
                continue
            generator = torch.Generator(device=conf.device)
            generator.manual_seed(
                int(seed)
                + 1_000_003 * int(stream_index)
                + int(sample_offset)
                + int(batch_index)
            )
            order = torch.randperm(
                valid_idx.numel(), generator=generator, device=conf.device
            )
            out[batch_index, valid_idx] = conf[
                batch_index, valid_idx[order]
            ]
        return out

    raise ValueError(f"unsupported confidence intervention: {mode!r}")


def install_intervention(
    infogate_module,
    mode: str,
    seed: int,
) -> tuple[Callable[[], None], dict[str, int]]:
    """Patch the InfoGate stack so primary and auxiliary confidence both change."""
    original_forward = infogate_module.forward
    state = {"sample_offset": 0}

    def patched_forward(
        self,
        B_p,
        conf_p,
        B_aux_list,
        conf_aux_list,
        tok_mask=None,
    ):
        offset = state["sample_offset"]
        transformed_primary = transform_confidence(
            conf_p,
            tok_mask,
            mode,
            seed=seed,
            stream_index=0,
            sample_offset=offset,
        )
        transformed_aux = [
            transform_confidence(
                confidence,
                tok_mask,
                mode,
                seed=seed,
                stream_index=index + 1,
                sample_offset=offset,
            )
            for index, confidence in enumerate(conf_aux_list)
        ]
        state["sample_offset"] += int(B_p.size(0))
        return original_forward(
            B_p,
            transformed_primary,
            B_aux_list,
            transformed_aux,
            tok_mask,
        )

    infogate_module.forward = MethodType(patched_forward, infogate_module)

    def cleanup() -> None:
        infogate_module.forward = original_forward

    return cleanup, state


def _default_deberta_cli(dataset: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=os.path.join(_MY, "deberta-v3-base"),
        dataset=dataset,
        max_seq_length=50,
        unified_dim=256,
        ib_hidden_dim=256,
        bottleneck_dim=128,
        num_heads=4,
        num_infogate_layers=3,
        dropout_prob=0.1,
        beta_ib=32.0,
        alpha_ib=0.01,
        mse_weight=0.5,
        selector_target_temp=0.35,
        selector_balance_weight=0.0,
        selector_rib_weight=0.05,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.5,
        ablation="none",
        use_l_lib=True,
        use_l_rib=True,
    )


def _load_mosei(checkpoint: str, split: str, batch_size: int):
    cli = _default_deberta_cli("mosei")
    _apply_ckpt_arch(cli, checkpoint)
    cli.dataset = "mosei"
    if not getattr(cli, "ablation", None):
        cli.ablation = "none"
    global_configs.set_dataset_config("mosei")

    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        cli.model, multimodal_config=cli, num_labels=1
    )
    saved = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    data_path = os.path.join(_MY, "datasets", "mosei.pkl")
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    if split == "dev":
        split_key = "dev" if "dev" in data else "valid"
    else:
        split_key = "test"
    tokenizer = DebertaV2Tokenizer.from_pretrained(cli.model)
    features = convert_to_features(
        data[split_key], cli.max_seq_length, tokenizer, "mosei"
    )
    loader = DataLoader(
        build_tensor_dataset(features),
        batch_size=batch_size,
        shuffle=False,
    )
    return cli, model, loader, model.dberta.infogate, saved


def _load_simsv2(checkpoint: str, split: str, batch_size: int):
    cli = default_simsv2_cli()
    apply_ckpt_arch_simsv2(cli, checkpoint)
    loader, _split_key, _data_path = build_simsv2_loader(
        cli, split, batch_size
    )
    model = load_simsv2_model(cli, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return cli, model, loader, simsv2_infogate(model), saved


def _regression_metrics(dataset: str, predictions, labels) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if dataset == "simsv2":
        return {
            key: float(value)
            for key, value in compute_simsv2_kuda_metrics(
                predictions, labels
            ).items()
        }

    mae = float(np.mean(np.abs(predictions - labels)))
    corr = (
        float(np.corrcoef(predictions, labels)[0, 1])
        if predictions.size > 1
        else 0.0
    )
    nonzero = labels != 0
    binary_pred = predictions[nonzero] >= 0
    binary_true = labels[nonzero] >= 0
    acc2 = float(accuracy_score(binary_true, binary_pred))
    f1 = float(f1_score(binary_true, binary_pred, average="weighted"))
    pred7 = np.clip(np.round(predictions), -3, 3).astype(int) + 3
    true7 = np.clip(np.round(labels), -3, 3).astype(int) + 3
    acc7 = float(accuracy_score(true7, pred7))
    return {
        "mae": mae,
        "corr": corr,
        "acc2": acc2,
        "f1": f1,
        "acc7": acc7,
    }


def _run_mode(
    *,
    dataset: str,
    model,
    loader,
    infogate,
    mode: str,
    seed: int,
) -> list[dict[str, object]]:
    captured: list[tuple[torch.Tensor | None, torch.Tensor]] = []

    def selector_hook(_module, _inputs, output):
        weights, primary = _weights_and_primary_from_mselector_output(output)
        captured.append(
            (
                weights.detach().cpu() if weights is not None else None,
                primary.detach().cpu(),
            )
        )

    selector_handle = infogate.mselector.register_forward_hook(selector_hook)
    cleanup_intervention, state = install_intervention(
        infogate.infogate, mode, seed
    )
    rows: list[dict[str, object]] = []
    offset = 0
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{dataset} confidence={mode}"):
                batch = tuple(t.to(DEVICE) for t in batch)
                captured.clear()
                if dataset == "simsv2":
                    (
                        input_ids,
                        visual,
                        acoustic,
                        labels,
                        input_mask,
                        segment_ids,
                    ) = unpack_simsv2_batch(batch)
                    logits, _, _, pooled = forward_simsv2(
                        model,
                        input_ids,
                        visual,
                        acoustic,
                        labels=None,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                    )
                else:
                    input_ids, visual, acoustic, labels = batch
                    logits, _, _, pooled = model(
                        input_ids,
                        visual.squeeze(1),
                        acoustic.squeeze(1),
                        labels=None,
                        stage=2,
                    )
                if not captured:
                    raise RuntimeError("DPR hook did not fire")
                weights, primary = captured[-1]
                if weights is None or weights.size(1) != 3:
                    raise RuntimeError("expected three DPR routing weights")

                pred_values = logits.view(-1).detach().cpu()
                label_values = labels.view(-1).detach().cpu()
                pooled_norms = pooled.norm(dim=-1).detach().cpu()
                for batch_index in range(primary.numel()):
                    primary_index = int(primary[batch_index].item())
                    sample_weights = weights[batch_index]
                    prediction = float(pred_values[batch_index].item())
                    label = float(label_values[batch_index].item())
                    rows.append(
                        {
                            "dataset": dataset,
                            "mode": mode,
                            "seed": int(seed),
                            "sample_index": offset + batch_index,
                            "true_label": label,
                            "prediction": prediction,
                            "abs_error": abs(prediction - label),
                            "primary_index": primary_index,
                            "primary_name": PRIMARY_NAMES[primary_index],
                            "w_acoustic": float(sample_weights[0].item()),
                            "w_language": float(sample_weights[1].item()),
                            "w_visual": float(sample_weights[2].item()),
                            "route_entropy": float(
                                -(
                                    sample_weights.clamp_min(1e-8)
                                    * sample_weights.clamp_min(1e-8).log()
                                ).sum().item()
                            ),
                            "pooled_l2": float(
                                pooled_norms[batch_index].item()
                            ),
                        }
                    )
                offset += int(primary.numel())
    finally:
        selector_handle.remove()
        cleanup_intervention()

    if state["sample_offset"] != len(rows):
        raise RuntimeError(
            "intervention sample counter mismatch: "
            f"{state['sample_offset']} != {len(rows)}"
        )
    return rows


def _assert_aligned(
    reference: list[dict[str, object]],
    candidate: list[dict[str, object]],
    mode: str,
) -> None:
    if len(reference) != len(candidate):
        raise RuntimeError(
            f"{mode}: sample count differs from original "
            f"({len(candidate)} != {len(reference)})"
        )
    for expected, actual in zip(reference, candidate):
        if expected["sample_index"] != actual["sample_index"]:
            raise RuntimeError(f"{mode}: sample ordering changed")
        if not np.isclose(
            float(expected["true_label"]),
            float(actual["true_label"]),
            rtol=0.0,
            atol=1e-8,
        ):
            raise RuntimeError(f"{mode}: labels changed")
        if expected["primary_index"] != actual["primary_index"]:
            raise RuntimeError(f"{mode}: primary routing changed")
        for field in ("w_acoustic", "w_language", "w_visual"):
            if not np.isclose(
                float(expected[field]),
                float(actual[field]),
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError(f"{mode}: DPR weights changed ({field})")


def _write_outputs(
    output_prefix: str,
    all_rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    checkpoint: str,
    split: str,
) -> None:
    parent = os.path.dirname(os.path.abspath(output_prefix))
    os.makedirs(parent, exist_ok=True)
    prediction_path = output_prefix + "_predictions.csv"
    summary_csv_path = output_prefix + "_summary.csv"
    summary_json_path = output_prefix + "_summary.json"

    prediction_fields = [
        "dataset", "mode", "seed", "sample_index", "true_label",
        "prediction", "abs_error", "primary_index", "primary_name",
        "w_acoustic", "w_language", "w_visual", "route_entropy",
        "pooled_l2",
    ]
    with open(prediction_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    metric_fields = sorted(
        {
            key
            for row in summaries
            for key in row
            if key not in {"dataset", "mode", "n"}
        }
    )
    summary_fields = ["dataset", "mode", "n", *metric_fields]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=summary_fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "checkpoint": os.path.abspath(checkpoint),
        "split": split,
        "confidence_scope": "primary_and_auxiliary_infogate_inputs",
        "uniform_definition": "1/L on valid tokens in every bottleneck dimension",
        "summaries": summaries,
    }
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"Wrote {prediction_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset", choices=("mosei", "simsv2"), required=True
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="Comma-separated subset of: " + ",".join(MODES),
    )
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    invalid = [mode for mode in modes if mode not in MODES]
    if invalid:
        raise SystemExit(f"unsupported modes: {invalid}")
    if not modes or modes[0] != "original":
        raise SystemExit("--modes must begin with original for alignment checks")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset == "simsv2":
        _cli, model, loader, infogate, saved = _load_simsv2(
            checkpoint, args.split, args.batch_size
        )
    else:
        _cli, model, loader, infogate, saved = _load_mosei(
            checkpoint, args.split, args.batch_size
        )

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] | None = None
    for mode in modes:
        rows = _run_mode(
            dataset=args.dataset,
            model=model,
            loader=loader,
            infogate=infogate,
            mode=mode,
            seed=args.seed,
        )
        if reference_rows is None:
            reference_rows = rows
        else:
            _assert_aligned(reference_rows, rows, mode)
        metrics = _regression_metrics(
            args.dataset,
            [row["prediction"] for row in rows],
            [row["true_label"] for row in rows],
        )
        summary: dict[str, object] = {
            "dataset": args.dataset,
            "mode": mode,
            "n": len(rows),
            **metrics,
        }
        summaries.append(summary)
        all_rows.extend(rows)
        print(f"{mode}: {json.dumps(metrics, sort_keys=True)}")

    if args.split == "test" and saved.get("test_results"):
        expected = saved["test_results"]
        actual = summaries[0]
        if "mae" in expected and not np.isclose(
            float(expected["mae"]),
            float(actual["mae"]),
            rtol=0.0,
            atol=5e-4,
        ):
            raise RuntimeError(
                "original-mode MAE does not reproduce checkpoint metadata: "
                f"{actual['mae']} vs {expected['mae']}"
            )

    _write_outputs(
        args.output_prefix,
        all_rows,
        summaries,
        checkpoint,
        args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
