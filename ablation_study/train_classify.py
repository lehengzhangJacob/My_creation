"""InfoGate binary classification (UR-FUNNY / MUStARD) with PRISM ``--ablation``.

Mirrors ``My_creation/train_classify.py``; uses ``ablation_study/albert_infogate.py``
and local ``infogate_modules``. Data loaders load from parent ``My_creation/``.
"""

import argparse
import csv
import os
import sys
import random
from collections import deque

import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup, AlbertTokenizer
from torch.optim import AdamW

from albert_infogate import InfoGate_AlbertForSequenceClassification
import global_configs
from global_configs import DEVICE

_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
_MY_CREATION_DIR = os.path.dirname(_ABL_DIR)
if _MY_CREATION_DIR not in sys.path:
    sys.path.insert(0, _MY_CREATION_DIR)
from data_humor import build_humor_loaders
from selection_utils import (
    SELECTION_METRIC_CHOICES,
    build_selection_tiebreak,
    compute_selection_score,
    selection_higher_is_better,
)

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="InfoGate Binary Classification Training")
parser.add_argument("--model", type=str,
                    default=os.path.join(_MY_CREATION_DIR, "albert-base-v2"),
                    help="Path to ALBERT weights (HKT-aligned MHD/MSD setup).")
parser.add_argument("--dataset", type=str, choices=["ur_funny", "mustard"], default="ur_funny")
parser.add_argument("--max_seq_length", type=int, default=64,
                    help="HKT default: 64 for humor, 77 for sarcasm.")
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--stage1_epochs", type=int, default=8)
parser.add_argument("--dropout_prob", type=float, default=0.25)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--ig_learning_rate", type=float, default=5e-4)
parser.add_argument("--gradient_accumulation_step", type=int, default=2)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=42)

# InfoGate-specific
parser.add_argument("--unified_dim", type=int, default=256)
parser.add_argument("--ib_hidden_dim", type=int, default=256)
parser.add_argument("--bottleneck_dim", type=int, default=128)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_infogate_layers", type=int, default=3)
parser.add_argument("--beta_ib", type=float, default=16.0)
parser.add_argument("--alpha_ib", type=float, default=0.005)
parser.add_argument("--selector_target_temp", type=float, default=0.6,
                    help="Temperature for modality-quality routing targets (BCE scale).")
parser.add_argument("--selector_balance_weight", type=float, default=0.0,
                    help="Batch-level routing entropy regularization weight.")
parser.add_argument("--selector_rib_weight", type=float, default=0.05,
                    help="Overall weight of the routing supervision loss.")
parser.add_argument(
    "--disable_hcf", "--disable-hcf", action="store_true",
    help="Disable the 4-D handcrafted feature modality.")
parser.add_argument(
    "--confidence_mode", "--confidence-mode", type=str, default="token",
    choices=("token", "modality", "uniform"),
    help="Confidence granularity used by InfoGate.")
parser.add_argument(
    "--routing_mode", "--routing-mode", type=str, default="gumbel",
    choices=("gumbel", "argmax", "soft", "batch_gumbel"),
    help="Primary-modality routing strategy.")
parser.add_argument("--disable_l_lib", action="store_true",
                    help="Ablate the label-level IB loss.")
parser.add_argument("--disable_l_rib", action="store_true",
                    help="Ablate the routing IB prior loss.")

parser.add_argument("--ema_decay", type=float, default=0.999)
parser.add_argument("--ema_start_epoch", type=int, default=5)

parser.add_argument("--gumbel_tau_start", type=float, default=1.0)
parser.add_argument("--gumbel_tau_end", type=float, default=0.5)

parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--selection_metric", type=str,
                    default="binary_acc",
                    choices=SELECTION_METRIC_CHOICES)

# Training dynamics (defaults preserve legacy behaviour)
parser.add_argument(
    "--ib_loss_mult_end", type=float, default=1.0,
    help="Linear IB multiplier at last epoch (stage2 only); 1.0 disables decay. "
         "Stage2 ramps from 1.0 at first stage-2 epoch to this value.")
parser.add_argument(
    "--freeze_backbone_stage2", action="store_true",
    help="Freeze ALBERT encoder weights (albert.model.*) during stage 2.")
parser.add_argument(
    "--bce_pos_weight_mode", type=str, default="none",
    choices=("none", "auto"),
    help="Training-only class balance for BCE: auto sets pos_weight=n_neg/n_pos.")
parser.add_argument(
    "--bce_pos_weight", type=float, default=None,
    help="Manual positive-class weight for BCE (overrides --bce_pos_weight_mode).")
parser.add_argument(
    "--focal_gamma", type=float, default=0.0,
    help="Focal loss gamma on task BCE (0 = standard BCE). Training only.")
parser.add_argument(
    "--rdrop_alpha", type=float, default=0.0,
    help="Symmetric KL consistency weight between two dropout samples (0 = off).")
parser.add_argument(
    "--early_stop_patience", type=int, default=0,
    help="Stop if dev selection score (after smoothing) does not improve for "
         "this many epochs past stage1; 0 disables.")
parser.add_argument(
    "--selection_smooth_window", type=int, default=1,
    help="Use max dev selection score over the last W epochs when saving / "
         "early stopping (W=1 = no smoothing).")
# Paper mapping: ``no_mselector`` -> w/o DPR; ``no_ib`` -> w/o VTB (identifiers kept for checkpoints/CLI).
ABLATION_CHOICES = (
    "none",
    "no_infogate",
    "no_mselector",
    "no_ib",
    "no_ib_no_mselector_no_infogate",
    "no_conf",
    "no_conf_gating",
    "no_adaptive_gate",
)
parser.add_argument(
    "--ablation",
    type=str,
    default="none",
    choices=ABLATION_CHOICES,
    help=(
        "PRISM ablation: ``no_mselector``=w/o DPR, ``no_ib``=w/o VTB, ``no_infogate``=w/o InfoGate; "
        "``no_conf`` aliases ``no_conf_gating``; "
        "``no_ib_no_mselector_no_infogate``=w/o all three (frozen recipe otherwise)."
    ),
)

args = parser.parse_args()

# Hard-code task type for this entry point.
args.task_type = "binary"
args.use_l_lib = not args.disable_l_lib
args.use_l_rib = not args.disable_l_rib
# Field expected by save dict / scoring helpers (regression placeholder).
args.mse_weight = 0.0

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
VISUAL_DIM = global_configs.VISUAL_DIM
TEXT_DIM = global_configs.TEXT_DIM
effective_hcf_dim = 0 if args.disable_hcf else 4
# ``multimodal_config`` is the parsed args namespace, so expose the effective
# dimension explicitly. In particular, keep 0 rather than falling back to the
# dataset-level HCF dimension in the ALBERT wrapper.
args.hcf_dim = effective_hcf_dim


# ============================================================
# Data
# ============================================================

def setup_data():
    tokenizer = AlbertTokenizer.from_pretrained(args.model)
    return build_humor_loaders(
        dataset=args.dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        acoustic_dim=ACOUSTIC_DIM,
        visual_dim=VISUAL_DIM,
        train_batch_size=args.train_batch_size,
        dev_batch_size=args.dev_batch_size,
        test_batch_size=args.test_batch_size,
        gradient_accumulation_step=args.gradient_accumulation_step,
        n_epochs=args.n_epochs,
        hcf_dim=effective_hcf_dim,
        slice_hkt=True,
    )


# ============================================================
# Seed & model setup
# ============================================================

def set_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(n_opt):
    model = InfoGate_AlbertForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1)
    backbone_prefix = "albert.model."
    model.to(DEVICE)

    no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight"}
    ig_lr = getattr(args, 'ig_learning_rate', 5e-4)

    backbone_decay, backbone_no_decay = [], []
    ig_decay, ig_no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_nd = any(nd in n for nd in no_decay)
        if n.startswith(backbone_prefix):
            (backbone_no_decay if is_nd else backbone_decay).append(p)
        else:
            (ig_no_decay if is_nd else ig_decay).append(p)

    groups = [
        {"params": backbone_decay,    "lr": args.learning_rate, "weight_decay": args.weight_decay},
        {"params": backbone_no_decay, "lr": args.learning_rate, "weight_decay": 0.0},
        {"params": ig_decay,          "lr": ig_lr,              "weight_decay": args.weight_decay},
        {"params": ig_no_decay,       "lr": ig_lr,              "weight_decay": 0.0},
    ]
    optimizer = AdamW(groups, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * n_opt),
        num_training_steps=max(n_opt, 1),
    )
    return model, optimizer, scheduler


# ============================================================
# EMA (identical to train.py)
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad and n in self.shadow}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict(self):
        return {n: v.clone() for n, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for n, v in state_dict.items():
            if n in self.shadow:
                self.shadow[n].copy_(v)

    def reset(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].copy_(p.data)


def toggle_state(enabled):
    return "on" if enabled else "off"


# ============================================================
# Train / Eval / Test
# ============================================================

def _unpack_batch(batch, use_hcf):
    """Unpack a humor-dataset batch. Layout depends on ``use_hcf``:
        use_hcf=False:  (input_ids, visual, acoustic, label)
        use_hcf=True:   (input_ids, visual, acoustic, hcf, label)
    Returns: input_ids, visual, acoustic, hcf (or None), label_ids
    """
    if use_hcf:
        input_ids, visual, acoustic, hcf, label_ids = batch
        visual = visual.squeeze(1)
        acoustic = acoustic.squeeze(1)
        hcf = hcf.squeeze(1)
        return input_ids, visual, acoustic, hcf, label_ids
    input_ids, visual, acoustic, label_ids = batch
    visual = visual.squeeze(1)
    acoustic = acoustic.squeeze(1)
    return input_ids, visual, acoustic, None, label_ids


BACKBONE_PREFIX = "albert.model."


def set_backbone_requires_grad(model, requires_grad):
    m = model.module if hasattr(model, "module") else model
    for n, p in m.named_parameters():
        if n.startswith(BACKBONE_PREFIX):
            p.requires_grad = requires_grad


def ib_loss_multiplier(epoch, stage1_epochs, n_epochs, mult_end):
    """Stage 1: 1.0. Stage 2: linear from 1.0 (first stage-2 epoch) to mult_end."""
    if mult_end >= 1.0 - 1e-9:
        return 1.0
    if epoch < stage1_epochs:
        return 1.0
    span = max(n_epochs - stage1_epochs, 1)
    t = (epoch - stage1_epochs) / float(span)
    return 1.0 + (mult_end - 1.0) * t


def estimate_bce_pos_weight(train_dl):
    """Return scalar tensor for class-1 over-representation (n_neg / n_pos)."""
    ds = train_dl.dataset
    if not hasattr(ds, "tensors"):
        return None
    labels = ds.tensors[-1].float().view(-1)
    n_pos = float((labels > 0.5).sum().item())
    n_neg = float((labels <= 0.5).sum().item())
    if n_pos < 1.0 or n_neg < 1.0:
        return None
    w = n_neg / max(n_pos, 1.0)
    return torch.tensor([w], dtype=torch.float32)


def resolve_pos_weight_tensor(mode, manual, train_dl):
    if manual is not None and manual > 0:
        return torch.tensor([manual], dtype=torch.float32, device=DEVICE)
    if mode == "auto":
        t = estimate_bce_pos_weight(train_dl)
        if t is None:
            return None
        return t.to(DEVICE)
    return None


def compute_task_loss(logits, label_ids, pos_weight_1d=None):
    """BCE (+ optional focal); eval path uses plain BCE in test_epoch."""
    logits_v = logits.view(-1)
    labels_f = label_ids.view(-1).float()
    pw = None
    if pos_weight_1d is not None:
        pw = pos_weight_1d.to(device=logits_v.device, dtype=logits_v.dtype)

    if args.focal_gamma and args.focal_gamma > 0:
        ce = F.binary_cross_entropy_with_logits(
            logits_v, labels_f, reduction="none", pos_weight=pw)
        pt = torch.where(
            labels_f > 0.5,
            torch.sigmoid(logits_v),
            torch.sigmoid(-logits_v)).clamp(min=1e-8)
        focal = (1.0 - pt).pow(args.focal_gamma) * ce
        return focal.mean()

    return F.binary_cross_entropy_with_logits(
        logits_v, labels_f, pos_weight=pw)


def symmetric_kl_binary_logits(logits_a, logits_b):
    """Symmetric KL between Bernoullis parameterized by logits (R-Drop)."""
    # pylint: disable=invalid-name
    p_a = torch.sigmoid(logits_a.view(-1)).clamp(min=1e-8, max=1 - 1e-8)
    p_b = torch.sigmoid(logits_b.view(-1)).clamp(min=1e-8, max=1 - 1e-8)
    kl_ab = p_a * (p_a.log() - p_b.log()) + (1 - p_a) * ((1 - p_a).log() - (1 - p_b).log())
    kl_ba = p_b * (p_b.log() - p_a.log()) + (1 - p_b) * ((1 - p_b).log() - (1 - p_a).log())
    return 0.5 * (kl_ab.mean() + kl_ba.mean())


def train_epoch(model, loader, optimizer, scheduler, stage, ib_mult,
                pos_weight_1d, ema=None):
    model.train()
    total_loss, steps = 0.0, 0
    sum_task, sum_ib = 0.0, 0.0
    sum_detail = {}
    use_hcf = effective_hcf_dim > 0
    use_rdrop = getattr(args, "rdrop_alpha", 0.0) and args.rdrop_alpha > 0

    train_pbar = tqdm(loader, desc=f"Train (stage {stage})")
    for step, batch in enumerate(train_pbar):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, hcf, label_ids = _unpack_batch(batch, use_hcf)

        if use_rdrop:
            logits1, ib1, loss_dict, _ = model(
                input_ids, visual, acoustic, hcf=hcf, labels=label_ids,
                stage=stage)
            logits2, ib2, _, _ = model(
                input_ids, visual, acoustic, hcf=hcf, labels=label_ids,
                stage=stage)
            pred_flat = logits1.view(-1)
            loss_dict = dict(loss_dict)
            loss_dict['pred_mean'] = pred_flat.mean().item()
            loss_dict['pred_std'] = pred_flat.std(unbiased=False).item()
            l_task = 0.5 * (
                compute_task_loss(logits1, label_ids, pos_weight_1d)
                + compute_task_loss(logits2, label_ids, pos_weight_1d))
            ib_av = 0.5 * (ib1 + ib2)
            rdrop_term = symmetric_kl_binary_logits(logits1, logits2)
            loss = l_task + ib_mult * ib_av + args.rdrop_alpha * rdrop_term
            ib_loss = ib_av
        else:
            logits, ib_loss, loss_dict, _ = model(
                input_ids, visual, acoustic, hcf=hcf, labels=label_ids,
                stage=stage)
            pred_flat = logits.view(-1)
            loss_dict['pred_mean'] = pred_flat.mean().item()
            loss_dict['pred_std'] = pred_flat.std(unbiased=False).item()
            l_task = compute_task_loss(logits, label_ids, pos_weight_1d)
            loss = l_task + ib_mult * ib_loss

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()
        total_loss += loss.item()
        sum_task += l_task.item()
        sum_ib += ib_loss.item()
        for k, v in loss_dict.items():
            sum_detail[k] = sum_detail.get(k, 0.0) + v
        steps += 1

        ib_show = ib_loss.item()
        train_pbar.set_postfix({"task": f"{l_task.item():.3f}",
                                "ib": f"{ib_show:.3f}",
                                "p_std": f"{loss_dict['pred_std']:.2f}"})

        if (step + 1) % args.gradient_accumulation_step == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

    n = max(steps, 1)
    detail = {k: v / n for k, v in sum_detail.items()}
    return total_loss / n, sum_task / n, sum_ib / n, detail


def test_epoch(model, loader, stage=2, desc="Test"):
    model.eval()
    preds, labels = [], []
    use_hcf = effective_hcf_dim > 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, hcf, label_ids = _unpack_batch(batch, use_hcf)

            logits, _, _, _ = model(input_ids, visual, acoustic, hcf=hcf, stage=stage)
            preds.extend(logits.view(-1).cpu().numpy().tolist())
            labels.extend(label_ids.view(-1).cpu().numpy().tolist())

    return np.array(preds).flatten(), np.array(labels).flatten()


def score(preds, y):
    """Binary classification metrics. Returns a dict with the field names that
    `fmt_metrics`, `selection_kwargs`, and `compute_selection_score` expect.
    Regression-only fields (mae/corr/acc7) are present as placeholders so the
    existing helpers keep working without branching.
    """
    preds = np.asarray(preds).flatten()
    y = np.asarray(y).flatten()
    prob = 1.0 / (1.0 + np.exp(-preds))
    yhat = (prob >= 0.5).astype(int)
    ytrue = y.astype(int)

    acc = accuracy_score(ytrue, yhat)
    f1_w = f1_score(ytrue, yhat, average="weighted") if len(ytrue) else 0.0

    return {
        "acc2": float(acc),
        "f1": float(f1_w),
        # Placeholders so legacy helpers keep working; not meaningful for binary.
        "mae": float(np.mean(np.abs(prob - ytrue))) if len(ytrue) else 0.0,
        "corr": 0.0,
        "acc7": 0.0,
    }


def fmt_metrics(m, prefix=""):
    return (f"{prefix}Acc={m['acc2']*100:.2f}%  F1={m['f1']*100:.2f}%  "
            f"BCE-prob-MAE={m['mae']:.4f}")


def selection_kwargs(m):
    return dict(acc2=m["acc2"], mae=m["mae"], corr=m["corr"], f1=m["f1"],
                acc7=m.get("acc7", 0.0))


def save_test_predictions_csv(preds, labels, output_path):
    """Write the test outputs associated with the best dev checkpoint."""
    logits = np.asarray(preds).reshape(-1)
    ytrue = np.asarray(labels).reshape(-1)
    if logits.shape != ytrue.shape:
        raise ValueError(
            f"test pred/label shape mismatch: {logits.shape} vs {ytrue.shape}")

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["sample_index", "true_label", "logit", "probability", "prediction"])
        for i, (label, logit, probability, prediction) in enumerate(
                zip(ytrue, logits, probabilities, predictions)):
            writer.writerow([
                i, int(label), float(logit), float(probability), int(prediction)])


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("InfoGate Binary Classification Training")
    print(f"  Dataset        : {args.dataset}")
    print(f"  Task type      : {args.task_type}")
    print(f"  Epochs         : {args.n_epochs} (stage1: {args.stage1_epochs})")
    print(f"  Batch size     : {args.train_batch_size}"
          f" x grad_accum {args.gradient_accumulation_step}")
    print(f"  Max seq len    : {args.max_seq_length}")
    print(f"  LR (backbone)  : {args.learning_rate}")
    print(f"  LR (InfoGate)  : {args.ig_learning_rate}")
    print(f"  InfoGate layers: {args.num_infogate_layers}")
    print(f"  Bottleneck dim : {args.bottleneck_dim}")
    print(f"  beta_ib        : {args.beta_ib}")
    print(f"  alpha_ib       : {args.alpha_ib}")
    print(f"  selector_temp  : {args.selector_target_temp}")
    print(f"  selector_bal_w : {args.selector_balance_weight}")
    print(f"  HCF modality   : {toggle_state(effective_hcf_dim > 0)}"
          f" (dim={effective_hcf_dim})")
    print(f"  Confidence mode: {args.confidence_mode}")
    print(f"  Routing mode   : {args.routing_mode}")
    print(f"  IB mult end    : {args.ib_loss_mult_end} (stage2 linear decay)")
    print(f"  Freeze BB stg2 : {args.freeze_backbone_stage2}")
    print(f"  BCE pos weight : {args.bce_pos_weight_mode}"
          f"{'' if args.bce_pos_weight is None else f' (manual {args.bce_pos_weight})'}")
    print(f"  focal_gamma    : {args.focal_gamma}")
    print(f"  rdrop_alpha    : {args.rdrop_alpha}")
    print(f"  early_stop_pat : {args.early_stop_patience}")
    print(f"  select_smooth W: {args.selection_smooth_window}")
    print(f"  Loss terms     : L_lib={toggle_state(args.use_l_lib)}"
          f"  L_rib={toggle_state(args.use_l_rib)}")
    print(f"  Ablation       : {args.ablation}")
    print("=" * 60)

    set_seed(args.seed)
    train_dl, dev_dl, test_dl, n_opt = setup_data()
    pos_weight_t = resolve_pos_weight_tensor(
        args.bce_pos_weight_mode, args.bce_pos_weight, train_dl)
    if pos_weight_t is not None:
        print(f"BCE pos_weight (training): {pos_weight_t.cpu().tolist()}")

    model, optimizer, scheduler = build_model(n_opt)
    ema = EMA(model, decay=args.ema_decay)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {train_p:,} trainable")
    print(f"EMA: decay={args.ema_decay}, start_epoch={args.ema_start_epoch}")
    print("=" * 60)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"infogate_{args.dataset}_best.pt")
    predictions_csv_path = os.path.join(
        args.checkpoint_dir, "best_test_predictions.csv")
    select_start = args.stage1_epochs
    select_higher_is_better = selection_higher_is_better(args.selection_metric)
    best_selection_score = float('-inf') if select_higher_is_better else float('inf')
    best_selection_tiebreak = None
    best_results = None
    best_test_acc = -1.0
    best_test_results = None
    last_test_results = None

    sel_hist = deque(maxlen=max(1, args.selection_smooth_window))
    epochs_without_improvement = 0
    final_epoch_reached = args.n_epochs

    for epoch in range(args.n_epochs):
        stage = 1 if epoch < args.stage1_epochs else 2
        freeze_bb = args.freeze_backbone_stage2 and stage == 2
        set_backbone_requires_grad(model, not freeze_bb)

        tau_s = getattr(args, 'gumbel_tau_start', 1.0)
        tau_e = getattr(args, 'gumbel_tau_end', 0.5)
        tau = tau_s + (tau_e - tau_s) * epoch / max(args.n_epochs - 1, 1)
        m = model.module if hasattr(model, 'module') else model
        ig = m.albert.infogate
        # DPR Gumbel temperature (``mselector``)
        ig.mselector.gumbel_tau = tau

        ib_mult = ib_loss_multiplier(
            epoch, args.stage1_epochs, args.n_epochs, args.ib_loss_mult_end)

        eval_with_ema = epoch >= args.ema_start_epoch
        if epoch == args.ema_start_epoch:
            ema.reset(model)
        tr_loss, tr_task, tr_ib, tr_detail = train_epoch(
            model, train_dl, optimizer, scheduler, stage, ib_mult,
            pos_weight_t, ema=ema if eval_with_ema else None)

        print(f"\nEpoch {epoch + 1}/{args.n_epochs}  [stage {stage}]"
              f"{'  (EMA eval)' if eval_with_ema else ''}"
              f"{'  backbone=Frozen' if freeze_bb else ''}")
        print(f"  Loss  total={tr_loss:.4f}  task={tr_task:.4f}  ib={tr_ib:.4f}"
              f"  ib_mult={ib_mult:.4f}")
        detail_str = "  ".join(f"{k}={v:.4f}" for k, v in tr_detail.items()
                                if k.startswith('L_'))
        print(f"  Detail  {detail_str}")
        diag_keys = ['w_acoustic', 'w_language', 'w_visual', 'w_hcf',
                     'target_acoustic', 'target_language', 'target_visual', 'target_hcf',
                     'primary_a', 'primary_l', 'primary_v', 'primary_h',
                     'err_acoustic', 'err_language', 'err_visual', 'err_hcf',
                     'conf_t', 'conf_a', 'conf_v', 'conf_h', 'fusion_conf',
                     'routing_entropy', 'pred_mean', 'pred_std']
        diag_vals = {k: tr_detail[k] for k in diag_keys if k in tr_detail}
        if diag_vals:
            w_parts = [f"a:{diag_vals.get('w_acoustic',0):.3f}",
                       f"l:{diag_vals.get('w_language',0):.3f}",
                       f"v:{diag_vals.get('w_visual',0):.3f}"]
            if 'w_hcf' in diag_vals:
                w_parts.append(f"h:{diag_vals['w_hcf']:.3f}")
            w_str = f"w=[{' '.join(w_parts)}]"

            tgt_parts = [f"a:{diag_vals.get('target_acoustic',0):.3f}",
                         f"l:{diag_vals.get('target_language',0):.3f}",
                         f"v:{diag_vals.get('target_visual',0):.3f}"]
            if 'target_hcf' in diag_vals:
                tgt_parts.append(f"h:{diag_vals['target_hcf']:.3f}")
            tgt_str = f"target=[{' '.join(tgt_parts)}]"

            p_parts = [f"a:{diag_vals.get('primary_a',0):.2f}",
                       f"l:{diag_vals.get('primary_l',0):.2f}",
                       f"v:{diag_vals.get('primary_v',0):.2f}"]
            if 'primary_h' in diag_vals:
                p_parts.append(f"h:{diag_vals['primary_h']:.2f}")
            p_str = f"primary=[{' '.join(p_parts)}]"

            err_parts = [f"a:{diag_vals.get('err_acoustic',0):.3f}",
                         f"l:{diag_vals.get('err_language',0):.3f}",
                         f"v:{diag_vals.get('err_visual',0):.3f}"]
            if 'err_hcf' in diag_vals:
                err_parts.append(f"h:{diag_vals['err_hcf']:.3f}")
            err_str = f"err=[{' '.join(err_parts)}]"

            c_parts = [f"t:{diag_vals.get('conf_t',0):.3f}",
                       f"a:{diag_vals.get('conf_a',0):.3f}",
                       f"v:{diag_vals.get('conf_v',0):.3f}"]
            if 'conf_h' in diag_vals:
                c_parts.append(f"h:{diag_vals['conf_h']:.3f}")
            c_parts.append(f"fused:{diag_vals.get('fusion_conf',0):.3f}")
            c_str = f"conf=[{' '.join(c_parts)}]"
            rib_str = f"route_H={diag_vals.get('routing_entropy',0):.3f}"
            pred_str = (f"pred_logit=[mean:{diag_vals.get('pred_mean',0):.3f} "
                        f"std:{diag_vals.get('pred_std',0):.3f}]")
            print(f"  Diag  {w_str}  {tgt_str}  {p_str}\n"
                  f"  Qual  {err_str}  {c_str}  {rib_str}  gumbel_tau={tau:.3f}\n"
                  f"  Stats {pred_str}")

        if eval_with_ema:
            ema.apply(model)

        dev_preds, dev_labels = test_epoch(model, dev_dl, stage=stage, desc="Dev")
        dev_m = score(dev_preds, dev_labels)
        dev_kw = selection_kwargs(dev_m)
        selection_score = compute_selection_score(args.selection_metric, **dev_kw)
        selection_tiebreak = build_selection_tiebreak(**dev_kw)
        print(f"  Dev   {fmt_metrics(dev_m)}")
        print(f"  Select {args.selection_metric}={selection_score:.6f}")

        compare_score = selection_score
        if epoch >= select_start:
            sel_hist.append(selection_score)
            compare_score = max(sel_hist)
            if args.selection_smooth_window > 1:
                print(f"  Select_smooth max@{len(sel_hist)} "
                      f"{args.selection_metric}={compare_score:.6f}")

        preds, labels = test_epoch(model, test_dl, stage=stage)
        test_m = score(preds, labels)
        print(f"  Test  {fmt_metrics(test_m)}")

        last_test_results = test_m
        final_epoch_reached = epoch + 1

        improved = False
        if epoch >= select_start:
            if best_selection_tiebreak is None:
                should_save = True
            else:
                better_score = (compare_score > best_selection_score
                                if select_higher_is_better
                                else compare_score < best_selection_score)
                same_score = abs(compare_score - best_selection_score) <= 1e-12
                should_save = better_score or (
                    same_score and selection_tiebreak > best_selection_tiebreak)

            if should_save:
                improved = True
                best_selection_score = compare_score
                best_selection_tiebreak = selection_tiebreak
                best_results = test_m
                best_test_preds = np.asarray(preds).reshape(-1).copy()
                best_test_labels = np.asarray(labels).reshape(-1).copy()
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dev_acc': dev_m["acc2"],
                    'dev_f1': dev_m["f1"],
                    'selection_metric': args.selection_metric,
                    'selection_score': selection_score,
                    'selection_compare_score': compare_score,
                    'ablation': {
                        'arch': args.ablation,
                        'use_l_lib': args.use_l_lib,
                        'use_l_rib': args.use_l_rib,
                    },
                    'test_results': best_results,
                    'test_preds': best_test_preds,
                    'test_labels': best_test_labels,
                    'args': args,
                }
                if eval_with_ema:
                    save_dict['ema_state_dict'] = ema.state_dict()
                torch.save(save_dict, ckpt_path)
                save_test_predictions_csv(
                    best_test_preds, best_test_labels, predictions_csv_path)
                print(f"  >> Best model saved ({args.selection_metric}="
                      f"{selection_score:.6f}, compare={compare_score:.6f}, "
                      f"Acc={dev_m['acc2']*100:.2f}%, F1={dev_m['f1']*100:.2f}%) "
                      f"to {ckpt_path}")
                print(f"  >> Best-test predictions saved to {predictions_csv_path}")

        if test_m["acc2"] > best_test_acc:
            best_test_acc = test_m["acc2"]
            best_test_results = {**test_m, "epoch": epoch + 1}

        if eval_with_ema:
            ema.restore(model)

        if (epoch >= select_start and args.early_stop_patience > 0
                and best_selection_tiebreak is not None):
            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print("\n  Early stopping: no improvement on smoothed dev "
                      f"{args.selection_metric} for "
                      f"{args.early_stop_patience} epochs.")
                break

    print("\n" + "=" * 60)
    print(f"Best Results ({args.selection_metric}, epoch >= {select_start + 1}):")
    if best_results:
        print(f"  Selection score (best compare): {best_selection_score:.6f}")
        print(f"  Acc: {best_results['acc2']*100:.2f}%")
        print(f"  F1:  {best_results['f1']*100:.2f}%")
    print(f"\nLast Epoch ({final_epoch_reached}) Results:")
    if last_test_results:
        print(f"  Acc: {last_test_results['acc2']*100:.2f}%")
        print(f"  F1:  {last_test_results['f1']*100:.2f}%")
    print("\nBest Test Acc (oracle, for reference only):")
    if best_test_results:
        print(f"  Epoch: {best_test_results['epoch']}")
        print(f"  Acc:   {best_test_results['acc2']*100:.2f}%")
        print(f"  F1:    {best_test_results['f1']*100:.2f}%")


if __name__ == '__main__':
    main()
