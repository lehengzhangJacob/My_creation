"""Frozen baseline InfoGate (verbatim snapshot of ``fixed_experiment/infogate_modules.py``).

Used only when ``multimodal_config.ablation == 'none'`` so MOSI trial234 matches the frozen
fixed run bit-for-bit at the module level. Re-sync from fixed_experiment when the baseline changes.
---
InfoGate: Information Bottleneck-Guided Adaptive Cross-Attention
for Robust Multimodal Fusion

Paper terminology (PRISM): **VTB** — variational token bottleneck per modality
(implemented by ``IBEncoder`` / ``ib_enc_*``). **DPR** — differentiable primary router
(Gumbel routing on bottleneck sequences; class ``MSelector`` / attr ``mselector``).

Addresses four limitations of PCCA (MODS, AAAI 2026):
L1. Unfiltered cross-attention         -> IB bottleneck filtering
L2. Equal-weight auxiliary fusion       -> Adaptive information gates
L3. No uncertainty awareness            -> Confidence-modulated attention
L4. No cross-modal consistency          -> Cyclic IB + translation losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def masked_sequence_mean(tensor, mask=None):
    if mask is None:
        return tensor.mean(dim=1)
    mask = mask.unsqueeze(-1).type_as(tensor)
    return (tensor * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


# ============================================================
# Building Blocks (adapted from CyIN / MODS)
# ============================================================

class IBEncoder(nn.Module):
    """
    Variational token bottleneck (VTB) block: Information Bottleneck Encoder.
    F_u -> (mu, logvar) -> B via reparameterization.
    Additionally returns per-token confidence: conf = sigma(-logvar).
    """
    def __init__(self, input_dim=256, hidden_dim=256, bottleneck_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim * 2),
        )
        self.bottleneck_dim = bottleneck_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
        Returns:
            B: bottleneck latent (same shape minus last dim -> bottleneck_dim)
            mu, logvar: distribution parameters
            conf: per-element confidence in [0, 1]
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        B = self.reparameterize(mu, logvar)
        conf = torch.sigmoid(-logvar)
        return B, mu, logvar, conf


class IBDecoder(nn.Module):
    """Information Bottleneck Decoder: B -> F_reconstructed."""
    def __init__(self, bottleneck_dim=128, hidden_dim=256, output_dim=256, dropout=0.3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, B):
        return self.decoder(B)


class PositionwiseFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ============================================================
# Novel Components
# ============================================================

class IBGuidedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with IB-guided confidence modulation (core novelty).

    Two modifications over standard attention:
      1. Score bias:  scores += scale * log(conf_K + eps)
         -> uncertain key positions receive lower attention weight
      2. Value gating: V_eff = V * conf_V
         -> uncertain value positions contribute less to the output

    The confidence signal conf = sigma(-logvar) comes from the VTB (``IBEncoder``)
    variance: trained to minimize I(B; X) while maximizing I(B; Y). Tokens
    with high logvar (high uncertainty) are therefore task-irrelevant.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.conf_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value, key_confidence=None, key_mask=None):
        """
        Args:
            query:          [B, T_q, D]
            key:            [B, T_k, D]
            value:          [B, T_k, D]
            key_confidence: [B, T_k, D] from VTB (IB encoder), or None for standard attention
        """
        B = query.size(0)

        Q = self.W_q(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_confidence is not None:
            conf_pos = key_confidence.mean(dim=-1)              # [B, T_k]
            conf_bias = torch.log(conf_pos.clamp(min=1e-6))    # [B, T_k]
            scores = scores + self.conf_scale * conf_bias.unsqueeze(1).unsqueeze(2)
            V = V * conf_pos.unsqueeze(1).unsqueeze(-1)

        if key_mask is not None:
            attn_mask = key_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, -1, self.hidden_dim)
        return self.W_o(ctx)


class AdaptiveInfoGate(nn.Module):
    """
    Adaptive information gate (novel).

    Learns per-sample, per-dimension gating weights for an auxiliary
    cross-attention contribution, replacing PCCA's equal-weight sum.
        g = sigma(W_2 ReLU(W_1 [primary || ca_output]))
        output = g * ca_output
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, primary, ca_output):
        g = self.gate(torch.cat([primary, ca_output], dim=-1))
        return g * ca_output


# ============================================================
# InfoGate Layer & Module
# ============================================================

class InfoGateLayer(nn.Module):
    """
    Single InfoGate layer.

    Flow (``num_aux`` auxiliaries):
    1. Pre-LN on all streams
    2. IB-guided CA: aux_i -> primary (confidence-modulated) for each i
    3. IB-guided SA: primary self-attention
    4. Adaptive gating:  g_i * CA_ai  replaces equal-weight sum
    5. Bidirectional: primary -> aux_i for each i (skipped at the last layer)
    6. FFN + skip for all streams

    ``num_aux`` defaults to 2 (matches the 3-modality regression path). When
    set to 3 (i.e. 4 modalities with HCF), a third bank of
    ``_a3``-named sub-modules is added alongside ``_a1`` / ``_a2`` so
    state-dict keys for the original 3-modality checkpoints keep loading.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, num_aux=2):
        super().__init__()
        assert num_aux in (2, 3), f"InfoGateLayer: num_aux must be 2 or 3, got {num_aux}"
        self.num_aux = num_aux

        # aux -> primary cross-attention
        self.ca_a1_to_p = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ca_a2_to_p = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)
        if num_aux >= 3:
            self.ca_a3_to_p = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)

        # primary self-attention
        self.sa_p = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)

        # adaptive gates
        self.gate_a1 = AdaptiveInfoGate(hidden_dim)
        self.gate_a2 = AdaptiveInfoGate(hidden_dim)
        if num_aux >= 3:
            self.gate_a3 = AdaptiveInfoGate(hidden_dim)

        # primary -> aux cross-attention
        self.ca_p_to_a1 = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ca_p_to_a2 = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)
        if num_aux >= 3:
            self.ca_p_to_a3 = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)

        self.ln_p1 = nn.LayerNorm(hidden_dim)
        self.ln_p2 = nn.LayerNorm(hidden_dim)
        self.ln_a1 = nn.LayerNorm(hidden_dim)
        self.ln_a1_ff = nn.LayerNorm(hidden_dim)
        self.ln_a2 = nn.LayerNorm(hidden_dim)
        self.ln_a2_ff = nn.LayerNorm(hidden_dim)
        if num_aux >= 3:
            self.ln_a3 = nn.LayerNorm(hidden_dim)
            self.ln_a3_ff = nn.LayerNorm(hidden_dim)

        self.ffn_p = PositionwiseFFN(hidden_dim, dropout=dropout)
        self.ffn_a1 = PositionwiseFFN(hidden_dim, dropout=dropout)
        self.ffn_a2 = PositionwiseFFN(hidden_dim, dropout=dropout)
        if num_aux >= 3:
            self.ffn_a3 = PositionwiseFFN(hidden_dim, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def _aux_modules(self):
        """Return parallel lists of aux sub-modules sized by ``self.num_aux``."""
        cas = [self.ca_a1_to_p, self.ca_a2_to_p]
        gates = [self.gate_a1, self.gate_a2]
        ca_ps = [self.ca_p_to_a1, self.ca_p_to_a2]
        lns = [self.ln_a1, self.ln_a2]
        ln_ffs = [self.ln_a1_ff, self.ln_a2_ff]
        ffns = [self.ffn_a1, self.ffn_a2]
        if self.num_aux >= 3:
            cas.append(self.ca_a3_to_p)
            gates.append(self.gate_a3)
            ca_ps.append(self.ca_p_to_a3)
            lns.append(self.ln_a3)
            ln_ffs.append(self.ln_a3_ff)
            ffns.append(self.ffn_a3)
        return cas, gates, ca_ps, lns, ln_ffs, ffns

    def forward(self, B_p, conf_p, B_aux_list, conf_aux_list,
                tok_mask=None, is_last_layer=False):
        """
        Args:
            B_aux_list:    list of length ``num_aux``, each [B, T, D]
            conf_aux_list: list of length ``num_aux``, each [B, T, D]
        Returns:
            B_p_out:       [B, T, D]
            B_aux_out_list: list of length ``num_aux`` (unchanged copies if
                            ``is_last_layer`` is True; otherwise updated
                            aux streams after primary -> aux CA + FFN)
        """
        assert len(B_aux_list) == self.num_aux, \
            f"InfoGateLayer: expected {self.num_aux} aux streams, got {len(B_aux_list)}"

        cas, gates, ca_ps, lns, ln_ffs, ffns = self._aux_modules()

        B_p_n = self.ln_p1(B_p)
        B_aux_n = [ln(B) for ln, B in zip(lns, B_aux_list)]

        # IB-guided cross-attention: auxiliaries -> primary
        ca_outs = [ca(B_p_n, Bn, Bn, c, tok_mask)
                   for ca, Bn, c in zip(cas, B_aux_n, conf_aux_list)]

        # IB-guided self-attention on primary
        sa_p = self.sa_p(B_p_n, B_p_n, B_p_n, conf_p, tok_mask)
        B_p_up = B_p + self.dropout(sa_p)

        # Alignment-modulated adaptive gating (cosine sim with clamp >= 0.3)
        p_pool = masked_sequence_mean(B_p, tok_mask)
        B_p_fused = B_p_up
        for gate, aux, ca_out in zip(gates, B_aux_list, ca_outs):
            aux_pool = masked_sequence_mean(aux, tok_mask)
            # Floor 0.3: empirically chosen from experiments (not searched in Optuna).
            align = (F.cosine_similarity(p_pool, aux_pool, dim=-1).clamp(min=-1, max=1) + 1) / 2
            align = align.clamp(min=0.3).view(-1, 1, 1)
            gated = align * gate(B_p_up, ca_out)
            B_p_fused = B_p_fused + self.dropout(gated)

        B_p_out = B_p_fused + self.ffn_p(self.ln_p2(B_p_fused))

        if is_last_layer:
            return B_p_out, list(B_aux_list)

        # Bidirectional: primary -> auxiliaries
        B_p_fn = self.ln_p2(B_p_fused)
        B_aux_out_list = []
        for ca_p, ffn, ln_ff, Bn, B_orig in zip(ca_ps, ffns, ln_ffs, B_aux_n, B_aux_list):
            ca_p_aux = ca_p(Bn, B_p_fn, B_p_fn, conf_p, tok_mask)
            B_aux_out = B_orig + self.dropout(ca_p_aux)
            B_aux_out = B_aux_out + ffn(ln_ff(B_aux_out))
            B_aux_out_list.append(B_aux_out)

        return B_p_out, B_aux_out_list


class InfoGateModule(nn.Module):
    """Multi-layer stacked InfoGate cross-attention."""
    def __init__(self, hidden_dim, num_layers=3, num_heads=4, dropout=0.1, num_aux=2):
        super().__init__()
        self.num_aux = num_aux
        self.layers = nn.ModuleList([
            InfoGateLayer(hidden_dim, num_heads, dropout, num_aux=num_aux)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)

    def forward(self, B_p, conf_p, B_aux_list, conf_aux_list, tok_mask=None):
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            B_p, B_aux_list = layer(
                B_p, conf_p, B_aux_list, conf_aux_list,
                tok_mask, is_last_layer=is_last)
        return self.final_ln(B_p)


# ============================================================
# DPR — Differentiable Primary Router (class name ``MSelector``; MODS-inspired)
# ============================================================

class MSelector(nn.Module):
    """Differentiable Primary Router (DPR) with configurable routing.

    Operates on **VTB bottleneck** sequences (not raw MODS-style pooled features).
    The default ``gumbel`` mode preserves the original behavior: straight-through
    hard Gumbel-Softmax during training and deterministic argmax at inference.

    ``num_modalities`` defaults to 3 (acoustic / language / visual) for the
    regression path. When set to 4, HCF becomes a 4th candidate primary and
    a ``W_hcf`` projection plus wider MLP are added on top of the existing
    ``W_a`` / ``W_l`` / ``W_v`` layers — the 3-way state dict still loads
    cleanly for regression checkpoints.
    """
    def __init__(self, hidden_dim, num_modalities=3, gumbel_tau=1.0,
                 routing_mode='gumbel'):
        super().__init__()
        assert num_modalities in (3, 4), \
            f"MSelector: num_modalities must be 3 or 4, got {num_modalities}"
        allowed_routing_modes = {'gumbel', 'argmax', 'soft', 'batch_gumbel'}
        if routing_mode not in allowed_routing_modes:
            raise ValueError(
                f"Unknown routing_mode={routing_mode!r}; "
                f"expected one of {sorted(allowed_routing_modes)}"
            )
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.gumbel_tau = gumbel_tau          # annealed externally
        self.routing_mode = routing_mode
        self.W_a = nn.Linear(hidden_dim, 1)
        self.W_l = nn.Linear(hidden_dim, 1)
        self.W_v = nn.Linear(hidden_dim, 1)
        if num_modalities >= 4:
            self.W_hcf = nn.Linear(hidden_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
        )

    def adaptive_aggregate(self, H, W_proj, mask=None):
        scores = W_proj(H) / math.sqrt(self.hidden_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask.to(dtype=torch.bool).unsqueeze(-1), torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=1)
        return torch.bmm(attn.transpose(1, 2), H).squeeze(1)

    def _route(self, pooled_list):
        """Route pooled modalities; all return tensors retain batch dimensions."""
        logits = self.mlp(torch.cat(pooled_list, dim=-1))  # [B, num_modalities]

        if self.routing_mode == 'gumbel':
            # Keep this branch verbatim so the default path remains numerically
            # identical to the frozen implementation.
            if self.training:
                # Gumbel-Softmax: hard one-hot in forward, soft gradient in backward
                primary_onehot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
                # Soft weights use the same logits (no gumbel noise) for stable scaling
                weights = F.softmax(logits / self.gumbel_tau, dim=-1)
            else:
                weights = F.softmax(logits, dim=-1)
                primary_onehot = F.one_hot(
                    torch.argmax(weights, dim=-1),
                    num_classes=self.num_modalities).float()
        elif self.routing_mode == 'batch_gumbel':
            # The hard primary is selected from batch-mean logits and shared.
            # Per-sample soft weights stay compatible with L_rib and diagnostics.
            shared_logits = logits.mean(dim=0, keepdim=True)
            if self.training:
                shared_primary = F.gumbel_softmax(
                    shared_logits, tau=self.gumbel_tau, hard=True)
                weights = F.softmax(logits / self.gumbel_tau, dim=-1)
            else:
                weights = F.softmax(logits, dim=-1)
                shared_primary = F.one_hot(
                    torch.argmax(shared_logits, dim=-1),
                    num_classes=self.num_modalities).float()
            primary_onehot = shared_primary.expand(logits.size(0), -1)
        else:
            if self.training:
                weights = F.softmax(logits / self.gumbel_tau, dim=-1)
            else:
                weights = F.softmax(logits, dim=-1)
            if self.routing_mode == 'soft':
                # The primary is a differentiable mixture rather than a hard route.
                primary_onehot = weights
            else:  # argmax
                # one_hot(argmax(.)) is detached from logits, so the task loss
                # cannot train the hard primary decision. ``weights`` remains
                # differentiable for L_rib and auxiliary scaling.
                primary_onehot = F.one_hot(
                    torch.argmax(weights, dim=-1),
                    num_classes=self.num_modalities).float()

        primary_idx = torch.argmax(primary_onehot, dim=-1)
        return weights, primary_onehot, primary_idx

    def forward(self, H_a, H_l, H_v, mask=None):
        """3-way forward. Positional signature preserved for backward compat.

        Args:
            H_a, H_l, H_v: [B, T, D]
        Returns:
            raw features, weights (soft), primary_onehot [B, 3], primary_idx [B]
        """
        assert self.num_modalities == 3, \
            "MSelector: 4-way forward must be called via forward4(...)"
        h_a = self.adaptive_aggregate(H_a, self.W_a, mask)
        h_l = self.adaptive_aggregate(H_l, self.W_l, mask)
        h_v = self.adaptive_aggregate(H_v, self.W_v, mask)
        weights, primary_onehot, primary_idx = self._route([h_a, h_l, h_v])
        return H_a, H_l, H_v, weights, primary_onehot, primary_idx

    def forward4(self, H_a, H_l, H_v, H_hcf, mask=None):
        """4-way forward. Slot order: (a, l, v, hcf).

        Args:
            H_a, H_l, H_v, H_hcf: [B, T, D]
        Returns:
            raw features, weights (soft), primary_onehot [B, 4], primary_idx [B]
        """
        assert self.num_modalities == 4, \
            f"MSelector: forward4 requires num_modalities=4, got {self.num_modalities}"
        h_a = self.adaptive_aggregate(H_a, self.W_a, mask)
        h_l = self.adaptive_aggregate(H_l, self.W_l, mask)
        h_v = self.adaptive_aggregate(H_v, self.W_v, mask)
        h_hcf = self.adaptive_aggregate(H_hcf, self.W_hcf, mask)
        weights, primary_onehot, primary_idx = self._route([h_a, h_l, h_v, h_hcf])
        return H_a, H_l, H_v, H_hcf, weights, primary_onehot, primary_idx


# ============================================================
# Main InfoGate Module
# ============================================================

class InfoGate(nn.Module):
    """
    InfoGate: Information Bottleneck-Guided Adaptive Cross-Attention
    for Robust Multimodal Fusion.

    Pipeline:
        text / acoustic / visual
          -> Unimodal Projectors  (unified space)
          -> VTB (``IBEncoder``)  (bottleneck + per-token confidence)
          -> DPR (``MSelector``)  (sample-wise primary routing)
          -> InfoGate Module      (IB-guided cross-attention with adaptive gates)
          -> Aggregation + MLP    (sentiment prediction)

    Also computes:
        - Cyclic token-level IB loss  (L_tib)
        - Fourth return value is the primary pooled representation (for diagnostics)
    """

    def __init__(self, args):
        super().__init__()

        text_dim = args.get('text_dim', 768)
        acoustic_dim = args.get('acoustic_dim', 74)
        visual_dim = args.get('visual_dim', 47)
        hcf_dim = args.get('hcf_dim', 0)
        unified_dim = args.get('unified_dim', 256)
        ib_hidden = args.get('ib_hidden_dim', 256)
        bn_dim = args.get('bottleneck_dim', 128)
        num_heads = args.get('num_heads', 4)
        num_layers = args.get('num_infogate_layers', 3)
        dropout = args.get('dropout_prob', 0.1)

        self.beta_ib = args.get('beta_ib', 32)
        self.alpha_ib = args.get('alpha_ib', 0.01)
        self.use_l_lib = args.get('use_l_lib', True)
        self.use_l_rib = args.get('use_l_rib', True)
        self.selector_target_temp = args.get('selector_target_temp', 0.35)
        self.selector_balance_weight = args.get('selector_balance_weight', 0.0)
        self.selector_rib_weight = args.get('selector_rib_weight', 0.05)
        self.bottleneck_dim = bn_dim
        self.confidence_mode = args.get('confidence_mode', 'token')
        self.routing_mode = args.get('routing_mode', 'gumbel')
        allowed_confidence_modes = {'token', 'modality', 'uniform'}
        allowed_routing_modes = {'gumbel', 'argmax', 'soft', 'batch_gumbel'}
        if self.confidence_mode not in allowed_confidence_modes:
            raise ValueError(
                f"Unknown confidence_mode={self.confidence_mode!r}; "
                f"expected one of {sorted(allowed_confidence_modes)}"
            )
        if self.routing_mode not in allowed_routing_modes:
            raise ValueError(
                f"Unknown routing_mode={self.routing_mode!r}; "
                f"expected one of {sorted(allowed_routing_modes)}"
            )
        # 'regression' (default) or 'binary'. Switches per-modality L_lib / L_rib losses.
        self.task_type = args.get('task_type', 'regression')

        # 4-modality classification (MHD/MSD, with HCF) when hcf_dim > 0;
        # otherwise fall back to the 3-modality regression path.
        self.use_hcf = hcf_dim > 0
        # Loop-order for decoders / label preds (matches original naming: t, a, v).
        self.modalities = ('t', 'a', 'v', 'h') if self.use_hcf else ('t', 'a', 'v')
        # Slot-order for DPR (``MSelector``) weights and routing (matches original: a, l, v).
        self.modality_order = ('a', 't', 'v', 'h') if self.use_hcf else ('a', 't', 'v')
        num_modalities = len(self.modalities)
        num_aux = num_modalities - 1

        # --- 1. Unimodal projectors ---
        self.proj_t = nn.Sequential(
            nn.Linear(text_dim, unified_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_a = nn.Sequential(
            nn.LayerNorm(acoustic_dim),
            nn.Linear(acoustic_dim, unified_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_v = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, unified_dim), nn.ReLU(), nn.Dropout(dropout))
        if self.use_hcf:
            self.proj_h = nn.Sequential(
                nn.LayerNorm(hcf_dim),
                nn.Linear(hcf_dim, unified_dim), nn.ReLU(), nn.Dropout(dropout))

        # --- 2. VTB encoders (variational token bottleneck; ``IBEncoder``) ---
        self.ib_enc_t = IBEncoder(unified_dim, ib_hidden, bn_dim, dropout)
        self.ib_enc_a = IBEncoder(unified_dim, ib_hidden, bn_dim, dropout)
        self.ib_enc_v = IBEncoder(unified_dim, ib_hidden, bn_dim, dropout)
        if self.use_hcf:
            self.ib_enc_h = IBEncoder(unified_dim, ib_hidden, bn_dim, dropout)

        # --- 3. IB decoders (one per modality) for self-reconstruction IB loss ---
        self.decoders = nn.ModuleDict({
            m: IBDecoder(bn_dim, ib_hidden, unified_dim, dropout)
            for m in self.modalities
        })

        # --- 5. DPR router (``MSelector``; Gumbel on VTB sequences) ---
        self.gumbel_tau = args.get('gumbel_tau', 1.0)
        self.mselector = MSelector(bn_dim, num_modalities=num_modalities,
                                    gumbel_tau=self.gumbel_tau,
                                    routing_mode=self.routing_mode)

        # --- 6. InfoGate cross-attention ---
        self.infogate = InfoGateModule(bn_dim, num_layers, num_heads, dropout,
                                        num_aux=num_aux)

        # --- 7. Aggregation ---
        self.agg_proj = nn.Linear(bn_dim, 1)

        # --- 9. Label-level IB predictors (per-modality, for L_lib) ---
        label_preds = {
            't': nn.Linear(bn_dim, 1),
            'a': nn.Linear(bn_dim, 1),
            'v': nn.Linear(bn_dim, 1),
        }
        if self.use_hcf:
            label_preds['h'] = nn.Linear(bn_dim, 1)
        self.label_preds = nn.ModuleDict(label_preds)

        # --- 10. ITHP-style residual fusion + prediction ---
        self.primary_ln = nn.LayerNorm(bn_dim)
        self.primary_dropout = nn.Dropout(dropout)
        self.primary_classifier = nn.Linear(bn_dim, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def adaptive_aggregate(self, H, mask=None):
        scores = self.agg_proj(H) / math.sqrt(self.bottleneck_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask.to(dtype=torch.bool).unsqueeze(-1), torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=1)
        return torch.bmm(attn.transpose(1, 2), H).squeeze(1)

    @staticmethod
    def _masked_mean(tensor, mask=None):
        if tensor.dim() == 2:
            return tensor
        if mask is None:
            return tensor.mean(dim=1)
        m = mask.unsqueeze(-1).type_as(tensor)
        return (tensor * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def _apply_confidence_mode(self, confidence, mask=None):
        """Apply the configured confidence granularity to one modality."""
        if self.confidence_mode == 'token':
            # Intentional no-op: preserves the original default path exactly,
            # including its values at padded positions.
            return confidence

        if mask is None:
            valid = confidence.new_ones(confidence.shape[:2]).unsqueeze(-1)
        else:
            valid = mask.unsqueeze(-1).type_as(confidence)

        if self.confidence_mode == 'modality':
            denom = (valid.sum(dim=(1, 2), keepdim=True)
                     * confidence.size(-1)).clamp_min(1.0)
            scalar = (confidence * valid).sum(
                dim=(1, 2), keepdim=True) / denom
            return scalar.expand_as(confidence) * valid

        # uniform: every valid token receives 1/L in every bottleneck
        # dimension; invalid/padded positions remain exactly zero.
        lengths = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return confidence.new_ones(confidence.shape) * valid / lengths

    # ------------------------------------------------------------------
    # IB loss computation
    # ------------------------------------------------------------------

    def _compute_kl(self, mu, logvar, mask=None):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if mask is None or mu.dim() == 2:
            return kl.mean()
        m = mask.unsqueeze(-1).type_as(kl)
        return (kl * m).sum() / (m.sum().clamp_min(1.0) * mu.size(-1))

    def _compute_rec(self, pred, target, mask=None):
        if mask is None or pred.dim() == 2:
            return F.mse_loss(pred, target)
        diff = (pred - target).pow(2)
        m = mask.unsqueeze(-1).type_as(diff)
        return (diff * m).sum() / (m.sum().clamp_min(1.0) * pred.size(-1))

    def _token_ib(self, B_s, mu_s, lv_s, F_t, decoder, mask):
        kl = self._compute_kl(mu_s, lv_s, mask)
        rec = self._compute_rec(decoder(B_s), F_t, mask)
        return kl + self.beta_ib * rec

    def _cyclic_tib(self, F_dict, B, mu, lv, mask):
        loss = torch.tensor(0.0, device=B['t'].device)
        
        # Intra-modal (self-reconstruction) only
        for m in self.modalities:
            loss = loss + self._token_ib(
                B[m], mu[m], lv[m], F_dict[m], self.decoders[m], mask)
                
        return loss

    def _label_ib(self, B_pooled, mu_pooled, lv_pooled, labels):
        """Label-level IB: each modality's bottleneck should predict the label."""
        labels = labels.view(-1)
        total = labels.new_tensor(0.0)
        for m in self.modalities:
            kl = self._compute_kl(mu_pooled[m], lv_pooled[m])
            y_pred = self.label_preds[m](B_pooled[m]).squeeze(-1)
            if self.task_type == 'binary':
                pred_loss = F.binary_cross_entropy_with_logits(y_pred, labels.float())
            else:
                pred_loss = F.l1_loss(y_pred, labels)
            total = total + kl + self.beta_ib * pred_loss
        return total / float(len(self.modalities))

    def _routing_regularizer(self, weights, B_pooled, labels):
        """Supervise routing with per-sample modality quality instead of a text-only prior.

        Column order matches ``self.modality_order`` (a, t, v[, h]) so ``weights``
        slot k aligns with ``preds`` slot k.
        """
        labels = labels.view(-1)
        preds = torch.stack([
            self.label_preds[m](B_pooled[m]).squeeze(-1)
            for m in self.modality_order
        ], dim=1)
        if self.task_type == 'binary':
            lbl = labels.float().unsqueeze(1).expand_as(preds)
            errors = F.binary_cross_entropy_with_logits(preds, lbl, reduction='none')
            diff_thresh = 0.1
        else:
            errors = torch.abs(preds - labels.unsqueeze(1))
            diff_thresh = 0.5

        target_logits = -errors.detach() / max(self.selector_target_temp, 1e-6)
        target = F.softmax(target_logits, dim=-1)

        kl = F.kl_div(
            torch.log(weights.clamp_min(1e-8)),
            target,
            reduction='none',
        ).sum(dim=-1)

        # High divergence mask: only enforce routing if max-min error gap is large enough
        # so modalities are clearly distinguishable. Threshold differs by loss scale:
        # L1 (regression) uses 0.5; BCE (binary) uses 0.1.
        error_diff = errors.max(dim=-1)[0] - errors.min(dim=-1)[0]
        mask = (error_diff > diff_thresh).float()

        rib_kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)

        batch_usage = weights.mean(dim=0)
        usage_entropy = -(batch_usage * torch.log(batch_usage.clamp_min(1e-8))).sum()
        rib_balance = math.log(float(weights.size(1))) - usage_entropy
        return rib_kl, rib_balance, target, errors.detach(), usage_entropy.detach()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_by_primary(self, B_all_list, conf_all_list,
                          weights, primary_onehot, primary_idx):
        """Route bottlenecks by primary selection using soft mixing (Level 2).

        Args:
            B_all_list:    list of length ``N`` (one per modality, in
                           ``self.modality_order``), each [B, T, D]
            conf_all_list: list of length ``N``, each [B, T, D]
            primary_onehot: [B, N] – hard/ST routing values, or soft
                            coefficients when ``routing_mode='soft'``.

        The primary branch mixes modalities using ``primary_onehot`` (the
        backward-compatible variable name is retained for all modes).
        Auxiliary branches are the remaining ``N - 1`` streams, sorted by
        descending DPR soft routing weight.
        """
        Bs = primary_idx.size(0)
        dev = primary_idx.device
        idx = torch.arange(Bs, device=dev)
        N = len(B_all_list)

        all_B = torch.stack(B_all_list, dim=1)                 # [B, N, T, D]
        all_conf = torch.stack(conf_all_list, dim=1)           # [B, N, T, D]

        # --- Primary: mix via hard/ST or soft routing coefficients ---
        oh = primary_onehot.unsqueeze(-1).unsqueeze(-1)        # [B, N, 1, 1]
        B_p = (all_B * oh).sum(dim=1)                          # [B, T, D]
        conf_p = (all_conf * oh).sum(dim=1)                    # [B, T, D]

        # --- Auxiliaries: use hard index (no gradient needed for ordering) ---
        mask = torch.ones(Bs, N, device=dev, dtype=torch.bool)
        mask[idx, primary_idx] = False
        rem_w = weights.masked_select(mask).view(Bs, N - 1)
        rem_i = torch.arange(N, device=dev).unsqueeze(0).expand(Bs, -1)
        rem_i = rem_i.masked_select(mask).view(Bs, N - 1)
        order = rem_w.argsort(dim=1, descending=True)
        sorted_i = rem_i.gather(1, order)
        sorted_w = rem_w.gather(1, order)

        B_aux_list = []
        conf_aux_list = []
        for k in range(N - 1):
            B_aux_k = all_B[idx, sorted_i[:, k]] * sorted_w[:, k].view(-1, 1, 1)
            conf_aux_k = all_conf[idx, sorted_i[:, k]]
            B_aux_list.append(B_aux_k)
            conf_aux_list.append(conf_aux_k)
        return B_p, conf_p, B_aux_list, conf_aux_list

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, text, acoustic, visual, hcf=None, labels=None, stage=1,
                attention_mask=None):
        """
        Args:
            text:     [B, T, text_dim]   from text backbone
            acoustic: [B, T, acoustic_dim]
            visual:   [B, T, visual_dim]
            hcf:      [B, T, hcf_dim]    (required when ``self.use_hcf`` is True,
                                          ignored otherwise; default None)
            labels:   [B]  (unused here, reserved for label-level IB extension)
            stage:    1 = IB only; 2 = with routing supervision
            attention_mask: [B, T] valid-token mask
        Returns:
            logits:     [B, 1]
            ib_loss:    scalar
            loss_dict:  dict of detailed losses
            h_p: pooled primary representation before the classifier (same as adaptive_aggregate output)
        """
        Bs, T = text.size(0), text.size(1)
        device = text.device

        if attention_mask is None:
            tok_mask = torch.ones(Bs, T, device=device, dtype=text.dtype)
        else:
            tok_mask = attention_mask.float()
        zero = text.new_tensor(0.0)

        # 1. Project --------------------------------------------------
        F_t = self.proj_t(text)
        F_a = self.proj_a(acoustic)
        F_v = self.proj_v(visual)
        F_dict = {'t': F_t, 'a': F_a, 'v': F_v}
        if self.use_hcf:
            assert hcf is not None, \
                "InfoGate.forward: 4-modality mode requires `hcf` tensor [B, T, hcf_dim]"
            F_h = self.proj_h(hcf)
            F_dict['h'] = F_h

        # 2. VTB encode (per-modality IB bottleneck) -----------------
        B_t, mu_t, lv_t, conf_t = self.ib_enc_t(F_t)
        B_a, mu_a, lv_a, conf_a = self.ib_enc_a(F_a)
        B_v, mu_v, lv_v, conf_v = self.ib_enc_v(F_v)

        B = {'t': B_t, 'a': B_a, 'v': B_v}
        mu = {'t': mu_t, 'a': mu_a, 'v': mu_v}
        lv = {'t': lv_t, 'a': lv_a, 'v': lv_v}
        conf = {'t': conf_t, 'a': conf_a, 'v': conf_v}
        if self.use_hcf:
            B_h, mu_h, lv_h, conf_h = self.ib_enc_h(F_h)
            B['h'] = B_h
            mu['h'] = mu_h
            lv['h'] = lv_h
            conf['h'] = conf_h

        conf = {
            m: self._apply_confidence_mode(conf[m], tok_mask)
            for m in self.modalities
        }

        # 3. Pool for loss computation --------------------------------
        B_pooled = {m: self._masked_mean(B[m], tok_mask) for m in self.modalities}
        mu_pooled = {m: self._masked_mean(mu[m], tok_mask) for m in self.modalities}
        lv_pooled = {m: self._masked_mean(lv[m], tok_mask) for m in self.modalities}

        # 4. Cyclic token-level IB loss --------------------------------
        L_tib = self._cyclic_tib(F_dict, B, mu, lv, tok_mask)

        # 5. Label-level IB loss (task-aware bottleneck supervision) ----
        if self.use_l_lib and labels is not None:
            L_lib = self._label_ib(B_pooled, mu_pooled, lv_pooled, labels)
        else:
            L_lib = zero

        # 7. DPR routing (``self.mselector``; slot order = self.modality_order) ---
        if self.use_hcf:
            _, _, _, _, weights, primary_onehot, primary_idx = self.mselector.forward4(
                B['a'], B['t'], B['v'], B['h'], tok_mask)
        else:
            _, _, _, weights, primary_onehot, primary_idx = self.mselector(
                B['a'], B['t'], B['v'], tok_mask)

        # 8. Route by primary with confidence -------------------------
        B_all_list = [B[m] for m in self.modality_order]
        conf_all_list = [conf[m] for m in self.modality_order]
        B_p, conf_p, B_aux_list, conf_aux_list = self._route_by_primary(
            B_all_list, conf_all_list, weights, primary_onehot, primary_idx)

        # Delay routing supervision to stage 2 or only selectively
        if self.use_l_rib and labels is not None and stage == 2:
            L_rib_kl, L_rib_balance, routing_target, routing_errors, routing_entropy = \
                self._routing_regularizer(weights, B_pooled, labels)
            L_rib = L_rib_kl + self.selector_balance_weight * L_rib_balance
        else:
            L_rib = zero
            L_rib_kl = zero
            L_rib_balance = zero
            routing_target = None
            routing_errors = None
            routing_entropy = zero

        # 9. InfoGate cross-attention ---------------------------------
        B_p_enhanced = self.infogate(B_p, conf_p, B_aux_list, conf_aux_list, tok_mask)

        # 10. Primary-centric prediction --------------------------------
        h_p = self.adaptive_aggregate(B_p_enhanced, tok_mask)

        # Consistent with MODS: only use the enhanced primary modality
        logits = self.primary_classifier(self.primary_dropout(self.primary_ln(h_p)))

        # 12. Combine IB losses ----------------------------------------
        ib_loss = self.alpha_ib * (L_tib + L_lib)

        # 13. Routing Information Bottleneck (L_rib)
        # Use per-sample modality quality as the routing target and a mild
        # batch-level entropy regularizer to prevent selector collapse.
        if self.use_l_rib and labels is not None:
            ib_loss = ib_loss + self.selector_rib_weight * L_rib

        # Slot-name maps:
        #   slot_names: long-form name used for w_/target_/err_ keys
        #   primary_chars: short letter for primary_* keys
        # Backward-compat note: the original 3-modality code emitted 'primary_l'
        # for the language slot; we keep that (t -> l) so downstream printers
        # (train.py / train_classify.py) continue to find the expected keys.
        slot_names = {'a': 'acoustic', 't': 'language', 'v': 'visual', 'h': 'hcf'}
        primary_chars = {'a': 'a', 't': 'l', 'v': 'v', 'h': 'h'}
        loss_dict = {
            'L_tib': L_tib.item() if torch.is_tensor(L_tib) else L_tib,
            'L_lib': L_lib.item() if torch.is_tensor(L_lib) else L_lib,
            'L_rib': L_rib.item() if torch.is_tensor(L_rib) else L_rib,
            'L_rib_kl': L_rib_kl.item() if torch.is_tensor(L_rib_kl) else L_rib_kl,
            'L_rib_balance': L_rib_balance.item() if torch.is_tensor(L_rib_balance) else L_rib_balance,
            'fusion_conf': conf_p.mean().item(),
            'routing_entropy': routing_entropy.item() if torch.is_tensor(routing_entropy) else routing_entropy,
        }
        # Per-slot weight / primary / confidence diagnostics (N-modality).
        for i, m in enumerate(self.modality_order):
            loss_dict[f'w_{slot_names[m]}'] = weights[:, i].mean().item()
            loss_dict[f'primary_{primary_chars[m]}'] = (primary_idx == i).float().mean().item()
        for m in self.modalities:
            loss_dict[f'conf_{m}'] = conf[m].mean().item()

        if routing_target is not None:
            for i, m in enumerate(self.modality_order):
                loss_dict[f'target_{slot_names[m]}'] = routing_target[:, i].mean().item()
                loss_dict[f'err_{slot_names[m]}'] = routing_errors[:, i].mean().item()

        return logits, ib_loss, loss_dict, h_p
