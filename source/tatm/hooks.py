"""TransformerLens hook operations for TATM experiments.

Two core operations used across F1 / F2 / F3 diagnostics:

1. extract_attention_to_positions — cache a forward pass and return
   the attention-weight matrix from specified source positions to the
   last (prediction) token.  Shape: [L, H].

2. attention_knockout — hook into the attention computation, zero out
   attention from the prediction position to specified source positions,
   and return the resulting logit changes for candidate answers.
"""
from __future__ import annotations

from functools import partial
from typing import Optional

import torch
from transformer_lens import HookedTransformer


# ── Attention extraction ─────────────────────────────────────────────────────

def extract_attention_to_positions(
    model: HookedTransformer,
    tokens: torch.Tensor,
    src_positions: list[int],
    *,
    dest_position: int = -1,
) -> torch.Tensor:
    """Extract attention weights from *src_positions* to *dest_position*.

    Uses run_with_hooks (not run_with_cache) so that attention patterns are
    extracted and immediately moved to CPU during the forward pass rather than
    being accumulated in GPU VRAM.  This avoids OOM on large models (phi-3,
    3.6 B params already occupy ~7 GB on a T4).

    Parameters
    ----------
    model : HookedTransformer
    tokens : [1, seq_len] token IDs
    src_positions : token indices to measure attention FROM
    dest_position : token index that attends (default: last token)

    Returns
    -------
    Tensor of shape [n_layers, n_heads].
    For each (layer, head), the value is the **max** attention weight
    across all src_positions (following the SAT Probe convention of
    taking max over constraint tokens).
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads

    if not src_positions:
        return torch.zeros(n_layers, n_heads)

    seq_len  = tokens.shape[-1]
    dest_idx = dest_position if dest_position >= 0 else seq_len + dest_position

    # Accumulate per-layer results on CPU; never keep the full pattern on GPU.
    result = torch.zeros(n_layers, n_heads)

    def make_hook(layer: int):
        def _hook(pattern: torch.Tensor, hook) -> torch.Tensor:
            # pattern: [batch, n_heads, seq_q, seq_k]
            attn = pattern[0, :, dest_idx, src_positions]   # [H, n_srcs]
            result[layer] = attn.max(dim=-1).values.float().cpu()
            return pattern   # return unchanged so forward pass continues
        return _hook

    fwd_hooks = [
        (f"blocks.{l}.attn.hook_pattern", make_hook(l))
        for l in range(n_layers)
    ]

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, prepend_bos=False)

    return result


# ── Attention knockout ───────────────────────────────────────────────────────

def _knockout_hook(
    attn_pattern: torch.Tensor,
    hook,
    *,
    src_positions: list[int],
    dest_idx: int,
) -> torch.Tensor:
    """Post-softmax hook: zero out attention from dest to src positions,
    then renormalize the remaining weights so they sum to 1."""
    modified = attn_pattern.clone()
    modified[:, :, dest_idx, src_positions] = 0.0
    row_sum = modified[:, :, dest_idx, :].sum(dim=-1, keepdim=True).clamp(min=1e-12)
    modified[:, :, dest_idx, :] /= row_sum
    return modified


def attention_knockout(
    model: HookedTransformer,
    tokens: torch.Tensor,
    src_positions: list[int],
    knockout_layers: Optional[list[int]] = None,
    *,
    answer_token_ids: Optional[list[int]] = None,
) -> dict:
    """Run a forward pass with attention to *src_positions* knocked out.

    Parameters
    ----------
    model : HookedTransformer
    tokens : [1, seq_len]
    src_positions : year-token positions to block attention to
    knockout_layers : which layers to apply the knockout (default: all)
    answer_token_ids : token IDs whose logit change we track.
        If None, returns full logit vector change.

    Returns
    -------
    dict with keys:
        logits_clean  : logits at last position from the clean run
        logits_ko     : logits at last position from the knockout run
        delta_logits  : logits_ko - logits_clean
        delta_probs   : prob change for answer_token_ids (if given)
    """
    if knockout_layers is None:
        knockout_layers = list(range(model.cfg.n_layers))

    seq_len = tokens.shape[-1]
    dest_idx = seq_len - 1

    # clean forward pass
    torch.cuda.empty_cache()
    with torch.no_grad():
        logits_clean = model(tokens, prepend_bos=False)[0, -1].float().cpu()

    # knockout forward pass
    hooks = [
        (
            f"blocks.{layer}.attn.hook_pattern",
            partial(_knockout_hook, src_positions=src_positions, dest_idx=dest_idx),
        )
        for layer in knockout_layers
    ]
    with torch.no_grad():
        logits_ko = model.run_with_hooks(
            tokens, fwd_hooks=hooks, prepend_bos=False
        )[0, -1].float().cpu()

    delta = logits_ko - logits_clean
    result = {
        "logits_clean": logits_clean,
        "logits_ko": logits_ko,
        "delta_logits": delta,
    }

    if answer_token_ids is not None:
        probs_clean = torch.softmax(logits_clean, dim=-1)
        probs_ko = torch.softmax(logits_ko, dim=-1)
        result["delta_probs"] = {
            tid: (probs_ko[tid] - probs_clean[tid]).item()
            for tid in answer_token_ids
        }
        result["probs_clean"] = {tid: probs_clean[tid].item() for tid in answer_token_ids}
        result["probs_ko"] = {tid: probs_ko[tid].item() for tid in answer_token_ids}

    return result
