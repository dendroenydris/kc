"""SAT Probe for F1 diagnosis — "Time Not Set".

Adapts the SAT Probe methodology (Yuksekgonul et al., ICLR 2024) to
temporal knowledge conflicts:

  For every B1 instance, extract the attention weight from each (layer, head)
  to the year constraint tokens at the prediction position.  Train an
  L1-regularised logistic regression to predict override success/failure.
  High-weight heads that overlap with independently identified temporal heads
  confirm that F1 failures (under-attention to year tokens) cause override
  failure.

Pipeline
--------
1. collect_features  — run model, return X [N, L*H] and y [N]
2. train_probe       — fit logistic regression, return metrics
3. analyse_weights   — rank (layer, head) pairs by probe coefficient
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import gc

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformer_lens import HookedTransformer

from tatm.hooks import extract_attention_to_positions
from tatm.model import (
    build_prompt,
    check_match,
    find_year_positions,
    generate_answer,
)


@dataclass
class ProbeResult:
    """Container for SAT Probe training results."""
    auroc: float
    auroc_std: float
    coef: np.ndarray          # [L*H] — probe coefficients
    n_samples: int
    n_positive: int
    top_heads: list[tuple[int, int, float]] = field(default_factory=list)


# ── Feature collection ───────────────────────────────────────────────────────

def collect_features(
    model: HookedTransformer,
    instances: list[dict],
    *,
    template: str = "plain",
    max_new_tokens: int = 32,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run model on B1 instances; return attention features and labels.

    Parameters
    ----------
    model : HookedTransformer
    instances : list of dicts with keys
        {question, evidence_new, answer_new, answer_old, t_new, ...}
    template : prompt template name (see model.build_prompt)
    max_new_tokens : max generation length

    Returns
    -------
    X : ndarray [N, L*H]  — flattened attention features
    y : ndarray [N]        — 1 if override success, 0 if failure
    meta : list of per-instance metadata dicts
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    feat_dim = n_layers * n_heads

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta: list[dict] = []

    bar = tqdm(
        instances,
        desc="F1-a  generate+attn",
        unit="inst",
        dynamic_ncols=True,
    )

    for idx, inst in enumerate(bar):
        context = inst.get("evidence_new", inst.get("context", ""))
        question = inst.get("question", "")
        answer_new = inst.get("answer_new", "")
        t_new = inst.get("t_new")

        prompt = build_prompt(context, question, template=template)
        tokens = model.to_tokens(prompt, prepend_bos=False)
        token_ids_flat = tokens[0]

        bar.set_postfix_str("tokenising…", refresh=False)
        # year_pos: only the target year (t_new) — these are the temporal
        # constraint tokens we want the model to attend to.
        # all_year_pos: every year in the passage, kept for diagnostics only.
        year_pos = find_year_positions(token_ids_flat, model.tokenizer, target_year=t_new)
        all_year_pos = find_year_positions(token_ids_flat, model.tokenizer)

        # generation first — frees its GPU tensors before the attention pass
        bar.set_postfix_str("generating…", refresh=False)
        generated = generate_answer(model, prompt, max_new_tokens=max_new_tokens)
        success = check_match(generated, answer_new)

        # clear any CUDA fragmentation left by the generation loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # attention features: use ONLY t_new positions as the constraint source.
        # Falling back to all_year_pos only when t_new is not found at all.
        # (Using all years would dilute the signal; passages can contain
        #  hundreds of year tokens unrelated to the temporal constraint.)
        src_positions = year_pos if year_pos else all_year_pos
        bar.set_postfix_str("attn hook…", refresh=False)
        attn_vec = extract_attention_to_positions(model, tokens, src_positions)
        features = attn_vec.numpy().flatten()
        assert features.shape[0] == feat_dim

        # clear again after the attention forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        X_list.append(features)
        y_list.append(int(success))
        meta.append({
            "idx": idx,
            "instance_id": inst.get("instance_id", f"inst_{idx}"),
            "generated": generated,
            "answer_new": answer_new,
            "answer_old": inst.get("answer_old", ""),
            "success": success,
            "n_target_year_tokens": len(year_pos),
            "n_all_year_tokens": len(all_year_pos),
            "used_fallback_years": len(year_pos) == 0,
        })

        n_ok = sum(y_list)
        bar.set_postfix(
            success=f"{n_ok}/{idx+1}",
            year_toks=len(all_year_pos),
            refresh=True,
        )

    bar.close()
    return np.stack(X_list), np.array(y_list), meta


# ── Probe training ───────────────────────────────────────────────────────────

def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 10.0,
    n_folds: int = 5,
) -> ProbeResult:
    """Train L1-regularised logistic regression with stratified CV.

    Parameters
    ----------
    X : [N, L*H]
    y : [N] binary labels
    C : inverse regularisation strength.
        With standardized features and n samples (n_pos pos / n_neg neg),
        the maximum gradient at w=0 is bounded by ~0.45 regardless of
        feature values (derived from the 27/11 split in our dataset).
        For any coefficient to survive L1, we need 1/C < 0.45, i.e. C > 2.2.
        C=10.0 gives 1/C=0.1, well below the threshold, so genuine signal
        can emerge while L1 still kills pure noise features.
        (The SAT Probe paper used C=0.05 on datasets with >500 instances
        where gradients are proportionally larger.)
    n_folds : number of CV folds

    Returns
    -------
    ProbeResult with AUROC, coefficients, and top-heads ranking.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    min_class = min(n_pos, n_neg)

    # Diagnostic: confirm C value and feature statistics
    feat_std = X.std(axis=0)
    n_constant = int((feat_std < 1e-8).sum())
    print(f"  [probe] C={C}  n={len(y)} ({n_pos}+/{n_neg}-)  "
          f"feat_range=[{X.min():.3e}, {X.max():.3e}]  "
          f"constant_feats={n_constant}/{X.shape[1]}")

    if min_class < 2:
        print(
            f"WARNING: only {n_pos} positive / {n_neg} negative samples. "
            "Cannot train a meaningful probe. Returning dummy result."
        )
        return ProbeResult(
            auroc=0.5, auroc_std=0.0,
            coef=np.zeros(X.shape[1]),
            n_samples=len(y), n_positive=n_pos,
        )

    # Attention weights are tiny (typically 0.001–0.05 for a single token
    # in a ~100-token sequence).  Without scaling, L1 with C=0.05 shrinks
    # every coefficient to exactly 0 because the penalty term dominates.
    # StandardScaler (fit on train fold only, transform both) brings all
    # features to mean=0 / std=1, making the regularisation meaningful.
    def _make_pipe() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1", C=C, solver="saga",
                max_iter=2000, random_state=42,
            )),
        ])

    actual_folds = min(n_folds, min_class)
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    aurocs: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        pipe = _make_pipe()
        pipe.fit(X[train_idx], y[train_idx])
        probs = pipe.predict_proba(X[val_idx])[:, 1]
        try:
            aurocs.append(roc_auc_score(y[val_idx], probs))
        except ValueError:
            pass

    # refit on full data for coefficient analysis
    pipe_full = _make_pipe()
    pipe_full.fit(X, y)
    coef_scaled = pipe_full.named_steps["clf"].coef_[0]

    # Convert scaled coefficients back to original-space units so that
    # the magnitude reflects actual attention-weight contribution.
    scaler: StandardScaler = pipe_full.named_steps["scaler"]
    std = scaler.scale_
    std_safe = np.where(std > 0, std, 1.0)
    coef_original = coef_scaled / std_safe

    return ProbeResult(
        auroc=float(np.mean(aurocs)) if aurocs else 0.5,
        auroc_std=float(np.std(aurocs)) if aurocs else 0.0,
        coef=coef_original,
        n_samples=len(y),
        n_positive=n_pos,
    )


# ── Weight analysis ──────────────────────────────────────────────────────────

def analyse_weights(
    result: ProbeResult,
    n_layers: int,
    n_heads: int,
    *,
    top_k: int = 10,
) -> list[tuple[int, int, float]]:
    """Rank (layer, head) pairs by absolute probe coefficient.

    Returns list of (layer, head, coefficient) sorted by |coef| desc.
    """
    coef = result.coef
    entries: list[tuple[int, int, float]] = []
    for i, c in enumerate(coef):
        layer = i // n_heads
        head = i % n_heads
        entries.append((layer, head, float(c)))

    entries.sort(key=lambda x: abs(x[2]), reverse=True)
    result.top_heads = entries[:top_k]
    return entries[:top_k]
