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

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
        # each instance runs 2 forward passes (generate + run_with_cache)
        # show running success rate as postfix
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
        year_pos = find_year_positions(token_ids_flat, model.tokenizer, target_year=t_new)
        all_year_pos = find_year_positions(token_ids_flat, model.tokenizer)

        # attention features (max over year positions)
        bar.set_postfix_str("attn cache…", refresh=False)
        attn_vec = extract_attention_to_positions(model, tokens, all_year_pos)
        features = attn_vec.numpy().flatten()
        assert features.shape[0] == feat_dim

        # model generation → label
        bar.set_postfix_str("generating…", refresh=False)
        generated = generate_answer(model, prompt, max_new_tokens=max_new_tokens)
        success = check_match(generated, answer_new)

        X_list.append(features)
        y_list.append(int(success))
        meta.append({
            "idx": idx,
            "instance_id": inst.get("instance_id", f"inst_{idx}"),
            "generated": generated,
            "answer_new": answer_new,
            "answer_old": inst.get("answer_old", ""),
            "success": success,
            "n_year_tokens": len(all_year_pos),
            "n_target_year_tokens": len(year_pos),
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
    C: float = 0.05,
    n_folds: int = 5,
) -> ProbeResult:
    """Train L1-regularised logistic regression with stratified CV.

    Parameters
    ----------
    X : [N, L*H]
    y : [N] binary labels
    C : inverse regularisation strength (lower = more L1 sparsity)
    n_folds : number of CV folds

    Returns
    -------
    ProbeResult with AUROC, coefficients, and top-heads ranking.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    min_class = min(n_pos, n_neg)

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

    actual_folds = min(n_folds, min_class)
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    aurocs: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(
            penalty="l1", C=C, solver="saga", max_iter=2000, random_state=42,
        )
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        try:
            aurocs.append(roc_auc_score(y[val_idx], probs))
        except ValueError:
            pass

    # refit on full data for coefficient analysis
    clf_full = LogisticRegression(
        penalty="l1", C=C, solver="saga", max_iter=2000, random_state=42,
    )
    clf_full.fit(X, y)

    return ProbeResult(
        auroc=float(np.mean(aurocs)) if aurocs else 0.5,
        auroc_std=float(np.std(aurocs)) if aurocs else 0.0,
        coef=clf_full.coef_[0],
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
