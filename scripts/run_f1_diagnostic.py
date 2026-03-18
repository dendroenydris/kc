#!/usr/bin/env python3
"""F1 Diagnostic — "Time Not Set"

Standalone script that runs the three F1 sub-experiments from the TATM
methodology:

  F1-a  SAT Probe        — logistic regression on attention-to-year features
  F1-b  Attention Compare — B1-success vs B1-failure vs B3 at temporal heads
  F1-c  Attention Knockout — causal test: block year-token attention, measure
                             probability drop for answer_new

Usage
-----
    python scripts/run_f1_diagnostic.py \\
        --data  data/processed/wikidata_layer2.jsonl \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --out   results/f1_diagnostic/

The script auto-detects the data format (EvalInstance layer2 or raw JSONL)
and constructs B1 / B3 prompt pairs accordingly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure source/ is on PYTHONPATH
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "source"))

from tatm.model import (
    build_prompt,
    check_match,
    find_year_positions,
    generate_answer,
    load_model,
)
from tatm.hooks import (
    attention_knockout,
    extract_attention_to_positions,
)
from tatm.sat_probe import (
    analyse_weights,
    collect_features,
    train_probe,
)


# ── Data loading ─────────────────────────────────────────────────────────────

def _is_layer2(record: dict) -> bool:
    return "instance_id" in record and "task_type" in record


def load_instances(path: str) -> tuple[list[dict], list[dict]]:
    """Load B1 and B3 instance pairs from a JSONL file.

    Returns (b1_instances, b3_instances).
    For layer-2 format, filters by task_type.
    For raw format, constructs B1/B3 pairs from each record.
    """
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {path}")

    if _is_layer2(records[0]):
        return _load_layer2(records)
    return _load_raw(records)


def _load_layer2(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Layer-2 EvalInstance format: separate B1 and B3 by instance_id prefix."""
    b1 = [r for r in records if r["instance_id"].startswith("B1")]
    b3 = [r for r in records if r["instance_id"].startswith("B3")]

    # align B3 to B1 by fact_id + t_old + t_new
    b3_map = {(r["fact_id"], r["t_old"], r["t_new"]): r for r in b3}
    b3_aligned = [
        b3_map.get((r["fact_id"], r["t_old"], r["t_new"]))
        for r in b1
    ]
    b3_aligned = [r for r in b3_aligned if r is not None]

    return b1, b3_aligned


def _load_raw(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Raw JSONL: each record has question, evidence_new, answer_new, etc.
    Construct B1 and B3 from each."""
    import re

    b1_list, b3_list = [], []
    for r in records:
        evidence = r.get("evidence_new", "")
        question = r.get("question", "")
        t_new = r.get("t_new")

        # parse t_new from ISO date if needed
        if isinstance(t_new, str) and "T" in t_new:
            t_new = int(t_new[:4])

        if not question:
            subj = r.get("subject_label", r.get("subject", ""))
            prop = r.get("property_label", r.get("property", ""))
            question = f"As of {t_new}, what is the {prop} of {subj}?"

        b1 = {**r, "t_new": t_new, "question": question, "evidence_new": evidence}
        b1_list.append(b1)

        # B3: strip years from evidence
        weak_evidence = re.sub(r"\b(19|20)\d{2}\b", "recently", evidence)
        b3 = {**b1, "evidence_new": weak_evidence, "instance_id": f"B3_{r.get('id', '')}"}
        b3_list.append(b3)

    return b1_list, b3_list


# ── A1: Parametric memory filter ─────────────────────────────────────────────

def run_a1_filter(
    model,
    b1_instances: list[dict],
    b3_instances: list[dict],
    template: str,
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Run A1 (explicit year, no context) to keep only instances the model
    cannot answer from parametric memory alone.

    Filtering criterion
    -------------------
    A1_KNOWS_NEW  : A1 outputs answer_new  → exclude (year cue alone is enough)
    A1_WRONG      : A1 outputs other       → keep (evidence is genuinely needed)
    """
    import gc

    print("\n" + "=" * 60)
    print("A1: Year-conditioned Parametric Memory Profiling")
    print("=" * 60)
    print("Prompt: question WITH year, WITHOUT context")
    print("Keeping instances where model does NOT output answer_new")
    print("(i.e., evidence passage is genuinely needed for B1 to succeed)\n")

    knows_new_ids: set[str] = set()
    a1_log = []

    bar = tqdm(b1_instances, desc="A1  parametric", unit="inst", dynamic_ncols=True)
    for inst in bar:
        question   = inst.get("question", "")
        answer_new = inst.get("answer_new", "")
        iid        = inst.get("instance_id", "")

        # A1: question with year cue intact, but NO context
        prompt    = build_prompt("", question, template=template)
        generated = generate_answer(model, prompt)
        already_knows = check_match(generated, answer_new)

        a1_log.append({
            "instance_id": iid,
            "question":    question,
            "generated":   generated,
            "answer_new":  answer_new,
            "knows_new":   already_knows,
        })

        if already_knows:
            knows_new_ids.add(iid)

        bar.set_postfix(
            knows_new=len(knows_new_ids),
            kept=len(b1_instances) - len(knows_new_ids),
            refresh=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    n_total   = len(b1_instances)
    n_known   = len(knows_new_ids)
    n_kept    = n_total - n_known
    print(f"\n  A1_KNOWS_NEW (excluded): {n_known}/{n_total} ({100*n_known/n_total:.1f}%)")
    print(f"  A1_WRONG     (kept):     {n_kept}/{n_total} ({100*n_kept/n_total:.1f}%)")

    # save A1 log
    with open(out_dir / "a1_parametric_memory.json", "w") as f:
        json.dump({"n_total": n_total, "n_knows_new": n_known, "log": a1_log},
                  f, indent=2, ensure_ascii=False)

    # filter both B1 and B3 to matching subset
    b1_filtered = [r for r in b1_instances if r.get("instance_id") not in knows_new_ids]

    # B3 has different instance_id hashes; match via (fact_id, t_old, t_new)
    excluded_keys = set()
    for r in b1_instances:
        if r.get("instance_id") in knows_new_ids:
            excluded_keys.add((r.get("fact_id"), r.get("t_old"), r.get("t_new")))
    b3_filtered = [
        r for r in b3_instances
        if (r.get("fact_id"), r.get("t_old"), r.get("t_new")) not in excluded_keys
    ]

    print(f"\n→ {len(b1_filtered)} B1 + {len(b3_filtered)} B3 instances kept for F1 diagnosis")
    return b1_filtered, b3_filtered


# ── F1-a: SAT Probe ─────────────────────────────────────────────────────────

def run_f1a(model, b1_instances, template, out_dir):
    """SAT Probe: logistic regression on attention-to-year features."""
    print("\n" + "=" * 60)
    print("F1-a: SAT Probe")
    print("=" * 60)

    X, y, meta = collect_features(
        model, b1_instances, template=template, verbose=True,
    )

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"\nOverride success: {n_pos}/{len(y)} ({100*n_pos/len(y):.1f}%)")
    print(f"Override failure: {n_neg}/{len(y)} ({100*n_neg/len(y):.1f}%)")

    probe_result = train_probe(X, y, C=0.05)
    print(f"\nSAT Probe AUROC: {probe_result.auroc:.3f} ± {probe_result.auroc_std:.3f}")

    top_heads = analyse_weights(
        probe_result, model.cfg.n_layers, model.cfg.n_heads, top_k=10,
    )
    n_nonzero = sum(1 for _, _, c in top_heads if c != 0.0)
    print(f"\nTop-10 (layer, head) by |coefficient|  [{n_nonzero} non-zero]:")
    for layer, head, coef in top_heads:
        direction = "↑ success" if coef > 0 else ("↓ success" if coef < 0 else "zero")
        print(f"  L{layer:2d}.H{head:2d}  coef={coef:+.4e}  ({direction})")

    if n_nonzero == 0:
        print("\n  WARNING: all coefficients are zero.")
        print("  Possible causes:")
        print("    - Old code running (C not updated) — check [probe] line above")
        print("    - All attention features nearly identical across instances")
        print("    → F1 (attention to year tokens) may genuinely not be the mechanism")

    # save — use scientific notation strings for coef so tiny values are visible
    probe_out = {
        "auroc": probe_result.auroc,
        "auroc_std": probe_result.auroc_std,
        "n_samples": probe_result.n_samples,
        "n_positive": probe_result.n_positive,
        "n_negative": probe_result.n_samples - probe_result.n_positive,
        "top_heads": [
            {"layer": l, "head": h, "coef": f"{c:.4e}"} for l, h, c in top_heads
        ],
        "instance_meta": meta,
    }
    with open(out_dir / "f1a_sat_probe.json", "w") as f:
        json.dump(probe_out, f, indent=2, ensure_ascii=False)

    return probe_result, X, y, meta


# ── F1-b: Attention comparison ───────────────────────────────────────────────

def run_f1b(model, b1_instances, b3_instances, y_labels, top_heads, template, out_dir):
    """Compare attention to year tokens: B1-success vs B1-failure vs B3."""
    print("\n" + "=" * 60)
    print("F1-b: Attention Comparison (B1-success vs B1-failure vs B3)")
    print("=" * 60)

    if not top_heads:
        print("No top heads from SAT probe. Using all heads.")
        head_set = None
    else:
        head_set = [(l, h) for l, h, _ in top_heads[:5]]
        print(f"Analysing top-5 heads: {head_set}")

    groups = {"b1_success": [], "b1_failure": [], "b3": []}

    import gc as _gc

    for idx, inst in enumerate(tqdm(b1_instances, desc="F1-b  B1 attn", unit="inst", dynamic_ncols=True)):
        context = inst.get("evidence_new", inst.get("context", ""))
        question = inst.get("question", "")
        t_new = inst.get("t_new")

        prompt = build_prompt(context, question, template=template)
        tokens = model.to_tokens(prompt, prepend_bos=False)
        year_pos = find_year_positions(tokens[0], model.tokenizer, target_year=t_new)
        if not year_pos:
            year_pos = find_year_positions(tokens[0], model.tokenizer)

        attn = extract_attention_to_positions(model, tokens, year_pos)

        label = y_labels[idx] if idx < len(y_labels) else 0
        key = "b1_success" if label == 1 else "b1_failure"
        groups[key].append(attn.numpy())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for inst in tqdm(b3_instances, desc="F1-b  B3 attn", unit="inst", dynamic_ncols=True):
        context = inst.get("evidence_new", inst.get("context", ""))
        question = inst.get("question", "")
        t_new = inst.get("t_new")

        prompt = build_prompt(context, question, template=template)
        tokens = model.to_tokens(prompt, prepend_bos=False)
        year_pos = find_year_positions(tokens[0], model.tokenizer, target_year=t_new)
        if not year_pos:
            year_pos = find_year_positions(tokens[0], model.tokenizer)
        attn = extract_attention_to_positions(model, tokens, year_pos)
        groups["b3"].append(attn.numpy())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # aggregate per group
    results = {}
    for group_name, attn_list in groups.items():
        if not attn_list:
            print(f"  {group_name}: no instances")
            continue
        stacked = np.stack(attn_list)  # [N, L, H]
        mean_attn = stacked.mean(axis=0)  # [L, H]
        results[group_name] = {
            "count": len(attn_list),
            "mean_attn_all": mean_attn.tolist(),
        }

        if head_set:
            head_vals = [mean_attn[l, h] for l, h in head_set]
            mean_val = np.mean(head_vals)
            print(f"  {group_name} (n={len(attn_list)}): "
                  f"mean attn at top-5 heads = {mean_val:.4f}")
            results[group_name]["mean_attn_top5"] = float(mean_val)
            results[group_name]["per_head"] = [
                {"layer": l, "head": h, "attn": float(mean_attn[l, h])}
                for l, h in head_set
            ]

    # statistical comparison
    if groups["b1_success"] and groups["b1_failure"] and head_set:
        succ_vals = np.array([
            np.mean([a[l, h] for l, h in head_set])
            for a in groups["b1_success"]
        ])
        fail_vals = np.array([
            np.mean([a[l, h] for l, h in head_set])
            for a in groups["b1_failure"]
        ])
        from scipy.stats import mannwhitneyu
        try:
            stat, pval = mannwhitneyu(succ_vals, fail_vals, alternative="greater")
            results["mann_whitney_U"] = float(stat)
            results["mann_whitney_p"] = float(pval)
            print(f"\n  Mann-Whitney U (success > failure): U={stat:.1f}, p={pval:.4f}")
        except ValueError:
            pass

    with open(out_dir / "f1b_attention_comparison.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ── F1-c: Attention knockout ─────────────────────────────────────────────────

def run_f1c(model, b1_instances, y_labels, top_heads, template, out_dir):
    """Causal test: knock out attention to year tokens, measure probability drop."""
    print("\n" + "=" * 60)
    print("F1-c: Attention Knockout")
    print("=" * 60)

    # run knockout on B1-success instances only (to measure causal necessity)
    success_indices = [i for i, y in enumerate(y_labels) if y == 1]
    if not success_indices:
        print("No B1-success instances to run knockout on.")
        return {}

    # determine knockout layer window from top heads
    if top_heads:
        head_layers = sorted(set(l for l, h, _ in top_heads[:5]))
        min_l, max_l = head_layers[0], head_layers[-1]
        window = 2
        ko_layers = list(range(max(0, min_l - window), min(model.cfg.n_layers, max_l + window + 1)))
    else:
        ko_layers = list(range(model.cfg.n_layers))

    print(f"Knockout layer window: {ko_layers[0]}–{ko_layers[-1]}")

    ko_results = []
    ko_bar = tqdm(
        success_indices,
        desc="F1-c  knockout",
        unit="inst",
        dynamic_ncols=True,
    )
    for idx in ko_bar:
        inst = b1_instances[idx]
        context = inst.get("evidence_new", inst.get("context", ""))
        question = inst.get("question", "")
        answer_new = inst.get("answer_new", "")
        answer_old = inst.get("answer_old", "")
        t_new = inst.get("t_new")

        prompt = build_prompt(context, question, template=template)
        tokens = model.to_tokens(prompt, prepend_bos=False)
        year_pos = find_year_positions(tokens[0], model.tokenizer, target_year=t_new)
        if not year_pos:
            year_pos = find_year_positions(tokens[0], model.tokenizer)

        if not year_pos:
            ko_bar.set_postfix_str("skip (no year toks)", refresh=True)
            continue

        # get first-token IDs for answer_new and answer_old
        new_tids = model.tokenizer.encode(f" {answer_new}", add_special_tokens=False)[:1]
        old_tids = model.tokenizer.encode(f" {answer_old}", add_special_tokens=False)[:1]
        track_tids = list(set(new_tids + old_tids))

        ko = attention_knockout(
            model, tokens, year_pos,
            knockout_layers=ko_layers,
            answer_token_ids=track_tids,
        )

        # compute probability drop for answer_new first token
        new_tid = new_tids[0] if new_tids else None
        old_tid = old_tids[0] if old_tids else None
        entry = {
            "instance_id": inst.get("instance_id", f"inst_{idx}"),
            "answer_new": answer_new,
            "answer_old": answer_old,
            "n_year_tokens_blocked": len(year_pos),
            "knockout_layers": f"{ko_layers[0]}-{ko_layers[-1]}",
        }

        if new_tid is not None and new_tid in ko["probs_clean"]:
            p_clean = ko["probs_clean"][new_tid]
            p_ko = ko["probs_ko"][new_tid]
            entry["p_new_clean"] = p_clean
            entry["p_new_knockout"] = p_ko
            entry["p_new_drop"] = p_clean - p_ko
            entry["p_new_drop_relative"] = (
                (p_clean - p_ko) / max(p_clean, 1e-12)
            )

        if old_tid is not None and old_tid in ko["probs_clean"]:
            p_clean = ko["probs_clean"][old_tid]
            p_ko = ko["probs_ko"][old_tid]
            entry["p_old_clean"] = p_clean
            entry["p_old_knockout"] = p_ko
            entry["p_old_gain"] = p_ko - p_clean

        ko_results.append(entry)
        drop_pct = entry.get("p_new_drop_relative", float("nan"))
        ko_bar.set_postfix(
            year_toks=len(year_pos),
            drop=f"{drop_pct:+.2f}" if drop_pct == drop_pct else "n/a",
            refresh=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if ko_results:
        drops = [r["p_new_drop_relative"] for r in ko_results if "p_new_drop_relative" in r]
        if drops:
            mean_drop = np.mean(drops)
            median_drop = np.median(drops)
            pct_above_10 = sum(1 for d in drops if d > 0.10) / len(drops)
            print(f"\nResults over {len(drops)} B1-success instances:")
            print(f"  Mean relative p(answer_new) drop:   {mean_drop:.4f}")
            print(f"  Median relative p(answer_new) drop: {median_drop:.4f}")
            print(f"  Fraction with >10% drop:            {pct_above_10:.2%}")

    summary = {
        "knockout_layers": ko_layers,
        "n_instances": len(ko_results),
        "per_instance": ko_results,
    }
    with open(out_dir / "f1c_attention_knockout.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F1 Diagnostic — Time Not Set")
    parser.add_argument("--data", required=True, help="Path to JSONL data file")
    parser.add_argument(
        "--model", default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model name (must be supported by TransformerLens)",
    )
    parser.add_argument("--template", default="plain", choices=["plain", "llama2", "llama3", "phi3"])
    parser.add_argument("--out", default="results/f1_diagnostic", help="Output directory")
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"],
        help="auto: float32 on MPS/CPU, float16 on CUDA",
    )
    parser.add_argument("--max-instances", type=int, default=None, help="Limit instances for quick testing")
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=["f1a", "f1b", "f1c"],
        help="Skip specific sub-experiments",
    )
    parser.add_argument(
        "--no-a1-filter", action="store_true",
        help=(
            "Skip the A1 parametric memory profiling step (NOT recommended). "
            "By default the script runs A1 (year cue, no context) first and "
            "excludes instances where the model already knows answer_new with "
            "just the year cue.  Use this flag only for debugging."
        ),
    )
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    if args.dtype == "auto":
        # MPS/CPU: float32; CUDA: float16
        _dev = args.device
        if _dev == "auto":
            _dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        resolved_dtype = torch.float16 if _dev == "cuda" else torch.float32
    else:
        resolved_dtype = dtype_map[args.dtype]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}")
    print(f"Template: {args.template}")
    print(f"Device:   {args.device}")
    print(f"Output:   {out_dir}")

    # load data
    b1_instances, b3_instances = load_instances(args.data)
    if args.max_instances:
        b1_instances = b1_instances[:args.max_instances]
        b3_instances = b3_instances[:args.max_instances]
    print(f"\nLoaded {len(b1_instances)} B1 instances, {len(b3_instances)} B3 instances")

    # load model
    print(f"\nLoading model {args.model}  (this may take 1–3 min on first run)…")
    with tqdm(total=1, desc="Loading weights", unit="model",
              bar_format="{desc}: {elapsed} elapsed {postfix}") as pbar:
        model = load_model(args.model, device=args.device, dtype=resolved_dtype)
        pbar.set_postfix_str("done")
        pbar.update(1)
    print(f"  {model.cfg.n_layers} layers × {model.cfg.n_heads} heads  "
          f"d_model={model.cfg.d_model}")

    # A1 filter: exclude instances where year cue alone is enough (default ON)
    if not args.no_a1_filter:
        b1_instances, b3_instances = run_a1_filter(
            model, b1_instances, b3_instances, args.template, out_dir,
        )
        if not b1_instances:
            print("\nNo instances remain after A1 filter. Exiting.")
            return

    # F1-a
    probe_result, X, y, meta = None, None, None, None
    if "f1a" not in args.skip:
        probe_result, X, y, meta = run_f1a(model, b1_instances, args.template, out_dir)

    top_heads_list = probe_result.top_heads if probe_result else []

    # F1-b
    if "f1b" not in args.skip:
        if y is None:
            # need labels — run quick feature collection
            X, y, meta = collect_features(model, b1_instances, template=args.template)
        run_f1b(model, b1_instances, b3_instances, y, top_heads_list, args.template, out_dir)

    # F1-c
    if "f1c" not in args.skip:
        if y is None:
            X, y, meta = collect_features(model, b1_instances, template=args.template)
        run_f1c(model, b1_instances, y, top_heads_list, args.template, out_dir)

    print("\n" + "=" * 60)
    print("F1 Diagnostic complete. Results saved to:", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
