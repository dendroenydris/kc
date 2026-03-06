import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch

from tatm_experiments import (
    cleanup_memory,
    get_discriminative_token_ids,
    get_logit_diff,
    load_data,
    load_hooked_model_with_phi3_compat,
)


@dataclass
class DiagnosisConfig:
    temporal_heads: List[Tuple[int, int]]
    routing_heads: List[Tuple[int, int]]
    late_mlp_layers: List[int]
    eps: float = 1e-3


def parse_heads(spec: str) -> List[Tuple[int, int]]:
    # Example: "10:5,11:2"
    if not spec.strip():
        return []
    pairs = []
    for item in spec.split(","):
        layer, head = item.split(":")
        pairs.append((int(layer), int(head)))
    return pairs


def parse_layers(spec: str) -> List[int]:
    # Example: "28,29,30,31"
    if not spec.strip():
        return []
    return [int(x) for x in spec.split(",")]


def locate_temporal_token_pos(model, prompt: str, year_text: str) -> int:
    # Keep last occurrence to prefer the explicit year in the question.
    tokens = model.to_tokens(prompt, prepend_bos=True)[0]
    year_tokens = model.to_tokens(" " + year_text, prepend_bos=False)[0]
    if len(year_tokens) == 0:
        return -1
    year_first_id = year_tokens[0].item()
    pos = -1
    for i, tok in enumerate(tokens):
        if tok.item() == year_first_id:
            pos = i
    return pos


def run_f1_patch_diff(model, clean_prompt: str, failure_prompt: str, t_new_id: int, t_old_id: int, temporal_heads, temporal_token_pos):
    with torch.inference_mode():
        baseline_logits = model(failure_prompt)
        baseline_diff = get_logit_diff(baseline_logits, t_new_id, t_old_id).item()

    layers_to_patch = sorted(set(layer for layer, _ in temporal_heads))
    hook_names = {f"blocks.{layer}.attn.hook_z" for layer in layers_to_patch}
    with torch.inference_mode():
        _, clean_cache = model.run_with_cache(clean_prompt, names_filter=lambda name: name in hook_names)

    def patch_temporal_head_activation(activations, hook):
        layer = int(hook.name.split(".")[1])
        for head_layer, head_idx in temporal_heads:
            if head_layer == layer:
                activations[0, temporal_token_pos, head_idx, :] = clean_cache[hook.name][0, temporal_token_pos, head_idx, :]
        return activations

    hooks = [(f"blocks.{layer}.attn.hook_z", patch_temporal_head_activation) for layer in layers_to_patch]
    with torch.inference_mode():
        patched_logits = model.run_with_hooks(failure_prompt, fwd_hooks=hooks)
        patched_diff = get_logit_diff(patched_logits, t_new_id, t_old_id).item()

    del baseline_logits, clean_cache, patched_logits
    return baseline_diff, patched_diff


def run_f2_knockout_diff(model, failure_prompt: str, t_new_id: int, t_old_id: int, routing_heads, temporal_token_pos: int):
    tokens = model.to_tokens(failure_prompt, prepend_bos=True)[0]
    final_token_pos = len(tokens) - 1

    with torch.inference_mode():
        baseline_logits = model(failure_prompt)
        baseline_diff = get_logit_diff(baseline_logits, t_new_id, t_old_id).item()

    def knockout_attention_path(pattern, hook):
        layer = int(hook.name.split(".")[1])
        for head_layer, head_idx in routing_heads:
            if head_layer == layer:
                pattern[0, head_idx, final_token_pos, temporal_token_pos] = 0.0
        return pattern

    layers = sorted(set(layer for layer, _ in routing_heads))
    hooks = [(f"blocks.{layer}.attn.hook_pattern", knockout_attention_path) for layer in layers]
    with torch.inference_mode():
        knockout_logits = model.run_with_hooks(failure_prompt, fwd_hooks=hooks)
        knockout_diff = get_logit_diff(knockout_logits, t_new_id, t_old_id).item()

    del baseline_logits, knockout_logits
    return baseline_diff, knockout_diff


def run_f3_ablation_diff(model, failure_prompt: str, t_new_id: int, t_old_id: int, late_mlp_layers):
    with torch.inference_mode():
        baseline_logits = model(failure_prompt)
        baseline_diff = get_logit_diff(baseline_logits, t_new_id, t_old_id).item()

    def ablate_mlp(activations, hook):
        activations[:] = 0.0
        return activations

    hooks = [(f"blocks.{layer}.hook_mlp_out", ablate_mlp) for layer in late_mlp_layers]
    with torch.inference_mode():
        ablated_logits = model.run_with_hooks(failure_prompt, fwd_hooks=hooks)
        ablated_diff = get_logit_diff(ablated_logits, t_new_id, t_old_id).item()

    del baseline_logits, ablated_logits
    return baseline_diff, ablated_diff


def classify_failure(
    baseline_diff: float,
    f1_patched_diff: float,
    f2_knockout_diff: float,
    f3_ablated_diff: float,
    eps: float,
) -> str:
    # Only failure cases should be classified.
    if baseline_diff >= 0:
        return "non_failure"
    if f1_patched_diff > 0:
        return "F1"
    if f3_ablated_diff > 0:
        return "F3"
    if abs(f2_knockout_diff - baseline_diff) <= eps:
        return "F2"
    return "unresolved"


def save_pie_chart(counter: Counter, out_path: str):
    labels = ["F1", "F2", "F3", "unresolved", "non_failure"]
    values = [counter.get(k, 0) for k in labels]
    total = sum(values)
    if total == 0:
        values = [1]
        labels = ["no_samples"]

    plt.figure(figsize=(7, 7))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("TATM Diagnosis Distribution (F1/F2/F3)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_summary_md(out_path: str, model_name: str, dataset_path: str, records: List[dict], counter: Counter):
    total = len(records)
    failures = sum(1 for r in records if r["baseline_diff"] < 0)
    f1 = counter.get("F1", 0)
    f2 = counter.get("F2", 0)
    f3 = counter.get("F3", 0)
    unresolved = counter.get("unresolved", 0)
    non_failure = counter.get("non_failure", 0)

    lines = []
    lines.append("# TATM Diagnosis Report")
    lines.append("")
    lines.append("## Run Config")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Samples processed: `{total}`")
    lines.append("")
    lines.append("## Headline Results")
    lines.append(f"- Failure cases (baseline prefers old fact): `{failures}/{total}`")
    lines.append(f"- F1 count: `{f1}`")
    lines.append(f"- F2 count: `{f2}`")
    lines.append(f"- F3 count: `{f3}`")
    lines.append(f"- Unresolved failures: `{unresolved}`")
    lines.append(f"- Non-failure cases: `{non_failure}`")
    lines.append("")
    lines.append("## Interpretation for Supervisor Meeting")
    lines.append("- `F1` suggests missing/weak time-state activation.")
    lines.append("- `F2` suggests time-state exists but routing signal is ineffective.")
    lines.append("- `F3` suggests late parametric-memory override after routing.")
    lines.append("- `unresolved` indicates these heuristics were inconclusive and need deeper per-head probing.")
    lines.append("")
    lines.append("## Representative Cases")
    for label in ["F1", "F2", "F3", "unresolved"]:
        examples = [r for r in records if r["diagnosis"] == label][:2]
        if not examples:
            continue
        lines.append(f"### {label}")
        for ex in examples:
            lines.append(
                f"- ID `{ex['id']}` | Δbase={ex['baseline_diff']:.4f}, "
                f"ΔF1={ex['f1_patched_diff']:.4f}, ΔF2={ex['f2_knockout_diff']:.4f}, ΔF3={ex['f3_ablated_diff']:.4f}"
            )
            lines.append(f"  - Q: {ex['question']}")
            lines.append(f"  - old/new: `{ex['answer_old']}` -> `{ex['answer_new']}`")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run TATM diagnosis and generate meeting-ready artifacts.")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset", required=True, help="Path to temporal conflict dataset JSONL.")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--out-dir", default="outputs/diagnosis")
    parser.add_argument("--temporal-heads", default="10:5,11:2")
    parser.add_argument("--routing-heads", default="15:0,16:7")
    parser.add_argument("--late-mlp-layers", default="28,29,30,31")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = DiagnosisConfig(
        temporal_heads=parse_heads(args.temporal_heads),
        routing_heads=parse_heads(args.routing_heads),
        late_mlp_layers=parse_layers(args.late_mlp_layers),
    )

    print("Initializing TATM diagnosis run...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.out_dir}")

    model = load_hooked_model_with_phi3_compat(args.model, args.device)
    model.eval()

    data = load_data(args.dataset)
    if args.max_samples > 0:
        data = data[: args.max_samples]

    records = []
    for idx, sample in enumerate(data, start=1):
        print(f"\n[{idx}/{len(data)}] ID={sample.get('id', idx)}")
        context = f"{sample['evidence_old']} {sample['evidence_new']}"
        clean_prompt = f"{context} Question: As of {sample['t_old'][:4]}, {sample['question']} Answer:"
        failure_prompt = f"{context} Question: As of {sample['t_new'][:4]}, {sample['question']} Answer:"

        t_new_id, t_old_id, diff_pos = get_discriminative_token_ids(model, sample["answer_new"], sample["answer_old"])
        if diff_pos == -1:
            print("  - Skipped (identical tokenization for old/new answers).")
            continue

        temporal_token_pos = locate_temporal_token_pos(model, failure_prompt, sample["t_new"][:4])
        if temporal_token_pos == -1:
            print("  - Skipped (cannot locate temporal token in prompt).")
            continue

        try:
            baseline_diff, f1_patched_diff = run_f1_patch_diff(
                model, clean_prompt, failure_prompt, t_new_id, t_old_id, cfg.temporal_heads, temporal_token_pos
            )
            _, f2_knockout_diff = run_f2_knockout_diff(
                model, failure_prompt, t_new_id, t_old_id, cfg.routing_heads, temporal_token_pos
            )
            _, f3_ablated_diff = run_f3_ablation_diff(
                model, failure_prompt, t_new_id, t_old_id, cfg.late_mlp_layers
            )
            diagnosis = classify_failure(
                baseline_diff, f1_patched_diff, f2_knockout_diff, f3_ablated_diff, cfg.eps
            )
            print(
                f"  - Δbase={baseline_diff:.4f}, ΔF1={f1_patched_diff:.4f}, "
                f"ΔF2={f2_knockout_diff:.4f}, ΔF3={f3_ablated_diff:.4f} -> {diagnosis}"
            )
        except Exception as exc:
            print(f"  - Error: {exc}")
            diagnosis = "error"
            baseline_diff = 0.0
            f1_patched_diff = 0.0
            f2_knockout_diff = 0.0
            f3_ablated_diff = 0.0

        records.append(
            {
                "id": sample.get("id", idx),
                "question": sample["question"],
                "answer_old": sample["answer_old"],
                "answer_new": sample["answer_new"],
                "baseline_diff": baseline_diff,
                "f1_patched_diff": f1_patched_diff,
                "f2_knockout_diff": f2_knockout_diff,
                "f3_ablated_diff": f3_ablated_diff,
                "diagnosis": diagnosis,
            }
        )
        cleanup_memory()

    counter = Counter(r["diagnosis"] for r in records)
    pie_path = os.path.join(args.out_dir, "f1_f2_f3_pie.png")
    save_pie_chart(counter, pie_path)

    json_path = os.path.join(args.out_dir, "diagnosis_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.out_dir, "diagnosis_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "answer_old",
                "answer_new",
                "baseline_diff",
                "f1_patched_diff",
                "f2_knockout_diff",
                "f3_ablated_diff",
                "diagnosis",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    summary_path = os.path.join(args.out_dir, "summary.md")
    write_summary_md(summary_path, args.model, args.dataset, records, counter)

    print("\nDiagnosis run completed.")
    print(f"- Pie chart: {pie_path}")
    print(f"- Detailed JSON: {json_path}")
    print(f"- Detailed CSV: {csv_path}")
    print(f"- Meeting summary: {summary_path}")


if __name__ == "__main__":
    main()
