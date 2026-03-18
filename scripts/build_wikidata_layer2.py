#!/usr/bin/env python3
"""Build Layer-2 (EvalInstances) from the Wikidata-sourced Layer-1 JSONL.

Generates EvalInstances per FactTimeline.  Use --layers to control which
instance types are emitted (default: A1 only — explicit-year temporal recall).

Instance types
--------------
  A1  explicit year, no context    "As of {year}, who was …?"       ← default
  A2  implicit time, no context    "Currently, who is …?"
  A3  yes/no correct               "In {year}, was X …?  (Yes)"
  A4  yes/no stale                 "In {year}, was X still …?  (No)"
  B1  real Wikipedia context + explicit question year
  B2  real Wikipedia context + implicit question
  B3  year-stripped context + explicit question year
  B4  year-stripped context + no time cue
  B5  multi-span context (evidence_old + evidence_new) + explicit question year
      → model must read year to select answer_new over answer_old
  C1  adversarial context (old answer labelled current)
  C2  mislabelled-year context
  C3  plain no-cue, no context (pure baseline)

Usage
-----
    # Only A1 (default, fast smoke-test)
    python code/scripts/build_wikidata_layer2.py

    # All 11 types
    python code/scripts/build_wikidata_layer2.py --layers all

    # Explicit temporal recall + context conflict
    python code/scripts/build_wikidata_layer2.py --layers A1 B1 B2
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "source"))

from fact_timeline.models import FactTimeline                       # noqa: E402
from fact_timeline.eval_builder import build_eval_instances         # noqa: E402

DEFAULT_LAYER1 = REPO_ROOT / "data/processed/wikidata_layer1.jsonl"
DEFAULT_OUT    = REPO_ROOT / "data/processed/wikidata_layer2.jsonl"


ALL_LAYERS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--layer1",  default=str(DEFAULT_LAYER1))
    parser.add_argument("--out",     default=str(DEFAULT_OUT))
    parser.add_argument(
        "--layers", nargs="+", default=["A1"], metavar="TYPE",
        help=(
            "Instance types to emit, e.g. A1 B1 B2 B5, or 'all' for all types. "
            "(default: A1)"
        ),
    )
    args = parser.parse_args()

    if args.layers == ["all"]:
        keep_layers = set(ALL_LAYERS)
    else:
        keep_layers = set(args.layers)

    layer1_path = Path(args.layer1)
    out_path    = Path(args.out)
    stats_path  = out_path.with_suffix("").with_name(out_path.stem + "_stats.txt")

    if not layer1_path.exists():
        sys.exit(f"[ERROR] Layer-1 not found: {layer1_path}\n"
                 "  Run build_wikidata_layer1.py first.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Layers      : {sorted(keep_layers)}")
    print(f"Reading Layer-1 from {layer1_path} …")
    timelines: list[FactTimeline] = []
    with open(layer1_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                timelines.append(FactTimeline.from_dict(json.loads(line)))
    print(f"  Loaded {len(timelines)} FactTimelines")

    # ── Generate instances ────────────────────────────────────────────────────
    all_instances = []
    skipped = 0
    layer_counts: Counter = Counter()
    task_counts:  Counter = Counter()
    rel_counts:   Counter = Counter()

    for tl in timelines:
        instances = build_eval_instances(tl)
        if not instances:
            skipped += 1
            continue
        for inst in instances:
            layer_id = inst.instance_id[:2]
            if layer_id not in keep_layers:
                continue
            layer_counts[layer_id] += 1
            task_counts[inst.task_type] += 1
            rel_counts[tl.property_label] += 1
            all_instances.append(inst)

    # ── Write ─────────────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as fh:
        for inst in all_instances:
            fh.write(inst.to_json() + "\n")

    # ── Stats ─────────────────────────────────────────────────────────────────
    stat_lines = [
        "=== Wikidata Layer-2 Stats ===",
        f"Input FactTimelines  : {len(timelines)}",
        f"Skipped (no pair)    : {skipped}",
        f"Total EvalInstances  : {len(all_instances)}",
        f"Avg instances/tl     : {len(all_instances) / max(len(timelines) - skipped, 1):.1f}",
        "",
        "── By layer ──────────────────────────────",
        *[f"  {k}: {v}" for k, v in sorted(layer_counts.items())],
        "",
        "── By task_type ──────────────────────────",
        *[f"  {k}: {v}" for k, v in sorted(task_counts.items())],
        "",
        "── By relation ───────────────────────────",
        *[f"  {k}: {v}" for k, v in rel_counts.most_common()],
    ]
    stats_text = "\n".join(stat_lines)
    with open(stats_path, "w", encoding="utf-8") as fh:
        fh.write(stats_text + "\n")

    print(f"\n{stats_text}")
    print(f"\n[OK] Layer-2 written to {out_path}")

    # ── Sample instances ──────────────────────────────────────────────────────
    for layer_id, label in [("A1", "Temporal Recall"), ("B1", "Context Conflict"), ("C1", "Adversarial")]:
        print(f"\n─── Sample {layer_id} ({label}) ───────────────────────────")
        for inst in all_instances:
            if inst.instance_id.startswith(layer_id):
                d = inst.to_dict()
                print(f"  subject  : {d['subject_label']}")
                print(f"  relation : {d['property_label']}")
                print(f"  t_old={d['t_old']} → {d['answer_old']}")
                print(f"  t_new={d['t_new']} → {d['answer_new']}")
                print(f"  question : {d['question']}")
                if d["context"]:
                    print(f"  context  : {d['context'][:120]}…")
                break


if __name__ == "__main__":
    main()
