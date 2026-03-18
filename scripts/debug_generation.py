"""Standalone generation debug script.

Loads Phi-3 (or any model) and runs a single generation with full diagnostic
output so we can pinpoint where generation breaks.

Usage (Colab):
    !python source/tatm/scripts/debug_generation.py \
        --model microsoft/Phi-3-mini-4k-instruct \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ── make sure source/ is on the path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from tatm.model import _collect_eos_ids, _patch_phi3_rope_scaling, _needs_trust_remote_code

SEP = "─" * 72


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def load_model_debug(model_name: str, device: str, dtype: torch.dtype):
    from transformer_lens import HookedTransformer

    name_lower = model_name.lower()
    if "phi-3" in name_lower or "phi3" in name_lower:
        _patch_phi3_rope_scaling(model_name)

    extra = {"trust_remote_code": True} if _needs_trust_remote_code(model_name) else {}
    model = HookedTransformer.from_pretrained(
        model_name, device=device, dtype=dtype, **extra
    )
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # ── 1. Load model ─────────────────────────────────────────────────────────
    section("1. Loading model")
    model = load_model_debug(args.model, args.device, dtype)
    tok = model.tokenizer
    print(f"  model.cfg.device  : {model.cfg.device}")
    print(f"  model.cfg.d_model : {model.cfg.d_model}")
    print(f"  vocab size        : {model.cfg.d_vocab}")
    print(f"  n_layers          : {model.cfg.n_layers}")
    print(f"  n_heads           : {model.cfg.n_heads}")

    # ── 2. Tokenizer special tokens ───────────────────────────────────────────
    section("2. Tokenizer special tokens")
    print(f"  eos_token         : {repr(tok.eos_token)}  (id={tok.eos_token_id})")
    print(f"  bos_token         : {repr(tok.bos_token)}  (id={tok.bos_token_id})")
    print(f"  pad_token         : {repr(tok.pad_token)}  (id={tok.pad_token_id})")
    eos_ids = _collect_eos_ids(tok)
    print(f"  all EOS ids       : {eos_ids}")
    for tid in eos_ids:
        print(f"    {tid} → {repr(tok.decode([tid]))}")

    # ── 3. Prompt encoding ────────────────────────────────────────────────────
    section("3. Prompt encoding")
    PROMPT = (
        "<|system|>\n"
        "Answer the question based on the provided context. "
        "Give a short, direct answer.<|end|>\n"
        "<|user|>\n"
        "Context: Joe Biden became the 46th President of the United States "
        "on January 20, 2021, succeeding Donald Trump.\n\n"
        "Question: As of 2021, who is the President of the United States?<|end|>\n"
        "<|assistant|>\n"
    )
    print("  Raw prompt:")
    print(PROMPT)

    enc = tok(PROMPT, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    print(f"  Encoded length: {input_ids.shape[1]} tokens")
    print("\n  Token-by-token breakdown:")
    for i, tid in enumerate(input_ids[0].tolist()):
        raw_tok = tok.convert_ids_to_tokens([tid])[0]
        decoded = tok.decode([tid])
        print(f"    [{i:3d}] id={tid:6d}  raw={repr(raw_tok):20s}  decoded={repr(decoded)}")

    # ── 4. Single forward pass ────────────────────────────────────────────────
    section("4. Single forward pass (logits at last position)")
    ids_on_device = input_ids.to(model.cfg.device)
    with torch.no_grad():
        logits = model(ids_on_device, prepend_bos=False)   # [1, seq, vocab]
    last_logits = logits[0, -1, :]
    top_k = torch.topk(last_logits, 10)
    print("  Top-10 predicted next tokens:")
    for rank, (score, tid) in enumerate(zip(top_k.values.tolist(), top_k.indices.tolist())):
        raw_tok = tok.convert_ids_to_tokens([tid])[0]
        decoded = tok.decode([tid])
        print(f"    rank {rank+1:2d}  score={score:8.3f}  id={tid:6d}  "
              f"raw={repr(raw_tok):20s}  decoded={repr(decoded)}")
    greedy_id = int(last_logits.argmax())
    print(f"\n  → greedy pick: id={greedy_id}  decoded={repr(tok.decode([greedy_id]))}")

    # ── 5. Manual greedy loop with full trace ─────────────────────────────────
    section("5. Manual greedy generation (full trace)")
    eos_set = set(eos_ids)
    current_ids = ids_on_device.clone()
    generated: list[int] = []

    print(f"  Generating up to {args.max_new_tokens} tokens…\n")
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            logits = model(current_ids, prepend_bos=False)
            next_id = int(logits[0, -1, :].argmax())
            raw_tok = tok.convert_ids_to_tokens([next_id])[0]
            decoded = tok.decode([next_id])
            is_eos = next_id in eos_set
            print(f"  step {step:2d}  id={next_id:6d}  raw={repr(raw_tok):22s}  "
                  f"decoded={repr(decoded):20s}  {'← EOS STOP' if is_eos else ''}")
            if is_eos:
                break
            generated.append(next_id)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_id]], device=current_ids.device)],
                dim=1,
            )

    # ── 6. Final answer ───────────────────────────────────────────────────────
    section("6. Final decoded answer")
    raw = tok.decode(generated, skip_special_tokens=True)
    print(f"  raw decoded : {repr(raw)}")
    print(f"  cleaned     : {repr(raw.strip().split(chr(10))[0].split('<|')[0].strip())}")


if __name__ == "__main__":
    main()
