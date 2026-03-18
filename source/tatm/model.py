"""Model loading, prompt formatting, and tokenization utilities for TATM."""
from __future__ import annotations

import re
from typing import Optional

import torch
from transformer_lens import HookedTransformer

YEAR_PAT = re.compile(r"\b(19|20)\d{2}\b")

# ── Model loading ────────────────────────────────────────────────────────────

def load_model(
    model_name: str,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
) -> HookedTransformer:
    """Load a HookedTransformer with sensible defaults for TATM experiments.

    device="auto" selects CUDA → MPS → CPU in order of availability.
    dtype defaults to float32 for MPS/CPU (float16 is unstable on MPS).
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # MPS does not support float16 reliably; fall back to float32
    if device == "mps" and dtype == torch.float16:
        dtype = torch.float32

    print(f"  → device={device}, dtype={dtype}")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


# ── Prompt formatting ────────────────────────────────────────────────────────

_TEMPLATES: dict[str, dict[str, str]] = {
    "plain": {
        "with_ctx": "Context: {context}\n\nQuestion: {question}\nAnswer:",
        "no_ctx":   "Question: {question}\nAnswer:",
    },
    "llama3": {
        "with_ctx": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Answer the question based on the provided context. "
            "Give a short, direct answer.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Context: {context}\n\nQuestion: {question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "no_ctx": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Answer the question. Give a short, direct answer.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Question: {question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    "llama2": {
        "with_ctx": (
            "[INST] <<SYS>>\nAnswer the question based on the provided context. "
            "Give a short, direct answer.\n<</SYS>>\n\n"
            "Context: {context}\n\nQuestion: {question} [/INST]"
        ),
        "no_ctx": (
            "[INST] <<SYS>>\nAnswer the question. "
            "Give a short, direct answer.\n<</SYS>>\n\n"
            "Question: {question} [/INST]"
        ),
    },
    # Phi-3 chat template  (<|user|> ... <|end|> \n <|assistant|>)
    "phi3": {
        "with_ctx": (
            "<|system|>\nAnswer the question based on the provided context. "
            "Give a short, direct answer.<|end|>\n"
            "<|user|>\nContext: {context}\n\nQuestion: {question}<|end|>\n"
            "<|assistant|>\n"
        ),
        "no_ctx": (
            "<|system|>\nAnswer the question. "
            "Give a short, direct answer.<|end|>\n"
            "<|user|>\nQuestion: {question}<|end|>\n"
            "<|assistant|>\n"
        ),
    },
}


def build_prompt(
    context: str,
    question: str,
    template: str = "plain",
) -> str:
    """Format context + question into a model prompt."""
    tpl = _TEMPLATES.get(template, _TEMPLATES["plain"])
    key = "with_ctx" if context.strip() else "no_ctx"
    return tpl[key].format(context=context, question=question)


# ── Year-token identification ────────────────────────────────────────────────

def find_year_positions(
    token_ids: torch.Tensor,
    tokenizer,
    *,
    target_year: Optional[int] = None,
) -> list[int]:
    """Find token positions that encode 4-digit years.

    Works across tokenizer families:
    - SentencePiece (▁ prefix, used by Phi-3 / LLaMA-2)
    - BPE with Ġ prefix (GPT-2 / RoBERTa style)
    - Plain decode (LLaMA-3 tiktoken style)

    Parameters
    ----------
    token_ids : 1-D tensor of token IDs
    tokenizer : HuggingFace-compatible tokenizer
    target_year : if set, only return positions for this specific year

    Returns
    -------
    Sorted list of 0-indexed token positions.
    """
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    positions: set[int] = set()

    # ── Primary: raw subword strings, strip word-start markers ────────────────
    # convert_ids_to_tokens returns e.g. "▁2021" or "Ġ2021"; decode() does
    # post-processing that can obscure the bare digits for some tokenizers.
    try:
        raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        raw_tokens = None

    if raw_tokens:
        for i, raw in enumerate(raw_tokens):
            if not raw:
                continue
            # strip SentencePiece (▁ U+2581) and GPT-2 BPE (Ġ U+0120) prefixes
            clean = raw.lstrip("\u2581\u0120").strip()
            if YEAR_PAT.fullmatch(clean):
                if target_year is None or int(clean) == target_year:
                    positions.add(i)

    # ── Fallback: decode each token individually ───────────────────────────────
    decoded = [tokenizer.decode([tid]) for tid in ids]

    if not positions:
        for i, tok_str in enumerate(decoded):
            clean = tok_str.strip()
            if YEAR_PAT.fullmatch(clean):
                if target_year is None or int(clean) == target_year:
                    positions.add(i)

    # ── Multi-token fallback: years split across two adjacent tokens ───────────
    # e.g.  "▁20" + "21"  →  "2021"
    for i in range(len(decoded) - 1):
        if i in positions or (i + 1) in positions:
            continue
        combined = (decoded[i] + decoded[i + 1]).strip()
        m = YEAR_PAT.search(combined)
        if m:
            year_val = int(m.group())
            if target_year is None or year_val == target_year:
                positions.update([i, i + 1])

    return sorted(positions)


# ── Answer matching ──────────────────────────────────────────────────────────

def check_match(generated: str, expected: str) -> bool:
    """Case-insensitive answer matching with two-tier logic.

    Tier 1 (strict): expected is a substring of generated.
    Tier 2 (loose) : the first token of expected (usually the first name or
                     a key entity word) appears in generated.  Handles title
                     variants like "Baron McFall" vs "Lord McFall".
    """
    gen_lower = generated.lower().strip()
    exp_lower = expected.lower().strip()
    if exp_lower in gen_lower:
        return True
    # loose: first whitespace-delimited word that is ≥4 chars (skip short titles)
    first_word = next((w for w in exp_lower.split() if len(w) >= 4), "")
    return bool(first_word and first_word in gen_lower)


def generate_answer(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    """Greedy-decode a short answer from *prompt*."""
    input_ids = model.to_tokens(prompt, prepend_bos=False)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            verbose=False,
        )
    new_ids = output_ids[0, input_ids.shape[1]:]
    return model.tokenizer.decode(new_ids, skip_special_tokens=True)
