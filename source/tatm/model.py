"""Model loading, prompt formatting, and tokenization utilities for TATM."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
from transformer_lens import HookedTransformer

YEAR_PAT = re.compile(r"\b(19|20)\d{2}\b")


# ── Phi-3 compatibility patch ────────────────────────────────────────────────

def _patch_phi3_rope_scaling(model_name: str) -> None:
    """Pre-download and patch Phi-3 modeling file before TransformerLens imports it.

    Timing fix: we call get_cached_module_file() to pull modeling_phi3.py to
    disk FIRST, then patch it, then clear sys.modules so the next import picks
    up the patched version.  Without this step the file is absent when we look
    for it and TL downloads+imports the unpatched version on its own.

    Two bugs fixed in the cached file:
      Bug 1 – KeyError 'type': old code uses rope_scaling["type"] but newer
              configs store the key as "rope_type".
      Bug 2 – ValueError for 'longrope'/'su': old code only handles "linear"
              and "dynamic".  We fall back to standard Phi3RotaryEmbedding
              (correct for the 4K context window, harmless for MI experiments).
              A regex captures the exact indentation of the raise line so the
              replacement is always syntactically valid.
    """
    import importlib
    import sys

    cache_root = (
        Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    )

    # ── Step 1: ensure the file is on disk ────────────────────────────────
    targets = list(cache_root.glob("**/modeling_phi3.py"))
    if not targets:
        try:
            from transformers.dynamic_module_utils import get_cached_module_file
            local = get_cached_module_file(
                model_name,
                "modeling_phi3.py",
                trust_remote_code=True,
            )
            targets = [Path(local)]
        except Exception as exc:
            print(f"  [patch] Warning: could not pre-download modeling_phi3.py: {exc}")
            return

    # ── Step 2: apply fixes to every copy found ───────────────────────────
    raise_pat = re.compile(
        r'^( *)raise ValueError\(f"Unknown RoPE scaling type \{scaling_type\}"\)',
        re.MULTILINE,
    )

    for p in targets:
        text = p.read_text(encoding="utf-8")
        changed = False

        # Fix 1: key name lookup
        old1 = 'scaling_type = self.config.rope_scaling["type"]'
        new1 = (
            'scaling_type = (self.config.rope_scaling.get("type") or '
            'self.config.rope_scaling.get("rope_type") or "unknown")'
        )
        if old1 in text:
            text = text.replace(old1, new1)
            changed = True

        # Fix 2: unknown type → standard RoPE (regex preserves actual indentation)
        if raise_pat.search(text):
            def _repl(m: re.Match) -> str:
                ind = m.group(1)      # exact whitespace of the raise line
                sub = ind + "    "    # one extra indent level for arguments
                return (
                    f"{ind}# patched: fall back to standard RoPE for unknown types\n"
                    f"{ind}self.rotary_emb = Phi3RotaryEmbedding(\n"
                    f"{sub}self.head_dim,\n"
                    f"{sub}max_position_embeddings=self.max_position_embeddings,\n"
                    f"{sub}base=self.rope_theta,\n"
                    f"{ind})"
                )
            text = raise_pat.sub(_repl, text)
            changed = True

        if changed:
            p.write_text(text, encoding="utf-8")
            print(f"  [patch] {p.name} in …/{p.parent.name[:40]}")

    # ── Step 3: clear any already-imported version from memory ───────────
    for key in list(sys.modules.keys()):
        if "modeling_phi3" in key:
            del sys.modules[key]
    importlib.invalidate_caches()


# ── Model loading ────────────────────────────────────────────────────────────

#  Models that require trust_remote_code=True
_TRUST_REMOTE_CODE_PATTERNS = ("phi-3", "phi3", "falcon", "starcoder")


def _needs_trust_remote_code(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(p in name_lower for p in _TRUST_REMOTE_CODE_PATTERNS)


def load_model(
    model_name: str,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
) -> HookedTransformer:
    """Load a HookedTransformer with sensible defaults for TATM experiments.

    device="auto" selects CUDA → MPS → CPU in order of availability.
    dtype defaults to float32 for MPS/CPU (float16 is unstable on MPS).
    Handles Phi-3 rope_scaling compatibility automatically.
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

    # Model-specific pre-loading fixes
    name_lower = model_name.lower()
    if "phi-3" in name_lower or "phi3" in name_lower:
        _patch_phi3_rope_scaling(model_name)

    extra_kwargs = {}
    if _needs_trust_remote_code(model_name):
        extra_kwargs["trust_remote_code"] = True

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        **extra_kwargs,
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


def _collect_eos_ids(tokenizer) -> list[int]:
    """Return all token IDs that should stop generation for this model."""
    eos: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    for marker in ("<|end|>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>"):
        vocab = tokenizer.get_vocab()
        if marker in vocab:
            tid = vocab[marker]
            if tid not in eos:
                eos.append(tid)
    return eos


def _clean_generated(text: str) -> str:
    """Extract the first clean answer phrase from raw model output."""
    text = text.strip()
    # If model echoed the "Answer:" prefix, extract what follows
    m = re.search(r"(?i)Answer\s*:\s*([^\n\r]+)", text)
    if m:
        text = m.group(1)
    else:
        text = text.split("\n")[0]
    # Truncate at special-token remnants, sentence boundary, or prompt echoes
    for stop in ("<|", "[", "Instruction"):
        text = text.split(stop)[0]
    text = text.split(".")[0]
    return text.strip()


def generate_answer(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    """Greedy-decode a short answer from *prompt*.

    We bypass TL's model.generate() entirely because TL may prepend a BOS
    token to an already-formatted tensor, corrupting the sequence and causing
    degenerate repetition (e.g. "Has Has Has…").

    Instead we run a manual token-by-token loop:
      1. Encode the full prompt with the HF tokenizer (handles Phi-3 special
         tokens like <|end|> correctly).
      2. Call model(ids, prepend_bos=False) to get logits for each step.
      3. Argmax → append → repeat until EOS or max_new_tokens.
    """
    tok = model.tokenizer
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    current_ids = enc["input_ids"].to(model.cfg.device)

    eos_ids = set(_collect_eos_ids(tok))
    generated: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # prepend_bos=False: the prompt is already fully formatted
            logits = model(current_ids, prepend_bos=False)   # [1, seq, vocab]
            next_id = int(logits[0, -1, :].argmax())
            if next_id in eos_ids:
                break
            generated.append(next_id)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_id]], device=current_ids.device)],
                dim=1,
            )

    raw = tok.decode(generated, skip_special_tokens=True)
    return _clean_generated(raw)
