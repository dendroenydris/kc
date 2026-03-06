import json
import argparse
import gc
import torch
import numpy as np
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
from transformers import AutoConfig

"""
TATM (Temporal Arbitration via Time-State Mediation) Diagnosis Experiments
Based on the F1, F2, F3 Failure Taxonomy.

This script uses `transformer_lens` to perform Mechanistic Interpretability 
experiments (Activation Patching, Attention Knockout, Logit Lens, MLP Ablation)
to diagnose why LLMs fail on temporal knowledge conflicts.
"""

def load_data(filepath: str):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_logit_diff(logits, correct_token_id, incorrect_token_id):
    """
    Core Observation Metric (Logit Difference): 
    Δ = Logit(T_new) - Logit(T_old)
    """
    # Assuming logits shape: [batch, seq_len, vocab_size]
    final_token_logits = logits[0, -1, :]
    return final_token_logits[correct_token_id] - final_token_logits[incorrect_token_id]


def get_discriminative_token_ids(model: HookedTransformer, answer_new: str, answer_old: str):
    """
    Pick token IDs that actually differ between two candidate answers.
    This avoids the bug where both IDs become the same whitespace/shared-prefix token.
    """
    new_ids = model.to_tokens(" " + answer_new, prepend_bos=False)[0].tolist()
    old_ids = model.to_tokens(" " + answer_old, prepend_bos=False)[0].tolist()

    max_len = max(len(new_ids), len(old_ids))
    for i in range(max_len):
        n_tok = new_ids[i] if i < len(new_ids) else None
        o_tok = old_ids[i] if i < len(old_ids) else None
        if n_tok != o_tok:
            # Fallback safety for variable-length answers
            if n_tok is None:
                n_tok = new_ids[-1]
            if o_tok is None:
                o_tok = old_ids[-1]
            return n_tok, o_tok, i

    # Extremely rare: both tokenized sequences are identical
    # Return first token and let caller skip this sample.
    return new_ids[0], old_ids[0], -1


# ============================================================================
# Experiment 1: Diagnose F1 (Time not set)
# ============================================================================
def diagnose_f1(model: HookedTransformer, clean_prompt: str, corrupted_prompt: str, 
                t_new_id: int, t_old_id: int, temporal_heads: list, temporal_token_pos: int):
    """
    Hypothesis: The model fails because it doesn't activate the time state Z_t.
    Method: Activation Patching on Temporal Heads.
    """
    print(f"\n--- Running F1 Diagnosis (Activation Patching) ---")
    
    # 1. Run successful case (Clean) and cache only required activations
    layers_to_patch = list(set([h[0] for h in temporal_heads]))
    hook_names = {f"blocks.{l}.attn.hook_z" for l in layers_to_patch}
    with torch.inference_mode():
        _, clean_cache = model.run_with_cache(clean_prompt, names_filter=lambda name: name in hook_names)
    
    # 2. Run failed case (Corrupted) to get baseline diff
    with torch.inference_mode():
        corrupted_logits = model(corrupted_prompt)
        baseline_diff = get_logit_diff(corrupted_logits, t_new_id, t_old_id)
    print(f"Baseline Logit Diff (Corrupted): {baseline_diff.item():.4f}")

    # 3. Patching Function
    def patch_temporal_head_activation(activations, hook):
        # activations shape: [batch, pos, head_index, d_head]
        # We patch the activation at the specific temporal token position
        for head in temporal_heads:
            # We assume hook.name contains the layer info, e.g., 'blocks.L.attn.hook_z'
            layer = int(hook.name.split('.')[1])
            if head[0] == layer:
                head_idx = head[1]
                # Inject the clean activation into the corrupted run
                activations[0, temporal_token_pos, head_idx, :] = clean_cache[hook.name][0, temporal_token_pos, head_idx, :]
        return activations

    # Create hooks for all layers involved in temporal_heads
    hooks = [(f"blocks.{l}.attn.hook_z", patch_temporal_head_activation) for l in layers_to_patch]

    # 4. Run failed case with patched activations
    with torch.inference_mode():
        patched_logits = model.run_with_hooks(corrupted_prompt, fwd_hooks=hooks)
        patched_diff = get_logit_diff(patched_logits, t_new_id, t_old_id)
    print(f"Patched Logit Diff: {patched_diff.item():.4f}")
    
    # 5. Judgment
    if patched_diff > 0 and baseline_diff < 0:
        print("Verdict: F1 Confirmed. (Δ reversed from A to B after injecting temporal activation)")
        del clean_cache, corrupted_logits, patched_logits
        return True
    del clean_cache, corrupted_logits, patched_logits
    return False


# ============================================================================
# Experiment 2: Diagnose F2 (Time set but not routed)
# ============================================================================
def diagnose_f2(model: HookedTransformer, corrupted_prompt: str, 
                t_new_id: int, t_old_id: int, temporal_heads: list, 
                routing_heads: list, temporal_token_pos: int, final_token_pos: int):
    """
    Hypothesis: Time state exists, but the routing circuit from time to fact fails.
    Method: Path Knockout / Attention Knockout from Temporal Heads to Routing Heads.
    """
    print(f"\n--- Running F2 Diagnosis (Path Knockout) ---")
    
    # 1. Knockout Function: Cut off attention from Final Token to Temporal Token
    def knockout_attention_path(pattern, hook):
        # pattern shape: [batch, head_index, dest_pos, src_pos]
        for head in routing_heads:
            layer = int(hook.name.split('.')[1])
            if head[0] == layer:
                head_idx = head[1]
                # Set attention score from final token to temporal token to 0
                pattern[0, head_idx, final_token_pos, temporal_token_pos] = 0.0
        return pattern

    layers_to_knockout = list(set([h[0] for h in routing_heads]))
    hooks = [(f"blocks.{l}.attn.hook_pattern", knockout_attention_path) for l in layers_to_knockout]

    # 2. Run with path knockout
    with torch.inference_mode():
        knockout_logits = model.run_with_hooks(corrupted_prompt, fwd_hooks=hooks)
        knockout_diff = get_logit_diff(knockout_logits, t_new_id, t_old_id)
    
    print(f"Logit Diff after Routing Knockout: {knockout_diff.item():.4f}")
    
    # 3. Judgment
    # Real F2 evaluation requires ensuring F1 passed, but path is broken.
    # In a real experiment, we check if attention score is low and if injecting F1 didn't help.
    print("Verdict: If Attention Score is low and model still outputs A despite F1 fix, F2 Confirmed.")
    return True


# ============================================================================
# Experiment 3: Diagnose F3 (Routed but overridden)
# ============================================================================
def diagnose_f3(model: HookedTransformer, corrupted_prompt: str, 
                t_new_id: int, t_old_id: int, late_mlp_layers: list):
    """
    Hypothesis: The context is processed correctly, but deep parametric memory overrides it.
    Method: Logit Lens & Late-Layer MLP Mean Ablation.
    """
    print(f"\n--- Running F3 Diagnosis (Late-Layer MLP Ablation) ---")
    
    # 1. Baseline
    with torch.inference_mode():
        baseline_logits = model(corrupted_prompt)
        baseline_diff = get_logit_diff(baseline_logits, t_new_id, t_old_id)
    
    # [Logit Lens Analysis Omitted here for brevity, but you'd decode cache["resid_post", layer]]
    # Example:
    # for layer in range(model.cfg.n_layers):
    #     residual = cache["resid_post", layer][0, -1, :]
    #     layer_logits = model.unembed(model.ln_final(residual))
    #     print(f"Layer {layer} T_new prob vs T_old prob...")

    # 2. Mean Ablation Function for MLPs
    def mean_ablate_mlp(activations, hook):
        # activations shape: [batch, pos, d_model]
        # Replace MLP output with zeros (or dataset mean)
        activations[:] = 0.0
        return activations

    hooks = [(f"blocks.{l}.hook_mlp_out", mean_ablate_mlp) for l in late_mlp_layers]

    # 3. Run with late MLP ablation
    with torch.inference_mode():
        ablated_logits = model.run_with_hooks(corrupted_prompt, fwd_hooks=hooks)
        ablated_diff = get_logit_diff(ablated_logits, t_new_id, t_old_id)
    
    print(f"Baseline Logit Diff: {baseline_diff.item():.4f}")
    print(f"Ablated Logit Diff: {ablated_diff.item():.4f}")
    
    # 4. Judgment
    if ablated_diff > 0 and baseline_diff < 0:
        print("Verdict: F3 Confirmed. (Silencing deep MLPs restored the correct output B).")
        del baseline_logits, ablated_logits
        return True
    del baseline_logits, ablated_logits
    return False


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_hooked_model_with_phi3_compat(model_name: str, device: str) -> HookedTransformer:
    """
    Load HookedTransformer with a compatibility fix for some Phi-3 configs where
    rope_scaling may contain 'rope_type' but not 'type'.
    """
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        if "type" not in rope_scaling and "rope_type" in rope_scaling:
            rope_scaling["type"] = rope_scaling["rope_type"]
        elif "type" not in rope_scaling and ("short_factor" in rope_scaling or "long_factor" in rope_scaling):
            # Common fallback for Phi-3 long context scaling schema.
            rope_scaling["type"] = "longrope"
        cfg.rope_scaling = rope_scaling

    load_dtype = torch.float16 if device == "cuda" else torch.float32
    return HookedTransformer.from_pretrained(
        model_name,
        device=device,
        trust_remote_code=True,
        config=cfg,
        dtype=load_dtype,
    )

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TATM diagnosis experiments safely.")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument(
        "--dataset",
        default="/Users/celes/Documents/Projects/knowledge/code/data/processed/temporal_sample_20.jsonl",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--skip-f2f3", action="store_true", help="Only run F1 for lighter memory usage.")
    args = parser.parse_args()

    print("Initializing TATM Diagnostic Framework...")

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    device = args.device
    print(f"Loading model microsoft/Phi-3-mini-4k-instruct on {device}...")
    
    # We use a relatively small model for local testing
    model = load_hooked_model_with_phi3_compat(args.model, device)
    model.eval()
    
    dataset_path = args.dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_data(dataset_path)
    if args.max_samples > 0:
        dataset = dataset[: args.max_samples]

    # For Phi-3-mini-4k-instruct, there are 32 layers. 
    # Let's mock some heads as temporal/routing heads for the sake of the demonstration
    mock_temporal_heads = [(10, 5), (11, 2)] # (layer, head_index)
    mock_routing_heads = [(15, 0), (16, 7)]
    mock_late_mlp_layers = [28, 29, 30, 31]

    for i, sample in enumerate(dataset):
        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(dataset)}] Testing Sample ID: {sample['id']}")
        
        # Build prompt:
        # Context structure:
        # "In 2004, the defense expediture of China in billion USD was 38.0. In 2005, the defense expediture of China in billion USD was 42.0."
        context = f"{sample['evidence_old']} {sample['evidence_new']}"
        
        # Clean prompt asks about t_new
        clean_question = f"Question: As of {sample['t_new'][:4]}, {sample['question']}"
        clean_prompt = f"{context} {clean_question} Answer:"
        
        # Corrupted prompt asks about t_old
        corrupted_question = f"Question: As of {sample['t_old'][:4]}, {sample['question']}"
        corrupted_prompt = f"{context} {corrupted_question} Answer:"
        
        print(f"Corrupted Prompt: {corrupted_prompt}")
        
        # Identify answer tokens
        # We find the single token ID for the answers to calculate logit diff
        # In a real setup, we should ensure the answers map cleanly to single tokens or handle multi-token answers.
        def get_first_token(text):
            # Encode and skip BOS token if present
            tokens = model.to_tokens(text, prepend_bos=False)[0]
            return tokens[0].item() if len(tokens) > 0 else 0

        t_new_id, t_old_id, diff_pos = get_discriminative_token_ids(
            model, sample["answer_new"], sample["answer_old"]
        )
        if diff_pos == -1:
            print("Warning: answer tokenization identical for new/old answer, skipping sample.")
            continue
        
        print(f"Token ID for T_new ('{sample['answer_new']}'): {t_new_id}")
        print(f"Token ID for T_old ('{sample['answer_old']}'): {t_old_id}")
        print(f"Using discriminative token position within answer: {diff_pos}")

        # Find the position of the temporal token (the year) in the corrupted prompt
        # A simple hack for the demo is to find the index of the year token.
        corrupted_tokens = model.to_tokens(corrupted_prompt, prepend_bos=True)[0]
        year_token_id = get_first_token(sample['t_old'][:4])
        
        # Find position of year_token_id in corrupted_tokens
        temporal_token_pos = -1
        for pos, token in enumerate(corrupted_tokens):
            if token.item() == year_token_id:
                temporal_token_pos = pos
                # We can break or take the last occurrence (the one in the question)
        
        if temporal_token_pos == -1:
            print(f"Warning: Could not reliably find the year token '{sample['t_old'][:4]}' in the prompt. Skipping.")
            continue
            
        print(f"Found temporal token at position: {temporal_token_pos}")
        final_token_pos = len(corrupted_tokens) - 1

        # Run Diagnostics
        try:
            is_f1 = diagnose_f1(model, clean_prompt, corrupted_prompt, t_new_id, t_old_id, mock_temporal_heads, temporal_token_pos)
            if not args.skip_f2f3:
                _ = diagnose_f2(model, corrupted_prompt, t_new_id, t_old_id, mock_temporal_heads, mock_routing_heads, temporal_token_pos, final_token_pos)
                _ = diagnose_f3(model, corrupted_prompt, t_new_id, t_old_id, mock_late_mlp_layers)
            cleanup_memory()
        except Exception as e:
            print(f"Error running diagnostics for sample: {e}")
            cleanup_memory()
            continue
