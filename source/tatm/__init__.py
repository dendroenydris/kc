"""TATM — Temporal Arbitration via Time-State Mediation.

Diagnostic framework for analyzing temporal knowledge conflicts in LLMs.
Part of the three-phase pipeline:

    Phase 1  RevisionReplayQA  (data foundation — see fact_timeline/)
    Phase 2  TATM              (mechanistic diagnosis — this package)
    Phase 3  Align-then-Answer (inference-time intervention — TBD)

Submodules
----------
model       Model loading, prompt formatting, tokenization utilities
hooks       TransformerLens hook operations (attention extraction, knockout)
sat_probe   SAT Probe for F1 diagnosis (attention → logistic regression)
"""
