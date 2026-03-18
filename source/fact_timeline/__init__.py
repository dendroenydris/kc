"""fact_timeline — self-contained two-layer temporal knowledge dataset builder.

Layer 1  Fact-State Layer
    Full year-by-year timeline for each (subject, property) pair, with
    Wikipedia evidence fetched from historical revisions.
    Key classes : FactTimeline, YearState
    Key modules : sparql, builder, wiki_evidence

Layer 2  Evaluation Instance Layer
    11 evaluation instances per FactTimeline covering three task families:
      A  Temporal Recall (A1–A4)
      B  Context-Memory Conflict (B1–B4)
      C  Mechanistic / Ablation Probes (C1–C3)
    Key classes : EvalInstance
    Key modules : eval_builder

External dependencies
---------------------
    requests           HTTP
    mwparserfromhell   wikitext → plain text  (pip install mwparserfromhell)
    tqdm               progress bars          (pip install tqdm, optional)
"""
from fact_timeline.models import YearState, FactTimeline, EvalInstance
from fact_timeline.builder import build_timelines_for_property
from fact_timeline.eval_builder import build_eval_instances
from fact_timeline.wiki_evidence import enrich_timelines

__all__ = [
    # models
    "YearState",
    "FactTimeline",
    "EvalInstance",
    # layer 1
    "build_timelines_for_property",
    "enrich_timelines",
    # layer 2
    "build_eval_instances",
]
