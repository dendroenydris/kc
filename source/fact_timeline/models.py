"""Data models for the two-layer temporal dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Layer 1 — Fact-State Layer
# ---------------------------------------------------------------------------

@dataclass
class YearState:
    """The observed state of a (subject, property) pair in one calendar year."""
    year: int
    objects: list[str]           # human-readable labels (canonical first)
    object_qids: list[str]       # Wikidata QIDs (empty if not entity-valued)
    changed_from_prev: bool      # True when objects differ from year-1
    evidence_text: str = ""      # supporting sentence or passage; auto-generated if no corpus
    source_url: str = ""         # URL of the source (Wikidata entity page, Wikipedia, etc.)


@dataclass
class FactTimeline:
    """Complete timeline for one (subject, property) pair.

    This is the atomic unit of the Fact-State Layer.  One FactTimeline can
    generate multiple Evaluation Instances (Layer 2) by choosing pairs of
    years where a real change occurred.

    Fields
    ------
    fact_id           : unique key  "<subject_qid>_<property_pid>"
    subject_qid       : Wikidata QID of the subject entity
    subject_label     : English label of the subject
    wikipedia_title   : English Wikipedia article title (empty if not available)
    property_pid      : Wikidata property ID (e.g. "P6")
    property_label    : English label of the property (e.g. "head of government")
    domain            : coarse domain tag from PROPERTY_DOMAINS in sparql.py
    year_start        : first year covered in `states`
    year_end          : last year covered in `states`
    states            : ordered list of YearState, one per year in [year_start, year_end]
    change_years      : years where the answer changed from the previous year
    n_changes         : len(change_years)
    distractors       : sibling values (same property, different subjects) useful as
                        hard negatives when building eval instances
    source            : provenance tag ("wikidata_sparql" | "chroknowbench" | ...)
    """
    fact_id: str
    subject_qid: str
    subject_label: str
    property_pid: str
    property_label: str
    domain: str
    year_start: int
    year_end: int
    states: list[YearState]
    change_years: list[int]
    n_changes: int
    wikipedia_title: str = ""
    distractors: list[str] = field(default_factory=list)
    source: str = "wikidata_sparql"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def objects_for_year(self, year: int) -> list[str]:
        """Return object labels for *year*, or [] if year out of range."""
        for s in self.states:
            if s.year == year:
                return s.objects
        return []

    def primary_object_for_year(self, year: int) -> Optional[str]:
        """Return the first (most prominent) object label, or None."""
        objs = self.objects_for_year(year)
        return objs[0] if objs else None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "FactTimeline":
        states = [YearState(**s) for s in d.pop("states")]
        return cls(states=states, **d)


# ---------------------------------------------------------------------------
# Layer 2 — Evaluation Instance Layer  (placeholder; full builder in eval_instance/)
# ---------------------------------------------------------------------------

@dataclass
class EvalInstance:
    """One temporal conflict evaluation sample.

    Derived from a FactTimeline by picking two years (t_old, t_new) where
    the answer changed.  This is the format consumed by model inference,
    behavioral eval, and TATM mechanistic analysis.

    Fields
    ------
    instance_id           : globally unique sample ID
    fact_id               : foreign key → FactTimeline.fact_id
    subject_qid           : Wikidata QID
    subject_label         : English label
    property_pid          : property ID
    property_label        : property label
    domain                : domain tag
    t_old                 : earlier year (parametric memory anchor)
    t_new                 : later  year (context update target)
    time_gap_years        : t_new - t_old  (enables stratified evaluation)
    answer_old            : primary object label at t_old
    answer_new            : primary object label at t_new
    answer_old_qid        : Wikidata QID at t_old
    answer_new_qid        : Wikidata QID at t_new
    evidence_old          : Wikipedia sentence proving answer_old (from Layer 1 YearState)
    evidence_new          : Wikipedia sentence proving answer_new (from Layer 1 YearState)
    provenance_old        : source URL / revision tag for evidence_old
    provenance_new        : source URL / revision tag for evidence_new
    distractors           : hard negative labels
    question              : natural-language question (generated)
    context               : text passage shown to the model during evaluation.
                            For B-type instances this is evidence_new (real Wikipedia),
                            optionally modified (weakened year cue, adversarial, etc.).
                            Empty for A-type (no context) and C3 (baseline).
    context_strength      : "strong" | "weak" | "none"
    has_explicit_time_cue : whether context/question contains explicit year
    time_expression       : the time phrase used, e.g. "As of 2015"
    task_type             : "temporal_recall" | "context_override" | "ablation"
    gold_preference       : "use_context" | "use_memory"
    conflict_type         : always "temporal_context_memory" for this dataset
    source                : provenance tag
    """
    instance_id: str
    fact_id: str
    subject_qid: str
    subject_label: str
    property_pid: str
    property_label: str
    domain: str
    t_old: int
    t_new: int
    time_gap_years: int
    answer_old: str
    answer_new: str
    answer_old_qid: str
    answer_new_qid: str
    evidence_old: str
    evidence_new: str
    provenance_old: str
    provenance_new: str
    distractors: list[str]
    question: str
    context: str
    context_strength: str
    has_explicit_time_cue: bool
    time_expression: str
    task_type: str
    gold_preference: str
    conflict_type: str = "temporal_context_memory"
    source: str = "wikidata_sparql"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalInstance":
        return cls(**d)
