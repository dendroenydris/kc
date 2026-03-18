"""Build Layer-2 EvalInstances from a Layer-1 FactTimeline.

Evidence flow
-------------
Layer 1 stores a real Wikipedia evidence sentence in every YearState
(YearState.evidence_text, YearState.source_url).  Layer 2 carries those
sentences forward as:

    EvalInstance.evidence_old  ← YearState(t_old).evidence_text
    EvalInstance.evidence_new  ← YearState(t_new).evidence_text
    EvalInstance.provenance_old/new ← YearState.source_url

The `context` field shown to the model during evaluation is derived from
`evidence_new`, optionally modified to vary the strength/presence of the
time cue:

    B1  evidence_new unchanged (strong: already contains year in provenance tag)
    B2  evidence_new, question is implicit ("Currently, …?")
    B3  evidence_new with year stripped from the sentence body ("recent reports…")
    B4  evidence_new, question has no time cue at all

Layer A and C3 leave `context` empty (pure parametric memory probing).
C1/C2 inject deliberately wrong or mislabelled context.

Three evaluation families
--------------------------
A  Temporal Recall — no context, tests parametric memory
     A1  explicit year   "As of {t_new}, …?"
     A2  implicit        "Currently, …?"
     A3  yes/no correct  "Was X {answer_new} in {t_new}?"  → expected: Yes
     A4  yes/no stale    "Was X still {answer_old} in {t_new}?"  → expected: No

B  Context-Memory Conflict — evidence_new as context
     B1  full evidence_new  + explicit question year
     B2  full evidence_new  + implicit question
     B3  year-stripped evidence_new + explicit question year
     B4  year-stripped evidence_new + no-cue question

C  Mechanistic Probes / Ablations
     C1  adversarial context (old answer labelled as current) → model should use memory
     C2  mislabelled-year context (new answer, old year label) → can model recover?
     C3  plain no-cue question, no context → pure baseline
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

from fact_timeline.models import EvalInstance, FactTimeline, YearState

# ── Relation language templates ───────────────────────────────────────────────

_REL_LANG: dict[str, dict[str, str]] = {
    "position held": {
        "q_phrase":     "what position did {subject} hold",
        "verb_past":    "{subject} held the position of {obj}",
        "verb_pres":    "{subject} holds the position of {obj}",
        "verb_changed": "{subject}'s position changed to {obj}",
    },
    "member of sports team": {
        "q_phrase":     "which sports team was {subject} a member of",
        "verb_past":    "{subject} was a member of {obj}",
        "verb_pres":    "{subject} is a member of {obj}",
        "verb_changed": "{subject} joined {obj}",
    },
    "employer": {
        "q_phrase":     "who was {subject}'s employer",
        "verb_past":    "{subject} was employed by {obj}",
        "verb_pres":    "{subject} is employed by {obj}",
        "verb_changed": "{subject}'s employer became {obj}",
    },
    "educated at": {
        "q_phrase":     "where was {subject} educated",
        "verb_past":    "{subject} was educated at {obj}",
        "verb_pres":    "{subject} is studying at {obj}",
        "verb_changed": "{subject} moved to study at {obj}",
    },
    "officeholder": {
        "q_phrase":     "who was the officeholder associated with {subject}",
        "verb_past":    "the officeholder of {subject} was {obj}",
        "verb_pres":    "the officeholder of {subject} is {obj}",
        "verb_changed": "the officeholder of {subject} changed to {obj}",
    },
    "director / manager": {
        "q_phrase":     "who was the director or manager of {subject}",
        "verb_past":    "{subject} was directed by {obj}",
        "verb_pres":    "{subject} is directed by {obj}",
        "verb_changed": "{subject}'s director changed to {obj}",
    },
    "member of": {
        "q_phrase":     "which organisation was {subject} a member of",
        "verb_past":    "{subject} was a member of {obj}",
        "verb_pres":    "{subject} is a member of {obj}",
        "verb_changed": "{subject} became a member of {obj}",
    },
    "chief executive officer": {
        "q_phrase":     "who was the CEO of {subject}",
        "verb_past":    "the CEO of {subject} was {obj}",
        "verb_pres":    "the CEO of {subject} is {obj}",
        "verb_changed": "{subject}'s CEO changed to {obj}",
    },
    "head of government": {
        "q_phrase":     "who was the head of government of {subject}",
        "verb_past":    "the head of government of {subject} was {obj}",
        "verb_pres":    "the head of government of {subject} is {obj}",
        "verb_changed": "the head of government of {subject} changed to {obj}",
    },
    "head of state": {
        "q_phrase":     "who was the head of state of {subject}",
        "verb_past":    "the head of state of {subject} was {obj}",
        "verb_pres":    "the head of state of {subject} is {obj}",
        "verb_changed": "the head of state of {subject} changed to {obj}",
    },
    "head coach": {
        "q_phrase":     "who was the head coach of {subject}",
        "verb_past":    "the head coach of {subject} was {obj}",
        "verb_pres":    "the head coach of {subject} is {obj}",
        "verb_changed": "{subject}'s head coach changed to {obj}",
    },
    "chairperson": {
        "q_phrase":     "who was the chairperson of {subject}",
        "verb_past":    "the chairperson of {subject} was {obj}",
        "verb_pres":    "the chairperson of {subject} is {obj}",
        "verb_changed": "the chairperson of {subject} changed to {obj}",
    },
}

_REL_LANG_DEFAULT: dict[str, str] = {
    "q_phrase":     "what is the {rel} of {subject}",
    "verb_past":    "{subject}'s {rel} was {obj}",
    "verb_pres":    "{subject}'s {rel} is {obj}",
    "verb_changed": "{subject}'s {rel} changed to {obj}",
}

# Matches the whole temporal phrase, e.g.:
#   "As of 2023,"  "In 2010,"  "Since 2005–06"  "2021"  "July2006"
# The leading preposition clause is optional and consumed when present.
_YEAR_RE = re.compile(
    r"(?:(?:As of|In|Since|From|Until|By)\s+)?(?<!\d)(19|20)\d{2}(?:[-–/]\d{2,4})?(?!\d),?",
    re.IGNORECASE,
)


def _lang(relation: str) -> dict[str, str]:
    return _REL_LANG.get(relation, _REL_LANG_DEFAULT)


def _fill(template: str, *, subject: str = "", obj: str = "", rel: str = "") -> str:
    return (template
            .replace("{subject}", subject)
            .replace("{obj}", obj)
            .replace("{rel}", rel))


# ── Evidence helpers ──────────────────────────────────────────────────────────

def _state_for_year(tl: FactTimeline, year: int) -> Optional[YearState]:
    for s in tl.states:
        if s.year == year:
            return s
    return None


def _evidence_for_year(tl: FactTimeline, year: int) -> tuple[str, str]:
    """Return (evidence_text, source_url) for *year*, falling back to synthetic."""
    state = _state_for_year(tl, year)
    if state and state.evidence_text:
        return state.evidence_text, state.source_url or ""
    # Synthetic fallback (e.g. ChroKnowBench source with no Wikipedia)
    obj = tl.primary_object_for_year(year) or ""
    synth = f"As of {year}, {tl.subject_label}'s {tl.property_label} was {obj}."
    url   = f"https://www.wikidata.org/wiki/{tl.subject_qid}" if tl.subject_qid else ""
    return synth, url


def _strip_years(text: str) -> str:
    """Delete temporal phrases from evidence text (for B3/B4 weak-context variants).

    Deletes whole phrases such as "As of 2023," and "In 2010," rather than
    substituting a placeholder like "recently", which would itself be a
    temporal cue (implying recency).
    """
    result = _YEAR_RE.sub("", text)
    result = re.sub(r"\s{2,}", " ", result)   # collapse gaps left by deletion
    result = re.sub(r"\s+,", ",", result)     # "Labour ," → "Labour,"
    result = re.sub(r"\s+\.", ".", result)    # "January ." → "January."
    result = re.sub(r"\s+\)", ")", result)    # "born )" → "born)"
    return result.strip()


# ── Change-pair selection ─────────────────────────────────────────────────────

def _has_real_evidence(tl: FactTimeline, year: int) -> bool:
    """True if the YearState for *year* has a real Wikipedia URL (not synthetic)."""
    state = _state_for_year(tl, year)
    return bool(state and "oldid=" in (state.source_url or ""))


def _pick_change_pair(tl: FactTimeline) -> Optional[tuple[int, int]]:
    """Pick the best (t_old, t_new) change pair.

    Scoring: prefer the pair with the longest stable run before the change
    (makes parametric memory anchor strong) and a gap ≥ 1 year.

    Key rule: t_new must be a *stable* year — one where answer_new is the
    sole occupant of the fact.  Transition years (e.g. US 2021 where both
    Trump and Biden appear) are ambiguous and skipped for t_new.
    """
    year_to_obj: dict[int, str] = {
        s.year: s.objects[0] for s in tl.states if s.objects
    }
    # Mark years as stable (exactly one answer) vs ambiguous (transition)
    year_is_stable: dict[int, bool] = {
        s.year: (len(s.objects) == 1) for s in tl.states if s.objects
    }
    sorted_years = sorted(year_to_obj)
    if len(sorted_years) < 2:
        return None

    best_pair: Optional[tuple[int, int]] = None
    best_score = -1

    # Build a lookup: year → set of objects (for multi-object transition years)
    year_to_obj_set: dict[int, set[str]] = {
        s.year: set(s.objects) for s in tl.states if s.objects
    }

    for change_year in tl.change_years:
        if change_year not in year_to_obj:
            continue

        # Determine the *incoming* new object:
        # the object that appears in change_year but was absent the year before.
        prev_years = [y for y in sorted_years if y < change_year]
        prev_obj_set = year_to_obj_set.get(prev_years[-1], set()) if prev_years else set()
        new_objects = [
            obj for obj in (year_to_obj_set.get(change_year) or set())
            if obj not in prev_obj_set
        ]
        # Fallback: if nothing looks "new" just take objects[0]
        change_obj = new_objects[0] if new_objects else year_to_obj[change_year]

        # Find the first stable year >= change_year where the new object is
        # the sole holder.  Prefer stable over the raw transition year.
        stable_new_cands = [
            y for y in sorted_years
            if y >= change_year
            and year_to_obj[y] == change_obj
            and year_is_stable.get(y, False)
        ]
        if not stable_new_cands:
            # No stable year found — fall back to the transition year itself.
            stable_new_cands = [change_year]

        t_new     = stable_new_cands[0]
        t_new_obj = year_to_obj[t_new]

        old_cands = [y for y in sorted_years if y < t_new and year_to_obj[y] != t_new_obj]
        if not old_cands:
            continue

        # Prefer t_old years that have real Wikipedia evidence (source_url has oldid).
        # Fall back to the most recent candidate if none have real evidence.
        real_cands = [y for y in old_cands if _has_real_evidence(tl, y)]
        t_old     = real_cands[-1] if real_cands else old_cands[-1]
        t_old_obj = year_to_obj[t_old]
        gap       = t_new - t_old

        if gap < 1:
            continue

        stable_run = sum(1 for y in sorted_years if y <= t_old and year_to_obj.get(y) == t_old_obj)
        score = stable_run * 1.5 + gap * 0.5

        if score > best_score:
            best_score = score
            best_pair  = (t_old, t_new)

    return best_pair


# ── Instance ID ───────────────────────────────────────────────────────────────

def _iid(fact_id: str, t_old: int, t_new: int, layer: str) -> str:
    h = hashlib.sha256(f"{fact_id}_{t_old}_{t_new}_{layer}".encode()).hexdigest()[:10]
    return f"{layer}_{h}"


# ── Question builders ─────────────────────────────────────────────────────────

def _q_explicit(subject: str, t_new: int, lang: dict, rel: str = "") -> str:
    return f"As of {t_new}, {_fill(lang['q_phrase'], subject=subject, rel=rel)}?"

def _q_implicit(subject: str, lang: dict, rel: str = "") -> str:
    return f"Currently, {_fill(lang['q_phrase'], subject=subject, rel=rel)}?"

def _q_yn_correct(subject: str, answer_new: str, t_new: int, lang: dict) -> str:
    return f"In {t_new}, {_fill(lang['verb_past'], subject=subject, obj=answer_new)}? (Yes or No)"

def _q_yn_stale(subject: str, answer_old: str, t_new: int, lang: dict) -> str:
    return f"In {t_new}, {_fill(lang['verb_past'], subject=subject, obj=answer_old)}? (Yes or No)"

def _q_plain(subject: str, lang: dict, rel: str = "") -> str:
    return f"What is {_fill(lang['q_phrase'], subject=subject, rel=rel)}?"


# ── Adversarial / mislabelled context builders ────────────────────────────────

def _ctx_adversarial(subject: str, obj_old: str, t_new: int, lang: dict) -> str:
    """Falsely claims old answer is current as of t_new (C1)."""
    verb = _fill(lang["verb_pres"], subject=subject, obj=obj_old)
    return f"As of {t_new}, {verb}."

def _ctx_mislabelled(subject: str, obj_new: str, t_old: int, lang: dict) -> str:
    """Correct new answer but stamped with old year (C2)."""
    verb = _fill(lang["verb_changed"], subject=subject, obj=obj_new)
    return f"According to {t_old} records, {verb}."


# ── Main generator ────────────────────────────────────────────────────────────

def build_eval_instances(tl: FactTimeline) -> list[EvalInstance]:
    """Generate all 11 Layer A/B/C EvalInstances for one FactTimeline.

    Returns [] if no valid (t_old, t_new) pair can be found.
    """
    pair = _pick_change_pair(tl)
    if pair is None:
        return []

    t_old, t_new = pair
    answer_old = tl.primary_object_for_year(t_old) or ""
    answer_new = tl.primary_object_for_year(t_new) or ""

    if not answer_old or not answer_new or answer_old == answer_new:
        return []

    # Pull real Wikipedia evidence from Layer 1
    ev_new, prov_new = _evidence_for_year(tl, t_new)
    ev_old, prov_old = _evidence_for_year(tl, t_old)

    lang = _lang(tl.property_label)
    subj = tl.subject_label
    rel  = tl.property_label
    gap  = t_new - t_old

    # Distractor order is deterministic (sorted) so results are reproducible
    distr = sorted(tl.distractors)[:3]

    def _inst(
        layer: str,
        question: str,
        context: str,
        ctx_strength: str,
        has_cue: bool,
        time_expr: str,
        task_type: str,
        gold_pref: str,
    ) -> EvalInstance:
        return EvalInstance(
            instance_id=_iid(tl.fact_id, t_old, t_new, layer),
            fact_id=tl.fact_id,
            subject_qid=tl.subject_qid,
            subject_label=subj,
            property_pid=tl.property_pid,
            property_label=rel,
            domain=tl.domain,
            t_old=t_old,
            t_new=t_new,
            time_gap_years=gap,
            answer_old=answer_old,
            answer_new=answer_new,
            answer_old_qid="",
            answer_new_qid="",
            evidence_old=ev_old,
            evidence_new=ev_new,
            provenance_old=prov_old,
            provenance_new=prov_new,
            distractors=distr,
            question=question,
            context=context,
            context_strength=ctx_strength,
            has_explicit_time_cue=has_cue,
            time_expression=time_expr,
            task_type=task_type,
            gold_preference=gold_pref,
            conflict_type="temporal_context_memory",
            source=tl.source,
        )

    # B-type contexts: derived from real evidence_new
    # Strong = full evidence sentence (may contain a year naturally)
    # Weak   = year digits stripped → "recently" substituted
    ctx_b_strong = ev_new
    ctx_b_weak   = _strip_years(ev_new)

    q_explicit = _q_explicit(subj, t_new, lang, rel)
    q_implicit = _q_implicit(subj, lang, rel)
    q_yn_ok    = _q_yn_correct(subj, answer_new, t_new, lang)
    q_yn_stale = _q_yn_stale(subj, answer_old, t_new, lang)
    q_plain    = _q_plain(subj, lang, rel)

    ctx_adv  = _ctx_adversarial(subj, answer_old, t_new, lang)
    ctx_mis  = _ctx_mislabelled(subj, answer_new, t_old, lang)

    return [
        # ── A: Temporal Recall (no context) ─────────────────────────────────
        _inst("A1", q_explicit, "", "none", True,  f"As of {t_new}", "temporal_recall", "use_memory"),
        _inst("A2", q_implicit, "", "none", False, "currently",      "temporal_recall", "use_memory"),
        _inst("A3", q_yn_ok,    "", "none", True,  f"in {t_new}",    "temporal_recall", "use_memory"),
        _inst("A4", q_yn_stale, "", "none", True,  f"in {t_new}",    "temporal_recall", "use_memory"),

        # ── B: Context-Memory Conflict (context = real Wikipedia evidence) ───
        _inst("B1", q_explicit, ctx_b_strong, "strong", True,  f"As of {t_new}", "context_override", "use_context"),
        _inst("B2", q_implicit, ctx_b_strong, "strong", True,  f"As of {t_new}", "context_override", "use_context"),
        _inst("B3", q_explicit, ctx_b_weak,   "weak",   False, "recently",       "context_override", "use_context"),
        _inst("B4", q_plain,    ctx_b_weak,   "weak",   False, "",               "context_override", "use_context"),

        # ── C: Mechanistic Probes ────────────────────────────────────────────
        _inst("C1", q_explicit, ctx_adv,  "strong", True,  f"As of {t_new}", "ablation", "use_memory"),
        _inst("C2", q_explicit, ctx_mis,  "strong", True,  f"As of {t_old}", "ablation", "use_context"),
        _inst("C3", q_plain,    "",        "none",  False, "",               "ablation", "use_memory"),
    ]
