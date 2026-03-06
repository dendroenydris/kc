from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TemporalConflictExample:
    id: str
    question: str

    as_of_mode: str  # "timestamp"
    t_old: str  # ISO 8601 date/time (UTC)
    t_new: str
    time_gap_days: int

    answer_old: str
    answer_new: str

    evidence_old: str
    evidence_new: str

    subject_qid: str
    property_pid: str

    subject_label: str
    property_label: str

    value_old_qid: str
    value_new_qid: str

    provenance_old: Optional[str] = None
    provenance_new: Optional[str] = None

    wikipedia_title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

