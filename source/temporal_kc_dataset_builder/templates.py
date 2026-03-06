from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class QuestionTemplate:
    pid: str
    template: str  # must include {subject}; may include {date}


TIME_IMPLICIT_TEMPLATES: Dict[str, QuestionTemplate] = {
    # P6: head of government
    "P6": QuestionTemplate(pid="P6", template="Who is the head of government of {subject}?"),
    # P35: head of state
    "P35": QuestionTemplate(pid="P35", template="Who is the head of state of {subject}?"),
    # P169: chief executive officer
    "P169": QuestionTemplate(pid="P169", template="Who is the chief executive officer of {subject}?"),
}


TIME_EXPLICIT_TEMPLATES: Dict[str, QuestionTemplate] = {
    # P6: head of government
    "P6": QuestionTemplate(pid="P6", template="Who was the head of government of {subject} as of {date}?"),
    # P35: head of state
    "P35": QuestionTemplate(pid="P35", template="Who was the head of state of {subject} as of {date}?"),
    # P169: chief executive officer
    "P169": QuestionTemplate(pid="P169", template="Who was the chief executive officer of {subject} as of {date}?"),
}


def render_question(
    *,
    pid: str,
    subject_label: str,
    as_of_date: str,
    property_label: str,
    include_time_in_question: bool = False,
) -> str:
    templates = TIME_EXPLICIT_TEMPLATES if include_time_in_question else TIME_IMPLICIT_TEMPLATES
    qt = templates.get(pid)
    if qt is None:
        if include_time_in_question:
            return f"What is the {property_label} of {subject_label} as of {as_of_date}?"
        return f"What is the {property_label} of {subject_label}?"
    return qt.template.format(subject=subject_label, date=as_of_date)

