from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "x"


def _sanitize_question(template: str, subject: str) -> str:
    q = template.replace("{subject}", subject).replace("{time}", "").strip()
    q = re.sub(r"\s+", " ", q)
    q = q.replace(" ,", ",")
    q = re.sub(r"\bin year\b", "", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"\s+", " ", q).strip(", ").strip()
    if not q.endswith("?"):
        q = f"{q}?"
    return q


def _year_to_iso_start(year: int) -> str:
    return datetime(year, 1, 1).strftime("%Y-%m-%dT00:00:00Z")


def _year_gap_days(year_old: int, year_new: int) -> int:
    d0 = datetime(year_old, 1, 1)
    d1 = datetime(year_new, 1, 1)
    return (d1 - d0).days


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assign_numeric_ids(*record_lists: List[Dict[str, Any]]) -> None:
    i = 1
    for records in record_lists:
        for r in records:
            r["id"] = i
            i += 1


def _build_temporal_records(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    samples = data.get("samples", [])
    if not samples:
        return []

    template_q = (data.get("prompt_templates_zs") or ["For {subject}, what is the value?"])[0]
    template_e = (data.get("prompt_templates") or ["For {subject}, the value in {time} was"])[0]
    property_name = data.get("name", path.stem)
    property_pid = f"THP:{_slug(path.stem)}"
    property_label = f"temporal-head/{property_name}"

    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for s in samples:
        subject = str(s["subject"]).strip()
        obj = str(s["object"]).strip()
        year = int(str(s["time"]).strip())
        grouped.setdefault(subject, []).append((year, obj))

    records: List[Dict[str, Any]] = []
    for subject in sorted(grouped.keys()):
        rows = grouped[subject]
        rows = sorted(rows, key=lambda x: x[0])
        for idx in range(1, len(rows)):
            y_old, a_old = rows[idx - 1]
            y_new, a_new = rows[idx]
            if a_old == a_new:
                continue
            question = _sanitize_question(template_q, subject)
            evidence_old = (
                template_e.replace("{subject}", subject)
                .replace("{time}", str(y_old))
                .strip()
            )
            evidence_new = (
                template_e.replace("{subject}", subject)
                .replace("{time}", str(y_new))
                .strip()
            )
            if not evidence_old.endswith("."):
                evidence_old = f"{evidence_old}."
            if not evidence_new.endswith("."):
                evidence_new = f"{evidence_new}."
            evidence_old = f"{evidence_old} {a_old}"
            evidence_new = f"{evidence_new} {a_new}"

            records.append(
                {
                    "id": 0,
                    "question": question,
                    "as_of_mode": "timestamp",
                    "t_old": _year_to_iso_start(y_old),
                    "t_new": _year_to_iso_start(y_new),
                    "time_gap_days": _year_gap_days(y_old, y_new),
                    "answer_old": a_old,
                    "answer_new": a_new,
                    "evidence_old": evidence_old,
                    "evidence_new": evidence_new,
                    "subject_qid": f"TH:{_slug(subject)}",
                    "property_pid": property_pid,
                    "subject_label": subject,
                    "property_label": property_label,
                    "value_old_qid": f"THV:{_slug(a_old)}",
                    "value_new_qid": f"THV:{_slug(a_new)}",
                    "provenance_old": (
                        f"temporal-head:Temporal/{path.name}|subject={subject}|time={y_old}"
                    ),
                    "provenance_new": (
                        f"temporal-head:Temporal/{path.name}|subject={subject}|time={y_new}"
                    ),
                    "wikipedia_title": subject,
                }
            )
    return records


def _build_invariant_records(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    samples = data.get("samples", [])
    if not samples:
        return []

    template_q = (data.get("prompt_templates_zs") or ["For {subject}, what is the value?"])[0]
    template_e = (data.get("prompt_templates") or ["For {subject}, the value is"])[0]
    property_name = data.get("name", path.stem)
    property_pid = f"THP:{_slug(path.stem)}"
    property_label = f"temporal-head/{property_name}"

    records: List[Dict[str, Any]] = []
    for idx, s in enumerate(samples):
        subject = str(s["subject"]).strip()
        answer = str(s["object"]).strip()
        question = _sanitize_question(template_q, subject)

        evidence = template_e.replace("{subject}", subject).strip()
        if not evidence.endswith("."):
            evidence = f"{evidence}."
        evidence = f"{evidence} {answer}"

        records.append(
            {
                "id": 0,
                "question": question,
                "as_of_mode": "proxy_static",
                "t_old": "STATIC",
                "t_new": "STATIC",
                "time_gap_days": 0,
                "answer_old": answer,
                "answer_new": answer,
                "evidence_old": evidence,
                "evidence_new": evidence,
                "subject_qid": f"TH:{_slug(subject)}",
                "property_pid": property_pid,
                "subject_label": subject,
                "property_label": property_label,
                "value_old_qid": f"THV:{_slug(answer)}",
                "value_new_qid": f"THV:{_slug(answer)}",
                "provenance_old": f"temporal-head:Invariant/{path.name}|subject={subject}",
                "provenance_new": f"temporal-head:Invariant/{path.name}|subject={subject}",
                "wikipedia_title": None,
            }
        )
    return records


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert temporal-head data to temporal_kc style JSONL."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("data/external/temporal-head"),
        help="temporal-head root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for JSONL files.",
    )
    args = parser.parse_args()

    input_root = args.input_root
    temporal_files = sorted((input_root / "Temporal").glob("*.json"))
    invariant_files = sorted((input_root / "Invariant").glob("*.json"))

    temporal_records: List[Dict[str, Any]] = []
    invariant_records: List[Dict[str, Any]] = []

    for p in temporal_files:
        temporal_records.extend(_build_temporal_records(p))
    for p in invariant_files:
        invariant_records.extend(_build_invariant_records(p))

    _assign_numeric_ids(temporal_records, invariant_records)

    temporal_out = args.output_dir / "temporal_head_temporal_kc_temporal.jsonl"
    invariant_out = args.output_dir / "temporal_head_temporal_kc_invariant.jsonl"
    merged_out = args.output_dir / "temporal_head_temporal_kc_all.jsonl"

    _write_jsonl(temporal_records, temporal_out)
    _write_jsonl(invariant_records, invariant_out)
    _write_jsonl([*temporal_records, *invariant_records], merged_out)

    print(f"Temporal records: {len(temporal_records)} -> {temporal_out}")
    print(f"Invariant records: {len(invariant_records)} -> {invariant_out}")
    print(
        f"All records: {len(temporal_records) + len(invariant_records)} -> {merged_out}"
    )


if __name__ == "__main__":
    main()

