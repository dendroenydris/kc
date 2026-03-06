from __future__ import annotations

import bz2
import gzip
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

from dateutil.parser import isoparse

from .schema import TemporalConflictExample
from .templates import render_question
from .text import find_evidence_snippet, wikitext_to_plaintext

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None  # type: ignore


PROPERTY_LABELS = {
    "P6": "head of government",
    "P35": "head of state",
    "P169": "chief executive officer",
}


@dataclass(frozen=True)
class Statement:
    value_qid: str
    start_time: Optional[str]
    end_time: Optional[str]


@dataclass(frozen=True)
class Candidate:
    subject_qid: str
    subject_label: str
    wikipedia_title: str
    property_pid: str
    property_label: str
    old_value_qid: str
    new_value_qid: str
    old_start: Optional[str]
    old_end: Optional[str]
    new_start: Optional[str]
    new_end: Optional[str]


def _open_text(path: str):
    p = str(path).lower()
    if p.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    if p.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _open_binary(path: str):
    p = str(path).lower()
    if p.endswith(".bz2"):
        return bz2.open(path, "rb")
    if p.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _normalize_wikidata_time(raw: str) -> str:
    # Example raw values: +2011-08-24T00:00:00Z, +1991-00-00T00:00:00Z
    s = raw.strip()
    if s.startswith("+"):
        s = s[1:]
    if len(s) >= 10:
        y, m, d = s[:4], s[5:7], s[8:10]
        if m == "00":
            m = "01"
        if d == "00":
            d = "01"
        s = f"{y}-{m}-{d}" + s[10:]
    return s


def _safe_parse_time(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return isoparse(raw)
    except Exception:
        try:
            return isoparse(_normalize_wikidata_time(raw))
        except Exception:
            return None


def _pick_time_within_interval(start: Optional[str], end: Optional[str], fallback: str) -> str:
    s = _safe_parse_time(start)
    e = _safe_parse_time(end)
    fb = _safe_parse_time(fallback) or datetime.now(timezone.utc)
    if s and e:
        if e <= s:
            return s.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        mid = s + (e - s) / 2
        return mid.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if s:
        return s.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if e:
        return e.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return fb.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _days_between(a: str, b: str) -> int:
    aa = _safe_parse_time(a) or datetime(1900, 1, 1, tzinfo=timezone.utc)
    bb = _safe_parse_time(b) or datetime(1900, 1, 1, tzinfo=timezone.utc)
    return abs((bb - aa).days)


def _iter_wikidata_entities(path: str) -> Iterator[dict]:
    with _open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s or s in {"[", "]"}:
                continue
            if s.endswith(","):
                s = s[:-1]
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "item":
                yield obj


def _extract_statement_list(entity: dict, pid: str) -> List[Statement]:
    out: List[Statement] = []
    claims = entity.get("claims", {})
    for stmt in claims.get(pid, []):
        mainsnak = stmt.get("mainsnak", {})
        if mainsnak.get("snaktype") != "value":
            continue
        dv = mainsnak.get("datavalue", {}).get("value", {})
        if not isinstance(dv, dict):
            continue
        value_qid = dv.get("id")
        if not value_qid:
            continue
        qualifiers = stmt.get("qualifiers", {})
        start = None
        end = None
        if "P580" in qualifiers and qualifiers["P580"]:
            start = qualifiers["P580"][0].get("datavalue", {}).get("value", {}).get("time")
        if "P582" in qualifiers and qualifiers["P582"]:
            end = qualifiers["P582"][0].get("datavalue", {}).get("value", {}).get("time")
        out.append(Statement(value_qid=value_qid, start_time=start, end_time=end))
    return out


def extract_candidates_from_wikidata_dump(
    *,
    wikidata_dump_path: str,
    properties: List[str],
    pages: int,
    page_size: int,
    progress: bool = True,
) -> Tuple[List[Candidate], Set[str]]:
    """
    "pages" here is interpreted as total WDQS-like page budget.
    We emulate this budget by processing at most pages*page_size candidate statements.
    """
    max_statements_budget = max(1, pages) * max(1, page_size)
    candidates: List[Candidate] = []
    needed_qids: Set[str] = set()
    statements_seen = 0

    entity_iter = _iter_wikidata_entities(wikidata_dump_path)
    if tqdm is not None:
        entity_iter = tqdm(entity_iter, desc="wikidata_entities", unit="item", disable=not progress)  # type: ignore

    for entity in entity_iter:
        subj_qid = entity.get("id", "")
        subj_label = entity.get("labels", {}).get("en", {}).get("value", "")
        title = entity.get("sitelinks", {}).get("enwiki", {}).get("title", "")
        if not subj_qid or not subj_label or not title:
            continue

        for pid in properties:
            stmts = _extract_statement_list(entity, pid)
            if not stmts:
                continue
            statements_seen += len(stmts)
            if statements_seen > max_statements_budget:
                return candidates, needed_qids

            def key_fn(x: Statement) -> datetime:
                t = _safe_parse_time(x.start_time or x.end_time or "1900-01-01T00:00:00Z")
                return t or datetime(1900, 1, 1, tzinfo=timezone.utc)

            stmts = sorted(stmts, key=key_fn)
            for a, b in zip(stmts, stmts[1:]):
                if a.value_qid == b.value_qid:
                    continue
                candidates.append(
                    Candidate(
                        subject_qid=subj_qid,
                        subject_label=subj_label,
                        wikipedia_title=title,
                        property_pid=pid,
                        property_label=PROPERTY_LABELS.get(pid, pid),
                        old_value_qid=a.value_qid,
                        new_value_qid=b.value_qid,
                        old_start=a.start_time,
                        old_end=a.end_time,
                        new_start=b.start_time,
                        new_end=b.end_time,
                    )
                )
                needed_qids.add(a.value_qid)
                needed_qids.add(b.value_qid)
    return candidates, needed_qids


def resolve_entity_labels_from_wikidata_dump(
    *, wikidata_dump_path: str, qids: Set[str], progress: bool = True
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    remaining = set(qids)
    if not remaining:
        return labels
    entity_iter = _iter_wikidata_entities(wikidata_dump_path)
    if tqdm is not None:
        entity_iter = tqdm(entity_iter, desc="wikidata_label_scan", unit="item", disable=not progress)  # type: ignore

    for entity in entity_iter:
        qid = entity.get("id")
        if qid not in remaining:
            continue
        lbl = entity.get("labels", {}).get("en", {}).get("value")
        if lbl:
            labels[qid] = lbl
        remaining.discard(qid)
        if not remaining:
            break
    return labels


def _localname(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def extract_revisions_from_wikipedia_dump(
    *,
    wikipedia_dump_path: str,
    title_to_targets: Dict[str, List[str]],
    progress: bool = True,
) -> Dict[Tuple[str, str], Tuple[int, str, str]]:
    """
    Return mapping (title, target_iso) -> (revid, rev_timestamp, wikitext)
    where revision timestamp <= target_iso and is closest.
    """
    result: Dict[Tuple[str, str], Tuple[int, str, str]] = {}
    needed_titles = set(title_to_targets.keys())
    if not needed_titles:
        return result

    page_bar = None
    if tqdm is not None:
        page_bar = tqdm(desc="wikipedia_pages", unit="page", disable=not progress)  # type: ignore

    with _open_binary(wikipedia_dump_path) as f:
        context = ET.iterparse(f, events=("end",))
        current_title: Optional[str] = None
        target_list: List[str] = []
        best_for_target: Dict[str, Tuple[int, str, str]] = {}
        in_target_page = False

        for _, elem in context:
            tag = _localname(elem.tag)
            if tag == "title":
                # Only valid when inside page context; lightweight enough for our purpose.
                pass

            if tag == "page":
                # finalize page and clear
                if in_target_page and current_title is not None:
                    for t in target_list:
                        if t in best_for_target:
                            result[(current_title, t)] = best_for_target[t]
                if page_bar is not None:
                    page_bar.update(1)
                    page_bar.set_postfix_str(f"matched={len(result)}")
                current_title = None
                target_list = []
                best_for_target = {}
                in_target_page = False
                elem.clear()
                continue

            if tag == "revision":
                if not in_target_page or current_title is None:
                    elem.clear()
                    continue
                rev_id = None
                rev_ts = None
                rev_text = ""
                for ch in elem:
                    ctag = _localname(ch.tag)
                    if ctag == "id" and rev_id is None:
                        if ch.text and ch.text.strip().isdigit():
                            rev_id = int(ch.text.strip())
                    elif ctag == "timestamp":
                        rev_ts = (ch.text or "").strip()
                    elif ctag == "text":
                        rev_text = ch.text or ""
                if rev_id is not None and rev_ts:
                    rev_dt = _safe_parse_time(rev_ts)
                    if rev_dt is not None:
                        for t in target_list:
                            tgt_dt = _safe_parse_time(t)
                            if tgt_dt is None:
                                continue
                            if rev_dt <= tgt_dt:
                                prev = best_for_target.get(t)
                                if prev is None or (_safe_parse_time(prev[1]) or rev_dt) < rev_dt:
                                    best_for_target[t] = (rev_id, rev_ts, rev_text)
                elem.clear()
                continue

            if tag == "title":
                # We rely on parent flow: whenever we observe a title text we can check interest.
                ttxt = (elem.text or "").strip()
                if ttxt in needed_titles:
                    current_title = ttxt
                    target_list = title_to_targets[ttxt]
                    in_target_page = True
                elem.clear()
                continue

            elem.clear()

    if page_bar is not None:
        page_bar.close()

    return result


def build_dataset_from_dumps(
    *,
    wikidata_dump_path: str,
    wikipedia_dump_path: str,
    out_path: str,
    properties: List[str],
    pages: int,
    page_size: int,
    limit: int = 500,
    min_gap_days: int = 30,
    include_time_in_question: bool = False,
    progress: bool = True,
) -> int:
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    candidates, needed_qids = extract_candidates_from_wikidata_dump(
        wikidata_dump_path=wikidata_dump_path,
        properties=properties,
        pages=pages,
        page_size=page_size,
        progress=progress,
    )
    if not candidates:
        Path(out_path).write_text("", encoding="utf-8")
        return 0

    label_map = resolve_entity_labels_from_wikidata_dump(
        wikidata_dump_path=wikidata_dump_path, qids=needed_qids, progress=progress
    )

    # Build per-title target timestamps needed for revision extraction.
    title_to_targets: Dict[str, List[str]] = {}
    for c in candidates:
        t_old = _pick_time_within_interval(c.old_start, c.old_end, fallback=now_iso)
        t_new = _pick_time_within_interval(c.new_start, c.new_end, fallback=now_iso)
        title_to_targets.setdefault(c.wikipedia_title, [])
        if t_old not in title_to_targets[c.wikipedia_title]:
            title_to_targets[c.wikipedia_title].append(t_old)
        if t_new not in title_to_targets[c.wikipedia_title]:
            title_to_targets[c.wikipedia_title].append(t_new)

    rev_map = extract_revisions_from_wikipedia_dump(
        wikipedia_dump_path=wikipedia_dump_path,
        title_to_targets=title_to_targets,
        progress=progress,
    )

    written = 0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cand_iter = enumerate(candidates)
    if tqdm is not None:
        cand_iter = enumerate(tqdm(candidates, desc="build_examples", unit="cand", disable=not progress))  # type: ignore

    with out.open("w", encoding="utf-8") as f:
        for idx, c in cand_iter:
            if written >= limit:
                break
            old_label = label_map.get(c.old_value_qid)
            new_label = label_map.get(c.new_value_qid)
            if not old_label or not new_label:
                continue
            if old_label == new_label:
                continue

            t_old = _pick_time_within_interval(c.old_start, c.old_end, fallback=now_iso)
            t_new = _pick_time_within_interval(c.new_start, c.new_end, fallback=now_iso)
            gap = _days_between(t_old, t_new)
            if gap < min_gap_days:
                continue

            ro = rev_map.get((c.wikipedia_title, t_old))
            rn = rev_map.get((c.wikipedia_title, t_new))
            if ro is None or rn is None:
                continue

            text_old = wikitext_to_plaintext(ro[2])
            text_new = wikitext_to_plaintext(rn[2])
            evidence_old = find_evidence_snippet(text=text_old, answer=old_label)
            evidence_new = find_evidence_snippet(text=text_new, answer=new_label)
            if not evidence_old or not evidence_new:
                continue

            q = render_question(
                pid=c.property_pid,
                subject_label=c.subject_label,
                as_of_date=( _safe_parse_time(t_new) or datetime.now(timezone.utc)).date().isoformat(),
                property_label=c.property_label,
                include_time_in_question=include_time_in_question,
            )

            ex = TemporalConflictExample(
                id=f"{c.subject_qid}_{c.property_pid}_{idx}",
                question=q,
                as_of_mode="timestamp",
                t_old=t_old,
                t_new=t_new,
                time_gap_days=gap,
                answer_old=old_label,
                answer_new=new_label,
                evidence_old=evidence_old,
                evidence_new=evidence_new,
                subject_qid=c.subject_qid,
                property_pid=c.property_pid,
                subject_label=c.subject_label,
                property_label=c.property_label,
                value_old_qid=c.old_value_qid,
                value_new_qid=c.new_value_qid,
                provenance_old=f"enwiki-dump:{c.wikipedia_title}|revid={ro[0]}|ts={ro[1]}",
                provenance_new=f"enwiki-dump:{c.wikipedia_title}|revid={rn[0]}|ts={rn[1]}",
                wikipedia_title=c.wikipedia_title,
            )
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            written += 1

    return written

