from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import requests

from .cache import FileCache


WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(frozen=True)
class WikidataStatement:
    subject_qid: str
    subject_label: str
    wikipedia_title: str

    property_pid: str
    property_label: str

    value_qid: str
    value_label: str

    start_time: Optional[str]  # ISO datetime
    end_time: Optional[str]  # ISO datetime


def _sparql_get(
    query: str,
    *,
    user_agent: str,
    cache: FileCache,
    cache_only: bool = False,
    max_retries: int = 6,
    backoff_s: float = 1.0,
) -> Dict:
    cached = cache.get("sparql", query)
    if cached is not None:
        return cached
    if cache_only:
        raise RuntimeError("cache-only miss (sparql)")

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": user_agent,
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_URL,
                params={"query": query, "format": "json"},
                headers=headers,
                timeout=60,
            )

            # WDQS sometimes returns HTML/text for rate limits or transient errors.
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"WDQS transient HTTP {resp.status_code}")

            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "json" not in ctype and "sparql-results+json" not in ctype:
                # Often an HTML error page.
                snippet = resp.text[:500]
                raise RuntimeError(f"WDQS non-JSON content-type={ctype}; head={snippet!r}")

            try:
                # requests' JSON parser is strict and can fail on control chars
                # occasionally leaked by upstream. Parse manually with strict=False.
                data = json.loads(resp.text, strict=False)
            except (json.JSONDecodeError, requests.exceptions.JSONDecodeError) as e:
                # Last-resort sanitization: strip illegal control chars and retry.
                sanitized = _CTRL_CHARS_RE.sub(" ", resp.text)
                try:
                    data = json.loads(sanitized, strict=False)
                except Exception as e2:  # noqa: BLE001
                    snippet = resp.text[:500]
                    raise RuntimeError(f"WDQS JSON decode failed; head={snippet!r}") from e2

            if not isinstance(data, dict) or "results" not in data:
                raise RuntimeError("WDQS response missing results")

            cache.set("sparql", query, data)
            time.sleep(0.2)  # be polite
            return data
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Exponential backoff with cap.
            sleep_s = min(30.0, backoff_s * (2**attempt))
            time.sleep(sleep_s)

    raise RuntimeError(f"WDQS query failed after {max_retries} retries") from last_err


def _qid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _pid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def fetch_time_scoped_statements(
    *,
    property_pids: Iterable[str],
    limit_per_property: int,
    page_size: int = 1000,
    max_pages_per_property: Optional[int] = None,
    max_total_pages: Optional[int] = None,
    user_agent: str,
    cache: FileCache,
    wiki_lang: str = "en",
    progress_callback: Optional[Callable[[str, Dict], None]] = None,
    cache_only: bool = False,
) -> List[WikidataStatement]:
    """
    Fetch statements with optional start/end time qualifiers.

    We query entity-valued properties and retrieve:
    - subject (with Wikipedia sitelink)
    - value entity
    - qualifiers: start time (P580), end time (P582) if present
    """
    out: List[WikidataStatement] = []
    total_pages_done = 0
    stop_all = False

    for pid in property_pids:
        if stop_all:
            break
        if progress_callback is not None:
            progress_callback("property_start", {"pid": pid})
        collected = 0
        offset = 0
        pages_done = 0
        while True:
            if max_total_pages is not None and total_pages_done >= max_total_pages:
                stop_all = True
                break
            if max_pages_per_property is not None and pages_done >= max_pages_per_property:
                break
            if collected >= limit_per_property:
                break

            if max_pages_per_property is not None:
                lim = page_size
            else:
                lim = min(page_size, limit_per_property - collected)
            query = f"""
SELECT ?subject ?subjectLabel ?title ?p ?pLabel ?value ?valueLabel ?start ?end WHERE {{
  ?subject p:{pid} ?statement .
  ?statement ps:{pid} ?value .
  OPTIONAL {{ ?statement pq:P580 ?start. }}
  OPTIONAL {{ ?statement pq:P582 ?end. }}

  ?sitelink schema:about ?subject ;
           schema:isPartOf <https://{wiki_lang}.wikipedia.org/> ;
           schema:name ?title .

  BIND(wdt:{pid} as ?p)

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {int(lim)}
OFFSET {int(offset)}
"""
            try:
                data = _sparql_get(query, user_agent=user_agent, cache=cache, cache_only=cache_only)
            except Exception as e:  # noqa: BLE001
                # Skip one failed page and continue; do not kill the entire run.
                print(
                    f"[temporal-kc] warning: WDQS page failed pid={pid} offset={offset} "
                    f"limit={lim}; err={e}",
                )
                if progress_callback is not None:
                    progress_callback("page_done", {"pid": pid, "ok": False, "rows": 0, "offset": offset, "limit": lim})
                total_pages_done += 1
                # In cache-only mode with unknown page budget, a miss means we should stop
                # to avoid unbounded offset scanning.
                if cache_only and max_pages_per_property is None:
                    break
                pages_done += 1
                offset += lim
                continue
            rows = data.get("results", {}).get("bindings", [])
            if progress_callback is not None:
                progress_callback(
                    "page_done",
                    {"pid": pid, "ok": True, "rows": len(rows), "offset": offset, "limit": lim},
                )
            total_pages_done += 1
            if not rows:
                break

            for row in rows:
                subj_uri = row["subject"]["value"]
                val_uri = row["value"]["value"]
                p_uri = row["p"]["value"]
                stmt = WikidataStatement(
                    subject_qid=_qid_from_uri(subj_uri),
                    subject_label=row.get("subjectLabel", {}).get("value", ""),
                    wikipedia_title=row.get("title", {}).get("value", ""),
                    property_pid=_pid_from_uri(p_uri),
                    property_label=row.get("pLabel", {}).get("value", pid),
                    value_qid=_qid_from_uri(val_uri),
                    value_label=row.get("valueLabel", {}).get("value", ""),
                    start_time=row.get("start", {}).get("value"),
                    end_time=row.get("end", {}).get("value"),
                )
                if not stmt.wikipedia_title or not stmt.value_label or not stmt.subject_label:
                    continue
                out.append(stmt)
                collected += 1
                if collected >= limit_per_property:
                    break

            offset += lim
            pages_done += 1

    return out

