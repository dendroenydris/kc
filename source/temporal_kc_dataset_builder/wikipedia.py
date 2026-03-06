from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests

from .cache import FileCache


@dataclass(frozen=True)
class WikipediaRevision:
    pageid: int
    revid: int
    timestamp: str
    wikitext: str


def _api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def fetch_revision_at_or_before(
    *,
    title: str,
    as_of_iso: str,
    lang: str,
    user_agent: str,
    cache: FileCache,
    cache_only: bool = False,
    max_retries: int = 6,
    base_backoff_s: float = 1.0,
    polite_sleep_s: float = 0.2,
) -> Optional[WikipediaRevision]:
    """
    Fetch the most recent revision at or before `as_of_iso` (UTC ISO 8601).
    Uses MediaWiki API with rvstart + rvdir=older.
    """
    key = f"{lang}|{title}|{as_of_iso}"
    cached = cache.get("wiki_revision", key)
    if cached is not None:
        return WikipediaRevision(
            pageid=cached["pageid"],
            revid=cached["revid"],
            timestamp=cached["timestamp"],
            wikitext=cached["wikitext"],
        )
    if cache_only:
        return None

    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "revisions",
        "titles": title,
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "rvlimit": "1",
        "rvstart": as_of_iso,
        "rvdir": "older",
        "redirects": "1",
    }
    headers = {"User-Agent": user_agent}
    data = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(_api_url(lang), params=params, headers=headers, timeout=60)

            # MediaWiki can throttle with 429 or transient 5xx.
            if resp.status_code in {429, 500, 502, 503, 504}:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = float(retry_after)
                else:
                    sleep_s = min(30.0, base_backoff_s * (2**attempt))
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            try:
                data = resp.json()
            except (json.JSONDecodeError, requests.exceptions.JSONDecodeError):
                # Rare transient malformed JSON from upstream.
                sleep_s = min(30.0, base_backoff_s * (2**attempt))
                time.sleep(sleep_s)
                continue
            break
        except requests.RequestException:
            sleep_s = min(30.0, base_backoff_s * (2**attempt))
            time.sleep(sleep_s)

    if data is None:
        print(
            f"[temporal-kc] warning: wikipedia revision fetch failed after retries "
            f"title={title!r} as_of={as_of_iso}",
            file=sys.stderr,
        )
        return None

    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        return None

    revs = pages[0].get("revisions", [])
    if not revs:
        return None

    rev = revs[0]
    wikitext = ""
    slots = rev.get("slots", {})
    if "main" in slots:
        wikitext = slots["main"].get("content", "") or ""
    if not wikitext:
        return None

    out = WikipediaRevision(
        pageid=int(pages[0]["pageid"]),
        revid=int(rev.get("revid", 0)),
        timestamp=str(rev["timestamp"]),
        wikitext=wikitext,
    )

    cache.set(
        "wiki_revision",
        key,
        {"pageid": out.pageid, "revid": out.revid, "timestamp": out.timestamp, "wikitext": out.wikitext},
    )
    time.sleep(polite_sleep_s)
    return out


def provenance_string(*, lang: str, title: str, revid: int, timestamp: str) -> str:
    return f"{lang}wiki:{title}|revid={revid}|ts={timestamp}"

