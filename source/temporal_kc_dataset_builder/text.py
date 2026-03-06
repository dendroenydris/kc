from __future__ import annotations

import re
from typing import List, Optional

import mwparserfromhell


_WS_RE = re.compile(r"\s+")


def wikitext_to_plaintext(wikitext: str) -> str:
    code = mwparserfromhell.parse(wikitext)
    text = code.strip_code(normalize=True, collapse=True)
    text = _WS_RE.sub(" ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    # Simple heuristic sentence split that works "well enough" for evidence snippets.
    # Avoid heavy dependencies; you can swap in spacy later if desired.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def find_evidence_snippet(
    *,
    text: str,
    answer: str,
    max_sentences: int = 2,
    window: int = 1,
) -> Optional[str]:
    """
    Return a short snippet (1–2 sentences) that contains the answer string.
    """
    if not answer:
        return None

    sentences = split_sentences(text)
    ans_lower = answer.lower()

    for i, s in enumerate(sentences):
        if ans_lower in s.lower():
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            snippet = " ".join(sentences[start:end][:max_sentences])
            return snippet.strip()

    return None

