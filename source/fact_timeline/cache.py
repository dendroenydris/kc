"""Simple filesystem cache keyed by SHA-256 hash of a string key.

Each (namespace, key) pair is stored as a single JSON file under
<root>/<namespace>/<sha256(key)>.json
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class FileCache:
    root: Path

    def _path(self, namespace: str, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{h}.json"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def set(self, namespace: str, key: str, value: Any) -> None:
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
