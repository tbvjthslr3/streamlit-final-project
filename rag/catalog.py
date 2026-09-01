"""기본 지식베이스 문서 목록(data/sources.json) 로딩."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .config import DATA_DIR

CATALOG_FILE = DATA_DIR / "sources.json"


@dataclass
class Document:
    file: str
    title: str
    publisher: str
    date: str
    lang: str
    topics: list[str]
    url: str

    @property
    def label(self) -> str:
        """인용 표시에 쓰는 짧은 이름."""
        year = self.date[:4]
        return f"{self.publisher} · {self.title} ({year})"


def load_catalog() -> list[Document]:
    if not CATALOG_FILE.exists():
        return []
    raw = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    return [Document(**d) for d in raw.get("documents", [])]


def by_label() -> dict[str, Document]:
    return {d.label: d for d in load_catalog()}
