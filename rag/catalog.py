"""지식베이스 정의(data/sources.json) 로딩.

앱 코드에는 특정 산업이 하드코딩되어 있지 않다. 어떤 산업의 보고서 모음인지는
이 파일이 읽는 sources.json 의 `collection` 블록이 결정하므로, PDF와 이 JSON만
교체하면 같은 앱을 다른 산업에 그대로 재사용할 수 있다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import DATA_DIR

CATALOG_FILE = DATA_DIR / "sources.json"

# sources.json 에 collection 이 없거나 사용자 업로드만 있을 때 쓰는 기본값.
GENERIC_QUESTIONS = [
    "수록된 보고서들의 핵심 결론을 요약해 주세요.",
    "시장 규모와 성장률 전망은 어떻게 되나요?",
    "가장 큰 리스크 요인은 무엇인가요?",
]


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


@dataclass
class Collection:
    """기본 지식베이스가 어떤 산업을 다루는지에 대한 메타데이터."""

    industry: str = ""
    name: str = "산업 보고서"
    summary: str = ""
    sample_questions: list[str] = field(default_factory=list)

    @property
    def domain_label(self) -> str:
        """프롬프트에 넣을 담당 영역 표현."""
        return f"{self.industry} 산업" if self.industry else "산업"


def _raw() -> dict:
    if not CATALOG_FILE.exists():
        return {}
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


def load_catalog() -> list[Document]:
    return [Document(**d) for d in _raw().get("documents", [])]


def load_collection() -> Collection:
    return Collection(**_raw().get("collection", {}))


def example_questions(limit: int = 5) -> list[str]:
    """수록 문서에 맞는 예시 질문을 만든다.

    sources.json 이 직접 지정한 질문을 먼저 쓰고, 모자라면 문서의 topics 에서
    파생 질문을 만든 뒤, 그래도 모자라면 산업 무관 일반 질문으로 채운다.
    """
    questions = list(load_collection().sample_questions)

    seen_topics: list[str] = []
    for doc in load_catalog():
        for topic in doc.topics:
            if topic not in seen_topics:
                seen_topics.append(topic)
    for topic in seen_topics:
        if len(questions) >= limit:
            break
        # 이미 나온 질문과 겹치는 주제는 건너뛴다(같은 질문이 두 번 보이지 않도록).
        keywords = [w for w in topic.split() if len(w) > 1]
        if keywords and any(all(w in q for w in keywords) for q in questions):
            continue
        questions.append(f"{topic}에 대해 정리해 주세요.")

    for q in GENERIC_QUESTIONS:
        if len(questions) >= limit:
            break
        if q not in questions:
            questions.append(q)

    return questions[:limit]
