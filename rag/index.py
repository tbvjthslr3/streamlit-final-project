"""하이브리드 검색 인덱스: 임베딩 코사인 유사도 + BM25 를 RRF 로 결합.

FAISS 같은 무거운 네이티브 의존성 없이 numpy 만으로 동작하도록 만들어
Streamlit Community Cloud 배포 실패 위험을 줄였다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import CHUNK_FILE, EMB_FILE, META_FILE, RRF_K
from .pdf import Chunk

_TOKEN = re.compile(r"[a-z0-9]+|[가-힣]+")


def tokenize(text: str) -> list[str]:
    """한국어는 형태소 분석기 없이 어절 + 음절 bigram 으로 근사한다.

    (konlpy/mecab 은 배포 환경에서 설치 실패 위험이 커서 의도적으로 배제)
    """
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        tokens.append(tok)
        if "\uac00" <= tok[0] <= "\ud7a3" and len(tok) > 2:
            tokens.extend(tok[i : i + 2] for i in range(len(tok) - 1))
    return tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, meta: dict | None = None):
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.meta = meta or {}
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    # ---------- 저장 / 로드 ----------
    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        # float16 으로 저장해 저장소 용량을 절반으로 줄인다 (검색 품질 영향 미미).
        np.save(EMB_FILE, self.embeddings.astype(np.float16))
        with open(CHUNK_FILE, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        META_FILE.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "HybridIndex":
        if not (EMB_FILE.exists() and CHUNK_FILE.exists()):
            raise FileNotFoundError(
                "인덱스 파일이 없습니다. `python scripts/build_index.py` 로 먼저 인덱스를 생성하세요."
            )
        embeddings = np.load(EMB_FILE).astype(np.float32)
        chunks = []
        with open(CHUNK_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))
        meta = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}
        return cls(chunks, embeddings, meta)

    # ---------- 검색 ----------
    def sources(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int = 6,
        allowed_sources: list[str] | None = None,
    ) -> list[Hit]:
        if not self.chunks:
            return []

        mask = np.ones(len(self.chunks), dtype=bool)
        if allowed_sources is not None:
            allow = set(allowed_sources)
            mask = np.array([c.source in allow for c in self.chunks], dtype=bool)
        candidate_idx = np.nonzero(mask)[0]
        if candidate_idx.size == 0:
            return []

        pool = min(max(top_k * 5, 20), candidate_idx.size)

        # 1) dense: 정규화 벡터이므로 내적 = 코사인 유사도
        dense_scores = self.embeddings[candidate_idx] @ query_vec.astype(np.float32)
        dense_order = candidate_idx[np.argsort(-dense_scores)[:pool]]

        # 2) lexical: BM25 (고유명사/숫자/약어 검색 보완)
        lexical_order: list[int] = []
        if self._bm25 is not None:
            bm_scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
            bm_scores[~mask] = -np.inf
            lexical_order = list(np.argsort(-bm_scores)[:pool])

        # 3) Reciprocal Rank Fusion
        fused: dict[int, float] = {}
        dense_rank: dict[int, int] = {}
        lex_rank: dict[int, int] = {}
        for rank, idx in enumerate(dense_order):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
            dense_rank[int(idx)] = rank + 1
        for rank, idx in enumerate(lexical_order):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
            lex_rank[int(idx)] = rank + 1

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=score,
                dense_rank=dense_rank.get(i),
                lexical_rank=lex_rank.get(i),
            )
            for i, score in ranked
        ]


def merge(base: HybridIndex, extra: HybridIndex | None) -> HybridIndex:
    """기본 지식베이스 + 사용자가 업로드한 문서를 합친 임시 인덱스."""
    if extra is None or not extra.chunks:
        return base
    chunks = base.chunks + extra.chunks
    embeddings = np.vstack([base.embeddings, extra.embeddings])
    return HybridIndex(chunks, embeddings, base.meta)
