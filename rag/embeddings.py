"""OpenAI 임베딩 호출 래퍼 (배치 + 재시도 + L2 정규화)."""
from __future__ import annotations

import time

import numpy as np
from openai import OpenAI

from .config import EMBED_MODEL

BATCH_SIZE = 96
MAX_RETRY = 4


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_texts(client: OpenAI, texts: list[str], progress=None) -> np.ndarray:
    """텍스트 목록을 (n, dim) float32 정규화 행렬로 변환한다.

    progress: 0.0~1.0 진행률을 받는 콜백(선택).
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = [t.replace("\n", " ")[:8000] for t in texts[start : start + BATCH_SIZE]]
        for attempt in range(MAX_RETRY):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                vectors.extend(d.embedding for d in resp.data)
                break
            except Exception:
                if attempt == MAX_RETRY - 1:
                    raise
                time.sleep(2 ** attempt)
        if progress:
            progress(min(1.0, (start + len(batch)) / total))

    return _normalize(np.asarray(vectors, dtype=np.float32))


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    return embed_texts(client, [query])[0]
