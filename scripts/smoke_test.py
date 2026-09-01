"""배포 전 점검용 스모크 테스트 (Streamlit 없이 RAG 파이프라인만 실행).

    python scripts/smoke_test.py            # 검색만 (API 호출 1회)
    python scripts/smoke_test.py --answer   # 답변 생성까지
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from rag.answer import stream_answer
from rag.catalog import example_questions, load_collection
from rag.config import DEFAULT_CHAT_MODEL, get_api_key
from rag.embeddings import embed_query
from rag.index import HybridIndex


def main() -> int:
    key = get_api_key()
    if not key:
        print("[오류] OPENAI_API_KEY 가 없습니다.")
        return 1

    index = HybridIndex.load()
    collection = load_collection()
    questions = example_questions(limit=3)
    print(f"지식베이스: {collection.name} ({collection.domain_label})")
    print(f"인덱스 로드 완료: 청크 {len(index.chunks):,}개 / 문서 {len(index.sources())}건")
    for src in index.sources():
        print(f"  - {src}")

    client = OpenAI(api_key=key)
    want_answer = "--answer" in sys.argv

    for q in questions:
        print("\n" + "=" * 78)
        print("Q:", q)
        hits = index.search(q, embed_query(client, q), top_k=5)
        if not hits:
            print("  [경고] 검색 결과 없음")
            continue
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h.chunk.source} p.{h.chunk.page} (score={h.score:.4f})")
            print(f"      {h.chunk.text[:110].replace(chr(10), ' ')}...")
        if want_answer:
            print("\nA:", end=" ")
            for tok in stream_answer(
                client, DEFAULT_CHAT_MODEL, q, hits, domain=collection.domain_label
            ):
                print(tok, end="", flush=True)
            print()
    print("\n스모크 테스트 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
