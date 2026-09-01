"""data/pdfs 의 산업보고서를 임베딩해 data/index 에 검색 인덱스를 만든다.

배포 전에 로컬에서 한 번만 실행하고, 생성된 인덱스를 저장소에 커밋한다.
(Streamlit Cloud 에서 런타임에 재임베딩하지 않으므로 첫 응답이 빠르고 비용도 들지 않는다.)

    python scripts/build_index.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from rag.catalog import load_catalog
from rag.config import EMBED_MODEL, INDEX_DIR, PDF_DIR, get_api_key
from rag.embeddings import embed_texts
from rag.index import HybridIndex
from rag.pdf import pdf_to_chunks


def main() -> int:
    api_key = get_api_key()
    if not api_key:
        print("[오류] OPENAI_API_KEY 를 찾을 수 없습니다 (.env 또는 환경변수).")
        return 1

    docs = load_catalog()
    if not docs:
        print("[오류] data/sources.json 에 문서 정보가 없습니다.")
        return 1

    chunks = []
    doc_stats = []
    for doc in docs:
        path = PDF_DIR / doc.file
        if not path.exists():
            print(f"[건너뜀] 파일 없음: {doc.file}")
            continue
        t0 = time.time()
        doc_chunks = pdf_to_chunks(path, source=doc.label, start_id=len(chunks))
        chunks.extend(doc_chunks)
        doc_stats.append(
            {
                "file": doc.file,
                "label": doc.label,
                "publisher": doc.publisher,
                "date": doc.date,
                "url": doc.url,
                "chunks": len(doc_chunks),
            }
        )
        print(f"  {doc.file}: {len(doc_chunks)} chunks ({time.time() - t0:.1f}s)")

    if not chunks:
        print("[오류] 추출된 텍스트가 없습니다.")
        return 1

    total_chars = sum(len(c.text) for c in chunks)
    print(f"\n총 {len(chunks)} chunks / {total_chars:,} chars -> 임베딩 시작 ({EMBED_MODEL})")

    client = OpenAI(api_key=api_key)
    last = [0.0]

    def progress(ratio: float) -> None:
        if ratio - last[0] >= 0.1 or ratio >= 1.0:
            last[0] = ratio
            print(f"  임베딩 {ratio * 100:5.1f}%")

    embeddings = embed_texts(client, [c.text for c in chunks], progress=progress)

    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embed_model": EMBED_MODEL,
        "chunk_count": len(chunks),
        "total_chars": total_chars,
        "documents": doc_stats,
    }
    HybridIndex(chunks, embeddings, meta).save(INDEX_DIR)

    print(f"\n완료: {INDEX_DIR}")
    for name in ("embeddings.npy", "chunks.jsonl", "meta.json"):
        p = INDEX_DIR / name
        print(f"  {name}: {p.stat().st_size / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
