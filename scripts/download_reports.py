"""기본 지식베이스로 쓰는 공개 산업보고서 PDF를 내려받는다.

data/sources.json 에 적힌 URL 을 그대로 사용하므로, 저장소를 새로 받은 사람도
같은 지식베이스를 재현할 수 있다. (PDF 는 저장소에 함께 커밋되어 있으므로
이 스크립트는 원문 갱신이 필요할 때만 실행하면 된다.)

    python scripts/download_reports.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.catalog import load_catalog
from rag.config import PDF_DIR

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  [실패] {exc}")
        return False
    if not data.startswith(b"%PDF"):
        print("  [실패] PDF 응답이 아닙니다(사이트 정책 변경 가능).")
        return False
    dest.write_bytes(data)
    print(f"  [완료] {len(data) / 1024 / 1024:.2f} MB")
    return True


def main() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for doc in load_catalog():
        dest = PDF_DIR / doc.file
        print(f"- {doc.publisher} / {doc.title}")
        if dest.exists() and dest.stat().st_size > 0:
            print("  [건너뜀] 이미 존재")
            ok += 1
            continue
        ok += download(doc.url, dest)
    print(f"\n{ok}개 문서 준비 완료 -> {PDF_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
