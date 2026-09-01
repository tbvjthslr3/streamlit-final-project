"""PDF -> 페이지 텍스트 -> 검색용 청크 변환."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    id: int
    text: str
    source: str        # 표시용 문서명
    file: str          # 파일명
    page: int          # 1-based 페이지 번호

    def to_dict(self) -> dict:
        return asdict(self)


_WS = re.compile(r"[ \t\u00a0]+")
_NL = re.compile(r"\n{3,}")
_BROKEN = re.compile(r"(?<=[가-힣a-zA-Z0-9,])\n(?=[가-힣a-zA-Z0-9])")


def clean_text(text: str) -> str:
    """PDF 추출 특유의 줄바꿈/공백 노이즈를 정리한다."""
    text = text.replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _BROKEN.sub(" ", text)      # 문장 중간에서 끊긴 줄 합치기
    text = _NL.sub("\n\n", text)
    return text.strip()


def read_pages(path: Path) -> list[tuple[int, str]]:
    """(페이지번호, 텍스트) 목록. 이미지 전용 페이지는 건너뛴다."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        text = clean_text(raw)
        if len(text) >= MIN_CHUNK_CHARS:
            pages.append((i, text))
    return pages


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """문단 -> 문장 경계를 우선 지키면서 고정 길이로 자른다."""
    if len(text) <= size:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            # 문장/문단 경계에서 자르기 위해 뒤에서부터 후보를 찾는다.
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("다. "), window.rfind("? "))
            if cut > size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            parts.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts


def pdf_to_chunks(path: Path, source: str, start_id: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    cid = start_id
    for page_no, page_text in read_pages(path):
        for piece in split_text(page_text):
            chunks.append(Chunk(id=cid, text=piece, source=source, file=path.name, page=page_no))
            cid += 1
    return chunks


def chunks_from_bytes(data: bytes, filename: str, source: str, start_id: int, tmp_dir: Path) -> list[Chunk]:
    """업로드된 PDF(bytes)를 임시 저장 후 청크로 변환한다."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / filename
    tmp_path.write_bytes(data)
    try:
        return pdf_to_chunks(tmp_path, source=source, start_id=start_id)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def iter_pdfs(directory: Path) -> Iterable[Path]:
    return sorted(p for p in directory.glob("*.pdf") if p.stat().st_size > 0)
