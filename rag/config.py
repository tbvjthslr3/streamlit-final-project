"""앱 전역 설정값과 API 키 로딩 규칙을 한 곳에서 관리한다."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"

# --- 인덱스 파일 ---
EMB_FILE = INDEX_DIR / "embeddings.npy"
CHUNK_FILE = INDEX_DIR / "chunks.jsonl"
META_FILE = INDEX_DIR / "meta.json"

# --- 모델 ---
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
CHAT_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"]

# --- 청킹 ---
CHUNK_SIZE = 900          # 문자 기준
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 60

# --- 검색 ---
DEFAULT_TOP_K = 6
MAX_TOP_K = 12
RRF_K = 60                # Reciprocal Rank Fusion 상수

# --- 비용/남용 방지 가드레일 (공개 배포용) ---
MAX_QUESTIONS_PER_SESSION = 30
MAX_UPLOAD_MB = 20
MAX_UPLOAD_PAGES = 120
MAX_QUESTION_CHARS = 500


def get_api_key(user_key: str | None = None) -> str | None:
    """OpenAI API 키를 우선순위대로 찾는다.

    1) 사용자가 사이드바에 직접 입력한 키
    2) Streamlit Secrets (배포 환경)
    3) 환경변수 / .env (로컬 개발)
    키를 코드나 화면에 하드코딩하지 않기 위한 단일 진입점이다.
    """
    if user_key and user_key.strip():
        return user_key.strip()

    try:  # streamlit 이 없는 스크립트 실행 환경도 지원
        import streamlit as st

        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass

    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()

    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None
