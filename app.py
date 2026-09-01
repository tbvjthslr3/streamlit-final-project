"""데이터센터 산업 리포트 Q&A 챗봇 (RAG).

공개된 데이터센터 산업보고서 PDF를 벡터 + BM25 하이브리드로 검색해,
근거 페이지를 인용하며 답변하는 Streamlit 앱.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import streamlit as st
from openai import OpenAI

from rag import config
from rag.answer import NO_CONTEXT_MESSAGE, stream_answer
from rag.catalog import load_catalog
from rag.embeddings import embed_query, embed_texts
from rag.index import HybridIndex, merge
from rag.pdf import chunks_from_bytes

st.set_page_config(
    page_title="데이터센터 산업 리포트 Q&A",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "2030년 국내 AI 데이터센터 수요 전망은 어느 정도인가요?",
    "글로벌 데이터센터 전력 소비량은 앞으로 얼마나 늘어나나요?",
    "국내 데이터센터 확충의 가장 큰 병목은 무엇인가요?",
    "데이터센터 냉각 방식의 기술 트렌드를 정리해 주세요.",
    "데이터센터 밸류체인에서 수혜가 예상되는 영역은 어디인가요?",
]

TMP_UPLOAD_DIR = Path(config.DATA_DIR) / "_tmp"


# ---------------------------------------------------------------- 리소스 로딩
@st.cache_resource(show_spinner="지식베이스를 불러오는 중...")
def load_index() -> HybridIndex | None:
    try:
        return HybridIndex.load()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=60.0, max_retries=2)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=256)
def cached_query_vector(_client: OpenAI, question: str) -> np.ndarray:
    """같은 질문이 반복될 때 임베딩 호출 비용을 아낀다."""
    return embed_query(_client, question)


def init_state() -> None:
    st.session_state.setdefault("messages", [])       # {role, content, hits}
    st.session_state.setdefault("question_count", 0)
    st.session_state.setdefault("upload_index", None)
    st.session_state.setdefault("upload_names", [])
    st.session_state.setdefault("pending_question", None)


init_state()
base_index = load_index()
catalog = {d.label: d for d in load_catalog()}


# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("🏢 데이터센터 리서치")

    api_key = config.get_api_key()
    if not api_key:
        st.warning("서버에 API 키가 설정되어 있지 않습니다. 직접 입력해 주세요.")
        typed = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="입력한 키는 이 브라우저 세션에서만 사용되며 저장되지 않습니다.",
        )
        api_key = config.get_api_key(typed)

    st.divider()

    if base_index is None:
        st.error("인덱스가 없습니다. `python scripts/build_index.py` 실행이 필요합니다.")
        all_sources: list[str] = []
    else:
        all_sources = base_index.sources()
        meta = base_index.meta
        st.caption(
            f"보고서 {len(all_sources)}건 · 검색 단위 {meta.get('chunk_count', 0):,}개 "
            f"· 인덱스 {str(meta.get('built_at', '-'))[:10]}"
        )

    st.subheader("검색 대상 문서")
    source_options = all_sources + st.session_state.upload_names
    selected_sources = st.multiselect(
        "선택한 보고서 안에서만 검색합니다",
        options=source_options,
        default=source_options,
        label_visibility="collapsed",
    )

    with st.expander("📄 보고서 원문 보기"):
        for doc in catalog.values():
            st.markdown(f"- [{doc.title}]({doc.url})  \n  `{doc.publisher} · {doc.date}`")

    st.divider()
    st.subheader("내 보고서 추가")
    uploads = st.file_uploader(
        f"PDF 업로드 (파일당 최대 {config.MAX_UPLOAD_MB}MB)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploads and st.button("업로드한 PDF 색인하기", use_container_width=True, type="primary"):
        if not api_key:
            st.error("API 키가 필요합니다.")
        else:
            try:
                client = get_client(api_key)
                new_chunks = []
                names: list[str] = []
                for up in uploads:
                    data = up.getvalue()
                    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
                        st.warning(f"{up.name}: 용량 초과로 건너뜁니다.")
                        continue
                    label = f"[내 문서] {Path(up.name).stem}"
                    chunks = chunks_from_bytes(
                        data, up.name, label, len(new_chunks), TMP_UPLOAD_DIR
                    )
                    if not chunks:
                        st.warning(f"{up.name}: 텍스트를 추출하지 못했습니다(스캔 이미지 PDF일 수 있음).")
                        continue
                    new_chunks.extend(chunks[: config.MAX_UPLOAD_PAGES * 3])
                    names.append(label)
                if new_chunks:
                    with st.spinner(f"{len(new_chunks)}개 조각 색인 중..."):
                        vecs = embed_texts(client, [c.text for c in new_chunks])
                    st.session_state.upload_index = HybridIndex(new_chunks, vecs)
                    st.session_state.upload_names = names
                    st.success(f"{len(names)}개 문서를 지식베이스에 추가했습니다.")
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"색인 실패: {exc}")

    if st.session_state.upload_names:
        st.caption(
            "추가됨: " + ", ".join(n.replace("[내 문서] ", "") for n in st.session_state.upload_names)
        )
        if st.button("추가 문서 비우기", use_container_width=True):
            st.session_state.upload_index = None
            st.session_state.upload_names = []
            st.rerun()

    st.divider()
    with st.expander("⚙️ 고급 설정"):
        model = st.selectbox("답변 모델", config.CHAT_MODELS, index=0)
        top_k = st.slider("검색 근거 개수", 3, config.MAX_TOP_K, config.DEFAULT_TOP_K)

    remaining = config.MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
    st.caption(f"이 세션 남은 질문: **{max(remaining, 0)}회**")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------- 본문
st.title("데이터센터 산업 리포트 Q&A")
st.caption(
    "국내외 공개 데이터센터 산업보고서를 근거로 답변합니다. "
    "모든 답변에는 출처 보고서와 페이지 번호가 붙습니다."
)

with st.expander("이 앱은 무엇인가요?"):
    st.markdown(
        """
**해결하려는 문제** — 데이터센터 관련 수치 하나를 확인하려면 증권사·연구기관 보고서 수백 페이지를
매번 열어 찾아야 합니다. 이 앱은 그 보고서들을 하나의 지식베이스로 묶어, 질문 한 줄로
**근거 페이지와 함께** 답을 돌려줍니다.

**동작 방식** — 질문을 임베딩해 의미 검색(코사인 유사도)을 하고, 동시에 BM25 키워드 검색을 돌린 뒤
두 결과를 RRF(Reciprocal Rank Fusion)로 결합합니다. 상위 근거만 LLM에 넘겨 답변을 만들고,
자료에 없는 내용은 지어내지 않도록 지시합니다.

**주의** — 답변은 공개 보고서 발췌에 기반한 요약이며, 투자 판단의 근거가 아닙니다.
        """
    )

if base_index is None:
    st.stop()

active_index = merge(base_index, st.session_state.upload_index)


def render_sources(hits) -> None:
    if not hits:
        return
    with st.expander(f"📎 근거 {len(hits)}건 보기"):
        for i, hit in enumerate(hits, start=1):
            doc = catalog.get(hit.chunk.source)
            head = f"**[{i}] {hit.chunk.source} — p.{hit.chunk.page}**"
            if doc:
                head += f"  ([원문 보기]({doc.url}))"
            st.markdown(head)
            st.markdown(
                "> " + hit.chunk.text[:900].replace("\n", "\n> ")
            )
            st.caption(
                f"관련도 {hit.score:.4f} · 의미검색 {hit.dense_rank or '-'}위 "
                f"· 키워드검색 {hit.lexical_rank or '-'}위"
            )
            st.divider()


# 예시 질문 (대화 시작 전에만 노출)
if not st.session_state.messages:
    st.markdown("**이런 질문으로 시작해 보세요**")
    for col, q in zip(st.columns(3), EXAMPLE_QUESTIONS[:3]):
        if col.button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()
    for col, q in zip(st.columns(2), EXAMPLE_QUESTIONS[3:]):
        if col.button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("hits", []))

typed_question = st.chat_input("데이터센터 산업에 대해 물어보세요")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    question = question.strip()[: config.MAX_QUESTION_CHARS]

    if not api_key:
        st.error("OpenAI API 키가 설정되지 않아 답변할 수 없습니다.")
        st.stop()
    if st.session_state.question_count >= config.MAX_QUESTIONS_PER_SESSION:
        st.warning("이 세션의 질문 한도에 도달했습니다. 페이지를 새로고침하면 다시 시작됩니다.")
        st.stop()
    if not selected_sources:
        st.warning("사이드바에서 검색할 문서를 하나 이상 선택해 주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            client = get_client(api_key)
            t0 = time.time()
            with st.spinner("보고서에서 근거를 찾는 중..."):
                qvec = cached_query_vector(client, question)
                hits = active_index.search(
                    question, qvec, top_k=top_k, allowed_sources=selected_sources
                )

            if not hits:
                st.markdown(NO_CONTEXT_MESSAGE)
                st.session_state.messages.append(
                    {"role": "assistant", "content": NO_CONTEXT_MESSAGE, "hits": []}
                )
            else:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                answer = st.write_stream(stream_answer(client, model, question, hits, history))
                st.caption(f"응답 {time.time() - t0:.1f}초 · 근거 {len(hits)}건 · {model}")
                render_sources(hits)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "hits": hits}
                )
            st.session_state.question_count += 1

        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
            low = detail.lower()
            if "quota" in low:
                msg = "OpenAI 사용 한도가 소진되었습니다. 관리자에게 문의해 주세요."
            elif "invalid_api_key" in low or "incorrect api key" in low:
                msg = "API 키가 올바르지 않습니다."
            elif "rate_limit" in low:
                msg = "요청이 몰리고 있습니다. 잠시 후 다시 시도해 주세요."
            else:
                msg = f"답변 생성 중 오류가 발생했습니다: {detail[:200]}"
            st.error(msg)
