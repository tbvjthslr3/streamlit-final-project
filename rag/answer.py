"""검색 결과를 근거로 답변을 생성하는 레이어 (근거 인용 강제).

프롬프트에 특정 산업을 하드코딩하지 않는다. 담당 영역은 지식베이스 정의에서
넘어온 `domain` 값으로 채워지므로, 보고서 모음만 바꾸면 그대로 재사용된다.
"""
from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from .index import Hit

SYSTEM_PROMPT_TEMPLATE = """당신은 {domain}을(를) 담당하는 시니어 리서치 애널리스트입니다.
아래에 주어진 '참고 자료'(산업보고서에서 검색된 발췌문)만을 근거로 한국어로 답변하세요.

규칙:
1. 참고 자료에 없는 내용은 절대 지어내지 마세요. 질문한 항목 중 자료에서 찾을 수 없는 부분이 있을 때만
   그 항목을 특정해 "○○는 제공된 보고서에서 확인되지 않습니다"라고 적습니다.
   답변할 내용이 충분하면 이 문구를 쓰지 말고, 습관적으로 마지막 줄에 덧붙이지 마세요.
2. 사실·수치·전망을 언급할 때마다 문장 끝에 [1], [2] 형태로 근거 번호를 답니다. 한 문장에 여러 근거 가능: [1][3]
3. 수치는 단위·연도·기준(전망치/실적치)을 함께 밝힙니다. 보고서마다 수치가 다르면 차이를 그대로 드러내세요.
4. 구조: 먼저 3줄 이내 핵심 요약, 이어서 근거가 되는 세부 내용을 불릿으로 정리합니다.
5. 실무자가 바로 쓸 수 있도록 간결하고 단정적인 문장을 쓰되, 추정과 확인된 사실을 구분해 표기합니다.
6. 답변 마지막에 출처 목록을 다시 쓰지 마세요. (앱이 별도로 표시합니다)
7. 참고 자료가 영문이어도 답변은 한국어로 하되, 고유명사와 지표명은 원어를 병기합니다."""

DEFAULT_DOMAIN = "산업 리서치"

NO_CONTEXT_MESSAGE = (
    "질문과 관련된 내용을 지식베이스에서 찾지 못했습니다. "
    "검색 범위(사이드바 문서 필터)를 넓히거나, 질문을 좀 더 구체적인 용어로 바꿔 보세요. "
    "다루려는 보고서가 아직 없다면 사이드바에서 PDF를 업로드해 주세요."
)


def system_prompt(domain: str | None = None) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(domain=domain or DEFAULT_DOMAIN)


def format_context(hits: list[Hit], max_chars: int = 12000) -> str:
    """검색 결과를 번호가 매겨진 참고 자료 블록으로 직렬화한다."""
    blocks: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        block = f"[{i}] 출처: {hit.chunk.source} (p.{hit.chunk.page})\n{hit.chunk.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def build_messages(
    question: str,
    hits: list[Hit],
    history: list[dict] | None = None,
    domain: str | None = None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt(domain)}]

    # 후속 질문("그럼 국내는?")을 이해할 수 있도록 최근 대화만 짧게 전달한다.
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"][:1500]})

    context = format_context(hits)
    messages.append(
        {
            "role": "user",
            "content": f"# 참고 자료\n{context}\n\n# 질문\n{question}",
        }
    )
    return messages


def stream_answer(
    client: OpenAI,
    model: str,
    question: str,
    hits: list[Hit],
    history: list[dict] | None = None,
    domain: str | None = None,
) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=model,
        messages=build_messages(question, hits, history, domain),
        temperature=0.2,
        max_tokens=1200,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
