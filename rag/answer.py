"""검색 결과를 근거로 답변을 생성하는 레이어 (근거 인용 강제)."""
from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from .index import Hit

SYSTEM_PROMPT = """당신은 데이터센터 산업을 담당하는 시니어 리서치 애널리스트입니다.
아래에 주어진 '참고 자료'(산업보고서에서 검색된 발췌문)만을 근거로 한국어로 답변하세요.

규칙:
1. 참고 자료에 없는 내용은 절대 지어내지 마세요. 질문한 항목 중 자료에서 찾을 수 없는 부분이 있을 때만
   그 항목을 특정해 "○○는 제공된 보고서에서 확인되지 않습니다"라고 적습니다.
   답변할 내용이 충분하면 이 문구를 쓰지 말고, 습관적으로 마지막 줄에 덧붙이지 마세요.
2. 사실·수치·전망을 언급할 때마다 문장 끝에 [1], [2] 형태로 근거 번호를 답니다. 한 문장에 여러 근거 가능: [1][3]
3. 수치는 단위·연도·기준(전망치/실적치)을 함께 밝힙니다. 보고서마다 수치가 다르면 차이를 그대로 드러내세요.
4. 구조: 먼저 3줄 이내 핵심 요약, 이어서 근거가 되는 세부 내용을 불릿으로 정리합니다.
5. 실무자가 바로 쓸 수 있도록 간결하고 단정적인 문장을 쓰되, 추정과 확인된 사실을 구분해 표기합니다.
6. 답변 마지막에 출처 목록을 다시 쓰지 마세요. (앱이 별도로 표시합니다)"""

NO_CONTEXT_MESSAGE = (
    "질문과 관련된 내용을 지식베이스에서 찾지 못했습니다. "
    "검색 범위(사이드바 문서 필터)를 넓히거나, 질문을 좀 더 구체적인 용어로 바꿔 보세요."
)


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


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

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
) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=model,
        messages=build_messages(question, hits, history),
        temperature=0.2,
        max_tokens=1200,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def suggest_followups(client: OpenAI, model: str, question: str, answer: str) -> list[str]:
    """답변 이후 이어서 물어볼 만한 질문 3개를 제안한다."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "데이터센터 산업 리서치 대화의 후속 질문 3개를 제안하세요. "
                    "각 줄에 질문 하나씩, 번호나 기호 없이, 25자 내외 한국어로만 출력합니다.",
                },
                {"role": "user", "content": f"직전 질문: {question}\n직전 답변 요약: {answer[:800]}"},
            ],
            temperature=0.6,
            max_tokens=150,
        )
        lines = [l.strip(" -*0123456789.") for l in resp.choices[0].message.content.splitlines()]
        return [l for l in lines if l][:3]
    except Exception:
        return []
