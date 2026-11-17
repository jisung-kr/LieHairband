"""LLM 역할별 호출 (답 판정, AI 턴, 정답 판정).

정보 분리 원칙: ai_turn()에는 AI 자신의 제시어를 절대 넘기지 않는다.
"""

import re
from dataclasses import dataclass, field

from .llm import LLM, LLMError

ANSWERS = ("예", "아니오", "애매함")


# ---------------------------------------------------------------- 답 판정


@dataclass
class AnswerResult:
    """answer는 진실한 답. 예/아니오 대사는 reaction_line()에서 최종 답 기준으로 따로 만든다."""

    answer: str  # 예 / 아니오 / 애매함 / 무효
    other_line: str  # 애매함 / 무효일 때의 대사


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "meaning": {"type": "string"},
        "fact": {"type": "string"},
        "answer": {"type": "string", "enum": ["예", "아니오", "애매함", "무효"]},
        "other_line": {"type": "string"},
    },
    "required": ["meaning", "fact", "answer", "other_line"],
}

ANSWER_SYSTEM = """너는 스무고개 게임에서 질문에 답하는 사람이다.
상대가 맞혀야 하는 제시어는 "{word}"(카테고리: {category})이다.
상대의 질문을 읽고, 제시어에 대한 일반 상식 기준으로 진실하게 판정해라.
제시어의 가장 대표적이고 흔한 모습을 기준으로 판단해라 (예: 사과는 "빨간색이야?"에 예).

meaning 규칙:
- 질문이 무엇을 묻는지 "제시어가 ~하는가?" 형태로 풀어 써라. 띄어쓰기가 틀렸거나 구어체여도 뜻을 정확히 읽어라.
- 문장 구조대로 읽어라. "~을/를 + 동사"는 동사구다.
  예: "음식을 집을 때 쓰는거야?" → "음식을 집다(잡다)"는 동사구 → meaning: "제시어가 음식을 집어 드는 데 쓰는 도구인가?"
- 몸이나 부품을 가리키는 말은 제시어의 부분으로 읽어라.
  예: 제시어가 탁자일 때 "다리가 있어?" → meaning: "탁자에 다리(받침) 부분이 있는가?"
- "그럼", "혹시" 같은 앞말은 무시해라.

fact 규칙:
- meaning에 대한 제시어의 실제 사실을 짧게 한 문장으로 적어라. 질문의 핵심에 직접 답해야 한다.
  (예: 제시어가 고양이이고 질문이 "알을 낳아?"면 → "고양이는 포유류라 새끼를 낳는다.")
- 생물학, 상식 지식을 정확하게 떠올려라. answer는 fact와 반드시 일치해야 한다.

answer 규칙:
- "예": 대부분의 사람이 그렇다고 동의할 때
- "아니오": 대부분의 사람이 아니라고 동의할 때
- "애매함": 기준이나 상황에 따라 답이 정말로 갈릴 때만 (예: 바나나에게 "비싸?"). 조금이라도 한쪽이 우세하면 예/아니오로 답해라.
- "무효": 예/아니오로 답할 수 없는 질문 (예: "무슨 색이야?", "어디서 살아?"),
  또는 제시어 자체를 직접 묻는 질문 (예: "{word}야?", "정답이 뭐야?")

other_line 규칙:
- answer가 "애매함"이면 애매하다는 짧은 반말 반응 (예: "글쎄, 경우에 따라 다르지.")
- answer가 "무효"면 왜 무효인지 짧게 안내 (예: "예/아니오로 답할 수 있게 물어봐 줘.", "정답은 정답 도전으로 말해줘.")
- 그 외에는 빈 문자열.
- 제시어나 동의어, 제시어에 대한 추가 힌트를 절대 넣지 마라."""

FALLBACK_LINES = {
    "예": "응, 맞아.",
    "아니오": "아니, 그건 아니야.",
    "애매함": "글쎄, 경우에 따라 다르지.",
    "무효": "예/아니오로 답할 수 있게 물어봐 줘.",
}


def answer_question(
    llm: LLM, word: str, category: str, synonyms: list[str], question: str
) -> AnswerResult:
    try:
        data = llm.chat_json(
            ANSWER_SYSTEM.format(word=word, category=category),
            f"질문: {question}",
            ANSWER_SCHEMA,
            temperature=0.1,
        )
        answer = data["answer"] if data["answer"] in (*ANSWERS, "무효") else "애매함"
    except LLMError:
        data, answer = {}, "애매함"

    line = str(data.get("other_line", "")).strip()
    if not line or leaks_word(line, word, synonyms):
        line = FALLBACK_LINES.get(answer, FALLBACK_LINES["애매함"])
    return AnswerResult(answer, line)


REACTION_SCHEMA = {
    "type": "object",
    "properties": {"line": {"type": "string"}},
    "required": ["line"],
}

REACTION_SYSTEM = """너는 스무고개 게임에서 질문에 답하는 캐릭터다.
질문과 네가 하기로 한 답이 주어진다. 그 답을 말하는 짧은 반말 대사 1문장을 line에 적어라.
- 대사의 뜻은 반드시 주어진 답과 같아야 한다.
  · 예 → "응, 맞아!", "그렇지~", "오, 좋은 질문인데? 맞아."
  · 아니오 → "아니, 전혀.", "땡! 그건 아니야.", "음, 그건 아니지."
- 매번 표현을 조금씩 바꿔라.
- 새로운 정보나 힌트는 넣지 마라."""


def reaction_line(llm: LLM, question: str, final_answer: str) -> str:
    """최종 답(거짓말이면 뒤집힌 답)에 맞는 대사. 제시어를 모르는 호출이라 진실을 흘릴 수 없다."""
    try:
        data = llm.chat_json(
            REACTION_SYSTEM,
            f"질문: {question}\n답: {final_answer}",
            REACTION_SCHEMA,
            temperature=0.8,
        )
        line = _NOTE_PATTERN.sub("", str(data["line"])).strip()
    except LLMError:
        line = ""
    return line or FALLBACK_LINES[final_answer]


def leaks_word(text: str, word: str, synonyms: list[str]) -> bool:
    """대사에 제시어/동의어가 들어있는지. 한 글자 단어는 오탐이 많아 어절 단위로 본다."""
    compact = _compact(text)
    tokens = [_compact(t) for t in text.split()]
    for w in (word, *synonyms):
        cw = _compact(w)
        if not cw:
            continue
        if len(cw) >= 2:
            if cw in compact:
                return True
        elif any(t.startswith(cw) and len(t) - len(cw) <= 1 for t in tokens):
            return True
    return False


# ---------------------------------------------------------------- AI 턴


@dataclass
class AITurn:
    action: str  # ask / guess
    question: str = ""
    guess: str = ""
    line: str = ""
    candidates: list[str] = field(default_factory=list)
    # 질문에 대해 후보 기준으로 예상되는 답 (의심 계산용)
    expected: str = "모름"
    confidence: float = 0.0


AI_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "candidates": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string", "enum": ["ask", "guess"]},
        "question": {"type": "string"},
        "expected": {"type": "string", "enum": ["예", "아니오", "모름"]},
        "confidence": {"type": "number"},
        "guess": {"type": "string"},
        "line": {"type": "string"},
    },
    "required": [
        "analysis", "candidates", "action", "question", "expected", "confidence", "guess", "line"
    ],
}

AI_TURN_SYSTEM = """너는 스무고개 게임의 플레이어다. 너는 네 제시어를 모르고, 상대의 예/아니오 답을 통해 맞혀야 한다.
네 제시어의 카테고리는 "{category}"이다.

해야 할 일:
0. analysis에 지금까지 알아낸 조건을 한 줄로 요약해라 (예: "살아있음, 날지 못함, 물에서 지냄").
1. analysis의 조건을 **전부** 만족하는 후보 단어를 최대 8개 candidates에 적어라. 가능성 높은 순서로.
   · 조건 하나라도 어긋나는 단어는 넣지 마라.
   · 후보는 누구나 아는 흔한 한국어 단어로만 적어라 (외국어, 한자 금지).
2. action을 정해라.
   - "ask": 후보를 절반 가까이 나눌 수 있는 예/아니오 질문을 question에 적어라.
     · 반드시 예/아니오로 답할 수 있는 질문이어야 한다.
     · 짧은 반말로 물어라 (예: "사람이 쓰는 물건이야?", "집에서 볼 수 있어?").
     · 특정 단어를 직접 묻지 마라 (예: "그거 ○○야?" 형태 금지). 그건 정답 도전으로 해야 한다.
     · 이미 한 질문을 반복하지 마라.
     · expected에는 지금 후보들 기준으로 이 질문의 답이 무엇일 것 같은지 적어라 (예/아니오/모름).
       confidence에는 그 예상의 확신도를 0~1 사이 숫자로 적어라 (후보 대부분이 같은 답이면 높게).
   - "guess": 정답이라고 생각하는 단어 하나를 guess에 적어라.
     · 후보가 1~2개로 좁혀졌거나, 남은 질문이 거의 없을 때만 정답 도전해라.
     · 이미 틀린 단어는 다시 말하지 마라.
3. line에는 질문이나 도전 앞에 붙일 짧은 반말 대사를 적어라 (예: "좋아, 이번엔 이걸 물어볼게.").
사용하지 않는 필드는 빈 문자열로 둬라.

주의: 상대는 가끔 거짓말을 한다. "(의심스러움)" 표시가 붙은 답은 거짓말일 수 있으니
그 답 하나만 믿고 후보를 지우지 마라. "(간파: 진짜 답은 ~)" 표시는 확실한 사실이다."""


# LLM이 쓸 만한 질문을 못 만들 때 쓰는 범용 질문
FALLBACK_QUESTIONS = (
    "살아있는 거야?",
    "집 안에서 볼 수 있어?",
    "손에 들 수 있는 크기야?",
    "사람이 만든 거야?",
    "매일 볼 수 있어?",
    "소리를 낼 수 있어?",
    "무거워?",
)


def ai_turn(
    llm: LLM,
    category: str,
    history: list[tuple[str, str]],
    wrong_guesses: list[str],
    tokens_left: int,
    guesses_left: int,
    notes: list[str] | None = None,
) -> AITurn:
    """notes: history와 같은 길이. 각 답에 붙일 표시 (예: "의심스러움"). 빈 문자열이면 표시 없음."""
    notes = notes or [""] * len(history)
    lines = [f"남은 질문 토큰: {tokens_left}개, 남은 정답 도전: {guesses_left}회"]
    if history:
        lines.append("지금까지의 질문과 답:")
        lines += [
            f"- Q: {q} → A: {a}" + (f" ({n})" if n else "") for (q, a), n in zip(history, notes)
        ]
    else:
        lines.append("아직 질문한 적이 없다.")
    if wrong_guesses:
        lines.append(f"이미 틀린 정답: {', '.join(wrong_guesses)}")
    if tokens_left == 0:
        lines.append("질문 토큰이 없으니 반드시 guess 해야 한다.")

    asked = {_compact(q) for q, _ in history}
    data: dict = {"candidates": [], "action": "ask", "question": "", "guess": "", "line": ""}
    for attempt in range(2):
        try:
            data = llm.chat_json(
                AI_TURN_SYSTEM.format(category=category),
                "\n".join(lines),
                AI_TURN_SCHEMA,
                temperature=0.4 + 0.3 * attempt,
            )
        except LLMError:
            break
        # 기록용 표시("(의심스러움)", "(간파: ...)")를 질문에 따라 쓰는 경우가 있어 지운다
        raw_q = _NOTE_PATTERN.sub("", str(data.get("question", ""))).strip()
        data["question"] = raw_q
        q = _compact(raw_q)
        if data.get("action") != "ask" or (q and q not in asked):
            break
        # 이미 한 질문이거나 빈 질문 → 한 번 더 요청
        data["question"] = ""
        lines.append(f'"{raw_q}"는 이미 했거나 비어 있는 질문이다. 다른 질문을 해라.')

    wrong = {_compact(g) for g in wrong_guesses}
    candidates = [
        c.strip()
        for c in data.get("candidates", [])
        if _HANGUL_WORD.match(c.strip()) and _compact(c) not in wrong
    ]
    action = data.get("action", "ask")
    question = str(data.get("question", "")).strip()
    guess = str(data.get("guess", "")).strip()
    line = _NOTE_PATTERN.sub("", str(data.get("line", ""))).strip()
    expected = data.get("expected", "모름")
    expected = expected if expected in ("예", "아니오") else "모름"
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    # 규칙 강제
    if tokens_left == 0 and guesses_left > 0:
        action = "guess"
    if action == "guess" and guesses_left == 0:
        action = "ask"
    if action == "ask" and not question:
        expected, confidence = "모름", 0.0  # 범용 질문은 예상이 없다
        question = next((f for f in FALLBACK_QUESTIONS if _compact(f) not in asked), "")
        if not question:
            action = "guess" if guesses_left > 0 else "ask"
            question = question or "일상생활에서 자주 볼 수 있는 거야?"
    if action == "guess" and (not guess or _compact(guess) in wrong):
        guess = candidates[0] if candidates else "모르겠다"

    return AITurn(
        action=action,
        question=question,
        guess=guess,
        line=line,
        candidates=candidates,
        expected=expected,
        confidence=confidence,
    )


# ---------------------------------------------------------------- 정답 판정

GUESS_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}},
    "required": ["correct"],
}

GUESS_SYSTEM = """정답 판정관이다. 정답은 "{word}"이다.
추측한 단어가 정답과 **같은 대상**을 가리키면 correct=true다 (예: "개"와 "강아지", "달걀"과 "계란").
상위 개념, 하위 개념, 비슷하지만 다른 대상은 false다 (예: 정답이 "강아지"일 때 "동물", "늑대", "푸들"은 false)."""

# 기록용 표시, qwen3 제어 토큰(/no_think 등) 찌꺼기
_NOTE_PATTERN = re.compile(r"\s*\((?:의심|간파)[^)]*\)|\s*/no\w*|\s*/think\w*")
_HANGUL_WORD = re.compile(r"^[가-힣A-Za-z0-9 ]+$")
_ENDINGS = re.compile(r"(이야|야|입니다|이에요|예요|인가|일까|이지)$")


def _compact(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).lower()


def check_guess(llm: LLM, word: str, synonyms: list[str], guess: str) -> bool:
    raw = _compact(guess)
    if not raw:
        return False
    variants = {raw, _ENDINGS.sub("", raw)}
    answers = {_compact(w) for w in (word, *synonyms)}
    if variants & answers:
        return True
    try:
        data = llm.chat_json(
            GUESS_SYSTEM.format(word=word), f"추측: {guess}", GUESS_SCHEMA, temperature=0.0
        )
        return bool(data["correct"])
    except LLMError:
        return False
