"""AI가 플레이어 답을 얼마나 의심하는지 계산한다.

LLM이 아니라 코드가 점수를 낸다 (일관성, 밸런스 조정 용이).
사용하는 정보는 AI가 알 수 있는 것뿐이다. 심판 판정(진실)은 절대 쓰지 않는다.
"""

import random
from dataclasses import dataclass, field
from statistics import median

# 점수 배분 (밸런스 조정 대상)
BASE = 10
MISMATCH_MAX = 65  # 예상과 반대 답일 때, 확신도 1.0 기준
MATCH_BONUS = -10  # 예상과 같은 답
SLOW_ANSWER = 15  # 평소보다 오래 망설임
SLOW_RATIO = 1.8
SLOW_MIN_SECONDS = 3.0
LATE_GAME = 10  # 거짓말이 남았을 수 있는데 후반부
LATE_GAME_TOKENS = 6

PRESS_AT = 35  # 이 이상이면 압박 대사
DEFAULT_ACCUSE_AT = 60  # 이 이상이면 간파 선언 (캐릭터별로 다르게 할 예정)
BLUFF_RATE = 0.15  # 의심이 낮아도 떠보는 대사

TRUST_LINES = ("흠, 그렇구나. 일단 믿어볼게.", "오케이, 알겠어.", "그래? 기억해 둘게.", "좋아, 그렇다 치자.")
PRESS_LINES = (
    "진짜? 좀 수상한데...",
    "음... 정말 그게 맞아?",
    "방금 표정, 뭔가 숨기는 것 같은데?",
)
SLOW_LINES = ("방금 좀 망설인 것 같은데?", "대답이 좀 늦었네? 흐음...")
BLUFF_LINES = ("혹시 지금 거짓말한 거 아니야? ...농담이야.", "내 눈은 못 속인다고~")
ACCUSE_LINES = ("그거 거짓말이지!", "딱 걸렸어! 거짓말이야!", "아니, 그건 아닐 텐데? 간파!")


@dataclass
class Suspicion:
    score: int
    accuse: bool
    line: str
    reasons: list[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        if self.accuse:
            return "accuse"
        return "press" if self.score >= PRESS_AT else "trust"


def evaluate(
    *,
    answer: str,
    expected: str,
    confidence: float,
    answer_seconds: float | None,
    previous_seconds: list[float],
    lie_possible: bool,
    ai_tokens: int,
    accuse_at: int = DEFAULT_ACCUSE_AT,
    rng: random.Random,
) -> Suspicion:
    """
    answer: 플레이어 답 / expected: AI가 질문할 때 예상한 답 (예/아니오/모름)
    lie_possible: 룰상 플레이어가 이번에 거짓말할 수 있었는지 (AI도 아는 정보)
    """
    if not lie_possible or answer == "애매함":
        return Suspicion(0, False, rng.choice(TRUST_LINES), ["거짓말이 불가능한 답"])

    score, reasons = BASE, []
    if expected in ("예", "아니오"):
        if answer != expected:
            add = round(MISMATCH_MAX * max(0.0, min(1.0, confidence)))
            score += add
            reasons.append(f"예상({expected})과 반대 +{add}")
        else:
            score += MATCH_BONUS
            reasons.append(f"예상과 일치 {MATCH_BONUS}")

    slow = False
    if answer_seconds is not None and previous_seconds:
        typical = median(previous_seconds)
        if answer_seconds >= SLOW_MIN_SECONDS and answer_seconds > typical * SLOW_RATIO:
            score += SLOW_ANSWER
            slow = True
            reasons.append(f"답변 지연 {answer_seconds:.1f}s (평소 {typical:.1f}s) +{SLOW_ANSWER}")

    if ai_tokens <= LATE_GAME_TOKENS:
        score += LATE_GAME
        reasons.append(f"후반부 +{LATE_GAME}")

    score = max(0, min(100, score))
    accuse = score >= accuse_at
    if accuse:
        line = rng.choice(ACCUSE_LINES)
    elif score >= PRESS_AT:
        line = rng.choice(SLOW_LINES if slow else PRESS_LINES)
    elif rng.random() < BLUFF_RATE:
        line = rng.choice(BLUFF_LINES)
    else:
        line = rng.choice(TRUST_LINES)
    return Suspicion(score, accuse, line, reasons)
