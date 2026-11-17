"""AI 거짓말 결정과 텔(버릇) 선택. 거짓말 여부는 LLM이 아니라 여기서 정한다."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Difficulty:
    name: str
    tell_seconds: float  # 텔(동요 애니메이션) 길이
    tell_scale: float  # 텔 움직임 크기 배율 (1.0 = 보통). 작을수록 알아채기 어렵다
    lies: int  # 게임당 거짓말 횟수


# 페이크 텔은 없다. 대신 캐릭터가 늘 미세하게 숨 쉬듯 움직여서(UI), 그 속에서 순간적인 동요를 알아채야 한다.
DIFFICULTIES: dict[str, Difficulty] = {
    "쉬움": Difficulty("쉬움", 1.0, 1.6, 1),
    "보통": Difficulty("보통", 0.7, 1.0, 2),
    "어려움": Difficulty("어려움", 0.45, 0.55, 3),
}
DEFAULT_DIFFICULTY = "보통"

NO_LIE_FIRST_N = 2  # 첫 N개 질문은 거짓말 금지
MAX_LIE_CHANCE = 0.6


class LieSystem:
    def __init__(self, difficulty: Difficulty, rng: random.Random, lie_rate: float = 1.0):
        """lie_rate: 캐릭터 성향 배율 (1.0 = 보통)."""
        self.difficulty = difficulty
        self.rng = rng
        self.lie_rate = lie_rate
        self.lies_left = difficulty.lies
        self.last_was_lie = False

    def can_lie(self, answer: str, answered_before: int) -> bool:
        return (
            self.lies_left > 0
            and answer in ("예", "아니오")
            and answered_before >= NO_LIE_FIRST_N
            and not self.last_was_lie
        )

    def lie_chance(self, remaining_questions: int) -> float:
        """남은 거짓말을 남은 질문에 고르게 퍼뜨리는 확률."""
        remaining = max(1, remaining_questions)
        return min(MAX_LIE_CHANCE, self.lies_left / remaining * 1.5 * self.lie_rate)

    def decide(self, answer: str, answered_before: int, remaining_questions: int) -> bool:
        """이번 답에 거짓말할지 결정하고 상태를 갱신한다."""
        lie = self.can_lie(answer, answered_before) and (
            self.rng.random() < self.lie_chance(remaining_questions)
        )
        if lie:
            self.lies_left -= 1
        self.last_was_lie = lie
        return lie

    def tell_for(self, is_lie: bool) -> str | None:
        """거짓말할 때만 텔(동요)이 나온다."""
        return "tell" if is_lie else None


def flip(answer: str) -> str:
    return {"예": "아니오", "아니오": "예"}[answer]
