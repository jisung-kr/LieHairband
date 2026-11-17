"""게임 진행 상태 머신. UI를 모른다.

턴 흐름:
  PLAYER_TURN ─(질문/정답 도전)→ AI_TURN ─(AI 질문)→ AWAIT_PLAYER_ANSWER ─(플레이어 답)→ PLAYER_TURN
                                         └(AI 정답 도전)→ PLAYER_TURN
  한쪽이 더 이상 행동할 수 없으면(토큰 0 + 정답 기회 0) 그 쪽 턴은 건너뛴다.

간파 (양방향):
  - 플레이어: AI의 가장 최근 답(예/아니오)은 다음 질문이나 정답 도전을 하기 전까지 언제든 간파할 수 있다.
  - AI: 플레이어가 답할 때마다 의심도를 계산하고, 임계값을 넘으면 즉시 간파한다.

정보 분리:
  심판 판정(AI 제시어 기준의 진실)은 엔진 안에만 있고, AI 쪽 LLM 호출(ai_turn)에는 절대 넘기지 않는다.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from . import roles, suspicion
from .lie_system import DEFAULT_DIFFICULTY, DIFFICULTIES, NO_LIE_FIRST_N, LieSystem, flip
from .llm import LLM
from .words import all_words, synonyms

DEFAULT_TOKENS = 15
DEFAULT_GUESSES = 3
CHALLENGE_REWARD = 2  # 사용한 토큰 환급 1 + 보너스 1
CHALLENGE_PENALTY = 1
GUESS_WRONG_LINES = ("땡~ 틀렸어!", "아쉽네, 그건 아니야.", "후훗, 땡이야~", "아니야~ 다시 생각해 봐.")


def josa(word: str, with_batchim: str, without_batchim: str) -> str:
    """마지막 글자 받침 유무에 따라 조사 선택 (예: 책+이었 / 의자+였)."""
    last = word[-1] if word else ""
    if "가" <= last <= "힣":
        return with_batchim if (ord(last) - 0xAC00) % 28 else without_batchim
    return without_batchim


def was(word: str) -> str:
    return josa(word, "이었습니다", "였습니다")


class Phase(Enum):
    PLAYER_TURN = "player_turn"
    AI_TURN = "ai_turn"
    AWAIT_PLAYER_ANSWER = "await_player_answer"
    GAME_OVER = "game_over"


class GameError(Exception):
    """룰상 허용되지 않는 행동."""


@dataclass
class Side:
    word: str
    category: str
    synonyms: list[str]
    tokens: int
    guesses_left: int
    wrong_guesses: list[str] = field(default_factory=list)
    # 이 쪽이 자기 제시어에 대해 받은 질문/답 기록 (받은 그대로. 거짓말 포함)
    history: list[tuple[str, str]] = field(default_factory=list)

    @property
    def can_act(self) -> bool:
        return self.tokens > 0 or self.guesses_left > 0


@dataclass
class AIAnswer:
    """AI가 플레이어 질문에 한 답의 내부 기록. truth/lie는 화면에 노출하지 않는다."""

    question: str
    truth: str
    final: str
    lie: bool
    history_index: int
    caught: bool = False


@dataclass
class PendingQuestion:
    """AI가 플레이어에게 한 질문. referee_truth는 AI에게 절대 전달하지 않는다."""

    question: str
    expected: str
    confidence: float
    referee_truth: str  # 예 / 아니오 / 애매함 / 무효
    asked_at: float


@dataclass
class PlayerAnswer:
    question: str
    answer: str
    lie: bool
    seconds: float
    suspicion: int
    caught: bool = False


@dataclass
class AnswerOutcome:
    committed: bool
    warning: str | None = None
    suspicion: suspicion.Suspicion | None = None


@dataclass
class Event:
    """화면에 보여줄 기록 한 줄."""

    actor: str  # player / ai / system
    kind: str  # question / answer / guess / challenge / challenge_result / reaction / result / info
    text: str
    answer: str = ""
    tell: str | None = None  # AI 답변 연출용: tell / fake1 / fake2 / None
    mood: str = ""  # 패널 연출용: trust / press / accuse / player_ok / player_fail / ai_ok / ai_fail


@dataclass
class ChallengeResult:
    success: bool
    truth: str


class Game:
    def __init__(
        self,
        llm: LLM,
        tokens: int = DEFAULT_TOKENS,
        guesses: int = DEFAULT_GUESSES,
        difficulty: str = DEFAULT_DIFFICULTY,
        rng: random.Random | None = None,
        words: tuple[tuple[str, str], tuple[str, str]] | None = None,
        lie_system: LieSystem | None = None,
        accuse_at: int = suspicion.DEFAULT_ACCUSE_AT,
        clock: Callable[[], float] = time.monotonic,
    ):
        """words: ((플레이어 카테고리, 플레이어 제시어), (AI 카테고리, AI 제시어)). 테스트용."""
        self.llm = llm
        self.rng = rng or random.Random()
        self.clock = clock
        if words is None:
            words = tuple(self.rng.sample(all_words(), 2))
        (p_cat, p_word), (a_cat, a_word) = words
        self.player = Side(p_word, p_cat, synonyms(p_cat, p_word), tokens, guesses)
        self.ai = Side(a_word, a_cat, synonyms(a_cat, a_word), tokens, guesses)
        self.difficulty = DIFFICULTIES[difficulty]
        self.lies = lie_system or LieSystem(self.difficulty, self.rng)
        self.accuse_at = accuse_at

        self.phase = Phase.PLAYER_TURN
        self.winner: str | None = None  # player / ai / draw
        self.pending: PendingQuestion | None = None
        self.ai_answers: list[AIAnswer] = []
        self.challengeable: AIAnswer | None = None
        self.player_lies_left = self.difficulty.lies
        self.player_answers: list[PlayerAnswer] = []
        self.ai_notes: list[str] = []  # ai.history와 같은 길이. AI 추리용 표시
        self._last_ai_line = ""
        self.stats = {"challenge_ok": 0, "challenge_fail": 0, "ai_challenge_ok": 0, "ai_challenge_fail": 0}
        self.events: list[Event] = [
            Event(
                "system",
                "info",
                f"게임 시작! (난이도: {self.difficulty.name}) 먼저 질문하세요. "
                f"AI와 나는 각각 최대 {self.difficulty.lies}번 거짓말할 수 있습니다.",
            )
        ]

    # ------------------------------------------------------------ 조회

    @property
    def can_challenge(self) -> bool:
        return self.challengeable is not None and self.phase in (
            Phase.PLAYER_TURN,
            Phase.AWAIT_PLAYER_ANSWER,
        )

    @property
    def player_can_lie(self) -> bool:
        return (
            self.phase == Phase.AWAIT_PLAYER_ANSWER
            and self.player_lies_left > 0
            and len(self.ai.history) >= NO_LIE_FIRST_N
        )

    @property
    def pending_ai_question(self) -> str | None:
        return self.pending.question if self.pending else None

    # ------------------------------------------------------------ 플레이어 행동

    def player_ask(self, question: str) -> AIAnswer | None:
        """무효 질문이면 None."""
        self._require(Phase.PLAYER_TURN)
        question = question.strip()
        if not question:
            raise GameError("질문을 입력하세요.")
        if self.player.tokens <= 0:
            raise GameError("질문 토큰이 없습니다. 정답 도전만 할 수 있어요.")

        result = roles.answer_question(
            self.llm, self.player.word, self.player.category, self.player.synonyms, question
        )
        self.events.append(Event("player", "question", question))

        if result.answer == "무효":
            self.events.append(Event("ai", "answer", result.other_line, answer="무효"))
            self.events.append(Event("system", "info", "무효 질문이라 토큰을 쓰지 않았습니다. 다시 질문하세요."))
            return None  # 턴 유지

        self.challengeable = None  # 새 질문을 했으니 이전 답은 더 이상 간파 불가
        lie = self.lies.decide(
            result.answer,
            answered_before=len(self.player.history),
            remaining_questions=self.player.tokens,
        )
        final = flip(result.answer) if lie else result.answer
        record = AIAnswer(question, result.answer, final, lie, len(self.player.history))
        self.ai_answers.append(record)
        self.player.history.append((question, final))
        # 예/아니오 대사는 진실/거짓 상관없이 항상 같은 방식으로 만든다 (응답 시간 동일)
        line = (
            roles.reaction_line(self.llm, question, final)
            if final in ("예", "아니오")
            else result.other_line
        )
        self.events.append(Event("ai", "answer", line, answer=final, tell=self.lies.tell_for(lie)))

        if final == "애매함":
            self.events.append(Event("system", "info", "애매한 질문이라 토큰을 돌려받았습니다."))
        else:
            self.player.tokens -= 1
            self.challengeable = record
        self._end_turn("player")
        return record

    def player_guess(self, guess: str) -> bool:
        self._require(Phase.PLAYER_TURN)
        guess = guess.strip()
        if not guess:
            raise GameError("정답을 입력하세요.")
        if self.player.guesses_left <= 0:
            raise GameError("정답 도전 기회가 없습니다.")

        self.challengeable = None
        correct = roles.check_guess(self.llm, self.player.word, self.player.synonyms, guess)
        self.events.append(Event("player", "guess", guess))
        self._resolve_guess(self.player, "player", guess, correct)
        return correct

    def player_challenge(self) -> ChallengeResult:
        """AI의 가장 최근 답을 거짓말이라고 지목한다."""
        if not self.can_challenge:
            raise GameError("지금은 간파할 답이 없습니다.")
        record = self.challengeable
        assert record is not None
        self.challengeable = None
        self.events.append(Event("player", "challenge", f"간파! \"{record.question}\"의 답은 거짓말이야!"))

        if record.lie:
            record.caught = True
            self.player.history[record.history_index] = (record.question, record.truth)
            self.player.tokens += CHALLENGE_REWARD
            self.stats["challenge_ok"] += 1
            self.events.append(
                Event(
                    "system",
                    "challenge_result",
                    f"⚡ 간파 성공! \"{record.question}\"의 진짜 답은 '{record.truth}'입니다. "
                    f"질문 토큰 +{CHALLENGE_REWARD}",
                    mood="player_ok",
                )
            )
        else:
            self.player.tokens = max(0, self.player.tokens - CHALLENGE_PENALTY)
            self.stats["challenge_fail"] += 1
            self.events.append(
                Event(
                    "system",
                    "challenge_result",
                    f"💨 간파 실패! 그 답은 진실이었습니다. 질문 토큰 -{CHALLENGE_PENALTY}",
                    mood="player_fail",
                )
            )
            if self.phase == Phase.PLAYER_TURN and not self.player.can_act:
                self._end_turn("player")
        return ChallengeResult(record.lie, record.truth)

    def check_player_answer(self, answer: str, lie: bool) -> str | None:
        """심판 판정과 비교해 경고 문구를 돌려준다. 문제 없으면 None."""
        assert self.pending is not None
        truth = self.pending.referee_truth
        if truth not in roles.ANSWERS:  # AI 질문이 무효였다면 판정하지 않는다
            return None
        if answer == "애매함":
            return None if truth == "애매함" else "이 질문은 예/아니오로 답할 수 있어요."
        if truth == "애매함":
            return None  # 어느 쪽으로 답해도 거짓말로 보지 않음
        if not lie and answer != truth:
            return "사실과 다른 답 같아요. 거짓말 토글을 켜거나 답을 고쳐주세요."
        if lie and answer == truth:
            return "이건 사실인 것 같아요. 거짓말하려면 반대로 답하세요."
        return None

    def player_answer(self, answer: str, lie: bool = False, force: bool = False) -> AnswerOutcome:
        """AI의 질문에 답한다. 심판 경고가 있고 force=False면 아무것도 바꾸지 않고 경고만 돌려준다."""
        self._require(Phase.AWAIT_PLAYER_ANSWER)
        if answer not in roles.ANSWERS:
            raise GameError(f"답은 {roles.ANSWERS} 중 하나여야 합니다.")
        if lie and answer == "애매함":
            raise GameError("애매함으로는 거짓말할 수 없어요.")
        if lie and not self.player_can_lie:
            raise GameError(
                "지금은 거짓말할 수 없어요."
                + (" (남은 거짓말 없음)" if self.player_lies_left <= 0 else " (첫 2개 답은 거짓말 금지)")
            )
        pending = self.pending
        assert pending is not None

        warning = self.check_player_answer(answer, lie)
        if warning and not force:
            return AnswerOutcome(committed=False, warning=warning)

        # 진실이 애매한 질문에서 토글을 켠 경우는 거짓말로 치지 않는다 (강제 진행 제외)
        if lie and pending.referee_truth == "애매함" and not force:
            lie = False
            self.events.append(Event("system", "info", "애매한 질문이라 거짓말로 치지 않았어요. (거짓말 횟수 유지)"))
        if lie:
            self.player_lies_left -= 1

        seconds = max(0.0, self.clock() - pending.asked_at)
        caught_lies = sum(1 for a in self.player_answers if a.caught)
        susp = suspicion.evaluate(
            answer=answer,
            expected=pending.expected,
            confidence=pending.confidence,
            answer_seconds=seconds,
            previous_seconds=[a.seconds for a in self.player_answers],
            # AI가 아는 정보만: 첫 2개 답 규칙 + 간파로 확인된 거짓말 수
            lie_possible=len(self.ai.history) >= NO_LIE_FIRST_N
            and self.difficulty.lies - caught_lies > 0,
            ai_tokens=self.ai.tokens,
            accuse_at=self.accuse_at,
            rng=self.rng,
        )
        record = PlayerAnswer(pending.question, answer, lie, seconds, susp.score)
        self.player_answers.append(record)
        self.ai.history.append((pending.question, answer))
        self.ai_notes.append("의심스러움" if susp.level == "press" else "")
        self.events.append(Event("player", "answer", answer, answer=answer, mood="lie" if lie else ""))
        self.events.append(Event("ai", "reaction", susp.line, mood=susp.level))
        self.pending = None

        if answer == "애매함":
            self.ai.tokens += 1  # 질문할 때 쓴 토큰 환급
            self.events.append(Event("system", "info", "애매함 답이라 AI가 토큰을 돌려받았습니다."))
        if susp.accuse:
            self._ai_challenge(record)
        self._end_turn("ai")
        return AnswerOutcome(committed=True, suspicion=susp)

    # ------------------------------------------------------------ AI 행동

    def ai_act(self) -> roles.AITurn:
        self._require(Phase.AI_TURN)
        turn = roles.ai_turn(
            self.llm,
            self.ai.category,
            self.ai.history,
            self.ai.wrong_guesses,
            self.ai.tokens,
            self.ai.guesses_left,
            notes=self.ai_notes,
        )
        if turn.action == "ask":
            self.ai.tokens -= 1
            # 심판 판정은 미리 계산해 둔다 (플레이어가 답하면 바로 확인 가능). AI에게는 전달하지 않는다.
            referee = roles.answer_question(
                self.llm, self.ai.word, self.ai.category, self.ai.synonyms, turn.question
            )
            # 같은 서두 대사가 연달아 반복되면 생략한다
            line = turn.line if turn.line and turn.line != self._last_ai_line else ""
            self._last_ai_line = turn.line
            text = f"{line} {turn.question}".strip()
            self.events.append(Event("ai", "question", text))
            self.pending = PendingQuestion(
                turn.question, turn.expected, turn.confidence, referee.answer, self.clock()
            )
            self.phase = Phase.AWAIT_PLAYER_ANSWER
        else:
            correct = roles.check_guess(self.llm, self.ai.word, self.ai.synonyms, turn.guess)
            text = f"{turn.line} 정답은... {turn.guess}!".strip() if turn.line else f"정답은... {turn.guess}!"
            self.events.append(Event("ai", "guess", text))
            self._resolve_guess(self.ai, "ai", turn.guess, correct)
        return turn

    def _ai_challenge(self, record: PlayerAnswer) -> None:
        if record.lie:
            record.caught = True
            truth = flip(record.answer)
            self.ai.history[-1] = (record.question, truth)
            self.ai_notes[-1] = f"간파: 진짜 답은 {truth}"
            self.ai.tokens += CHALLENGE_REWARD
            self.stats["ai_challenge_ok"] += 1
            self.events.append(
                Event(
                    "system",
                    "challenge_result",
                    f"⚡ AI의 간파 성공! 거짓말이 들켰습니다. \"{record.question}\"의 진짜 답 '{truth}'{josa(truth, '이', '가')} "
                    f"AI에게 공개됩니다. AI 질문 토큰 +{CHALLENGE_REWARD}",
                    mood="ai_ok",
                )
            )
        else:
            self.ai.tokens = max(0, self.ai.tokens - CHALLENGE_PENALTY)
            self.stats["ai_challenge_fail"] += 1
            self.events.append(
                Event(
                    "system",
                    "challenge_result",
                    f"💨 AI의 간파 실패! 진실이었는데 의심받았네요. AI 질문 토큰 -{CHALLENGE_PENALTY}",
                    mood="ai_fail",
                )
            )

    # ------------------------------------------------------------ 내부

    def _resolve_guess(self, side: Side, actor: str, guess: str, correct: bool) -> None:
        name = "플레이어" if actor == "player" else "AI"
        side.guesses_left -= 1
        if correct:
            self.events.append(
                Event("system", "result", f"🎉 {name} 정답! 제시어는 '{side.word}'{was(side.word)}.")
            )
            self._finish(actor)
            return
        side.wrong_guesses.append(guess)
        side.tokens = max(0, side.tokens - 1)
        self.events.append(
            Event("system", "result", f"❌ {name} 오답 ('{guess}'). 질문 토큰 1개를 잃었습니다.")
        )
        if actor == "player":
            # 플레이어가 틀리면 AI 가 먼저 틀렸다고 말한 뒤에 자기 차례로 넘어간다
            self.events.append(Event("ai", "reaction", self.rng.choice(GUESS_WRONG_LINES), mood="guess_wrong"))
        self._end_turn(actor)

    def _end_turn(self, actor: str) -> None:
        if self.phase == Phase.GAME_OVER:
            return
        if not self.player.can_act and not self.ai.can_act:
            self.events.append(
                Event(
                    "system",
                    "result",
                    f"🤝 무승부! 내 제시어는 '{self.player.word}', "
                    f"AI 제시어는 '{self.ai.word}'{was(self.ai.word)}.",
                )
            )
            self._finish("draw")
            return
        other = "ai" if actor == "player" else "player"
        nxt = other if self._side(other).can_act else actor
        self.phase = Phase.AI_TURN if nxt == "ai" else Phase.PLAYER_TURN

    def _finish(self, winner: str) -> None:
        self.winner = winner
        self.phase = Phase.GAME_OVER
        self.challengeable = None
        self.pending = None
        if winner == "ai":
            self.events.append(
                Event("system", "info", f"내 제시어는 '{self.player.word}'{was(self.player.word)}.")
            )
        lies = [a for a in self.ai_answers if a.lie]
        if lies:
            lines = [
                f"- \"{a.question}\" → '{a.final}'라고 했지만 진실은 '{a.truth}'"
                + (" (간파 성공)" if a.caught else " (속음)")
                for a in lies
            ]
            self.events.append(Event("system", "info", "🤥 AI의 거짓말 공개\n" + "\n".join(lines)))
        else:
            self.events.append(Event("system", "info", "🤥 이번 판에서 AI는 거짓말을 하지 않았습니다."))
        my_lies = [a for a in self.player_answers if a.lie]
        if my_lies:
            fooled = sum(1 for a in my_lies if not a.caught)
            self.events.append(
                Event("system", "info", f"🎭 내 거짓말 {len(my_lies)}번 중 AI를 {fooled}번 속였습니다.")
            )

    def _side(self, actor: str) -> Side:
        return self.player if actor == "player" else self.ai

    def _require(self, phase: Phase) -> None:
        if self.phase != phase:
            raise GameError(f"지금은 이 행동을 할 수 없습니다. (현재: {self.phase.value})")
