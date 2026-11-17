"""게임 룰 테스트. 가짜 LLM을 써서 Ollama 없이 돌아간다."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game import roles  # noqa: E402
from game.engine import Game, GameError, Phase  # noqa: E402
from game.lie_system import DIFFICULTIES, LieSystem  # noqa: E402


class FakeLLM:
    """스키마 종류를 보고 미리 정한 응답을 돌려준다."""

    def __init__(self, answer="예", ai_action=None, guess_correct=False, referee="무효"):
        """answer: AI가 플레이어 질문에 판정하는 진실한 답
        referee: 심판이 AI 질문에 판정하는 진실한 답 ("무효"면 심판 경고 없음)"""
        self.answer = answer
        self.referee = referee
        self.ai_action = ai_action or {"action": "ask", "question": "살아있어?", "guess": ""}
        self.guess_correct = guess_correct
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system, user, schema, temperature=0.3):
        self.calls.append((system, user))
        props = schema["properties"]
        if "answer" in props:
            is_referee = '"펭귄"' in system  # AI 제시어 기준 판정 = 심판
            return {"answer": self.referee if is_referee else self.answer, "other_line": "글쎄."}
        if "line" in props and len(props) == 1:  # 반응 대사
            return {"line": "응!" if user.endswith("답: 예") else "아니!"}
        if "action" in props:
            out = {"analysis": "", "candidates": ["펭귄", "물개"], "line": "", **self.ai_action}
            if out["question"]:  # 같은 질문 반복은 엔진이 막으므로 매번 다르게
                out["question"] = f"{out['question']} ({len(self.calls)})"
            return out
        if "correct" in props:
            return {"correct": self.guess_correct}
        raise AssertionError(schema)


WORDS = (("음식", "바나나"), ("동물", "펭귄"))


class NoLies(LieSystem):
    def decide(self, answer, answered_before, remaining_questions):
        self.last_was_lie = False
        return False


class AlwaysLie(LieSystem):
    """규칙상 거짓말이 허용되면 항상 거짓말한다."""

    def lie_chance(self, remaining_questions):
        return 1.0


def make(llm=None, lies=NoLies, difficulty="보통", **kw):
    rng = random.Random(0)
    return Game(
        llm or FakeLLM(),
        words=WORDS,
        difficulty=difficulty,
        rng=rng,
        lie_system=lies(DIFFICULTIES[difficulty], rng),
        **kw,
    )


def ask_and_skip_ai(g, q="먹을 수 있어?", reply="예"):
    """플레이어 질문 → AI 질문 → 플레이어 답까지 한 라운드."""
    rec = g.player_ask(q)
    g.ai_act()
    g.player_answer(reply)
    return rec


def test_question_uses_token_and_passes_turn():
    g = make()
    g.player_ask("먹을 수 있어?")
    assert g.player.tokens == 14
    assert g.player.history == [("먹을 수 있어?", "예")]
    assert g.phase == Phase.AI_TURN


def test_invalid_question_keeps_turn_and_token():
    g = make(FakeLLM(answer="무효"))
    g.player_ask("무슨 색이야?")
    assert g.player.tokens == 15
    assert g.player.history == []
    assert g.phase == Phase.PLAYER_TURN


def test_ambiguous_answer_refunds_token():
    g = make(FakeLLM(answer="애매함"))
    g.player_ask("비싸?")
    assert g.player.tokens == 15
    assert g.phase == Phase.AI_TURN


def test_ai_asks_then_player_answers():
    g = make()
    g.player_ask("먹을 수 있어?")
    g.ai_act()
    assert g.phase == Phase.AWAIT_PLAYER_ANSWER
    assert g.ai.tokens == 14
    g.player_answer("예")
    assert len(g.ai.history) == 1 and g.ai.history[0][1] == "예"
    assert g.phase == Phase.PLAYER_TURN


def test_player_ambiguous_answer_refunds_ai_token():
    g = make()
    g.player_ask("먹을 수 있어?")
    g.ai_act()
    g.player_answer("애매함")
    assert g.ai.tokens == 15


def test_ai_prompt_has_no_word_bank_examples():
    """AI 턴 프롬프트의 예시 문구에 제시어 목록 단어가 있으면 힌트가 된다."""
    from game.words import all_words

    text = roles.AI_TURN_SYSTEM
    assert [w for _, w in all_words() if len(w) >= 2 and w in text] == []


def test_ai_never_sees_its_own_word():
    llm = FakeLLM()
    g = make(llm)
    g.player_ask("먹을 수 있어?")
    llm.calls.clear()
    g.ai_act()
    system, user = llm.calls[0]
    assert "펭귄" not in system  # 후보 목록은 FakeLLM 응답일 뿐, 프롬프트에 제시어가 있으면 안 됨
    assert "펭귄" not in user


def test_exact_guess_wins_without_llm():
    llm = FakeLLM(guess_correct=False)
    g = make(llm)
    assert g.player_guess("바나나야") is True
    assert g.phase == Phase.GAME_OVER and g.winner == "player"


def test_wrong_guess_costs_token_and_chance():
    g = make(FakeLLM(guess_correct=False))
    assert g.player_guess("사과") is False
    assert g.player.tokens == 14
    assert g.player.guesses_left == 2
    assert g.phase == Phase.AI_TURN


def test_synonym_guess_counts():
    g = Game(FakeLLM(), words=(("동물", "강아지"), ("동물", "펭귄")))
    assert g.player_guess("개") is True


# ---------------------------------------------------------------- 거짓말 / 간파


def test_no_lie_in_first_two_questions_and_not_consecutive():
    g = make(lies=AlwaysLie)
    r1 = ask_and_skip_ai(g)
    r2 = ask_and_skip_ai(g)
    r3 = ask_and_skip_ai(g)
    r4 = ask_and_skip_ai(g)
    r5 = ask_and_skip_ai(g)
    assert [r.lie for r in (r1, r2, r3, r4, r5)] == [False, False, True, False, True]
    assert g.lies.lies_left == 0  # 보통: 2회


def test_lie_flips_answer_and_line_and_shows_tell():
    g = make(lies=AlwaysLie)
    ask_and_skip_ai(g)
    ask_and_skip_ai(g)
    rec = g.player_ask("먹을 수 있어?")
    assert rec.lie and rec.truth == "예" and rec.final == "아니오"
    ev = [e for e in g.events if e.actor == "ai" and e.kind == "answer"][-1]
    assert ev.answer == "아니오" and ev.text == "아니!" and ev.tell == "tell"
    assert g.player.history[-1] == ("먹을 수 있어?", "아니오")


def test_ambiguous_answers_are_never_lies():
    g = make(FakeLLM(answer="애매함"), lies=AlwaysLie)
    for _ in range(4):
        rec = g.player_ask("비싸?")
        assert not rec.lie
        g.ai_act()
        g.player_answer("예")


def test_challenge_success_reveals_truth_and_rewards():
    g = make(lies=AlwaysLie)
    ask_and_skip_ai(g)
    ask_and_skip_ai(g)
    g.player_ask("먹을 수 있어?")  # 거짓말
    tokens = g.player.tokens
    g.ai_act()
    assert g.can_challenge  # AI 질문에 답하는 중에도 간파 가능
    res = g.player_challenge()
    assert res.success and res.truth == "예"
    assert g.player.tokens == tokens + 2
    assert g.player.history[-1] == ("먹을 수 있어?", "예")
    assert not g.can_challenge


def test_challenge_fail_costs_token():
    g = make()
    g.player_ask("먹을 수 있어?")
    g.ai_act()
    g.player_answer("예")
    tokens = g.player.tokens
    res = g.player_challenge()
    assert not res.success
    assert g.player.tokens == tokens - 1


def test_challenge_window_closes_after_next_question():
    g = make()
    ask_and_skip_ai(g)
    assert g.can_challenge
    g.player_ask("살아있어?")
    g.ai_act()
    g.player_answer("예")
    assert g.challengeable.question == "살아있어?"
    g.player_guess("사과")
    assert not g.can_challenge


def test_cannot_challenge_ambiguous_or_invalid():
    g = make(FakeLLM(answer="애매함"))
    g.player_ask("비싸?")
    assert not g.can_challenge
    with pytest.raises(GameError):
        g.player_challenge()


def test_invalid_question_keeps_previous_challenge_window():
    llm = FakeLLM()
    g = make(llm)
    ask_and_skip_ai(g)
    llm.answer = "무효"
    g.player_ask("무슨 색이야?")
    assert g.can_challenge


def test_reaction_line_never_receives_word():
    llm = FakeLLM()
    g = make(llm, lies=AlwaysLie)
    for _ in range(3):
        ask_and_skip_ai(g)
    reaction_calls = [(s, u) for s, u in llm.calls if "하기로 한 답" in s]
    assert reaction_calls
    assert all("바나나" not in s + u for s, u in reaction_calls)


def test_tell_only_on_lie():
    """페이크 텔은 없다: 진실이면 텔 없음, 거짓말이면 항상 텔."""
    for name in DIFFICULTIES:
        ls = LieSystem(DIFFICULTIES[name], random.Random(0))
        assert {ls.tell_for(False) for _ in range(100)} == {None}
        assert ls.tell_for(True) == "tell"


def test_tell_gets_subtler_with_difficulty():
    easy, normal, hard = (DIFFICULTIES[n] for n in ("쉬움", "보통", "어려움"))
    assert easy.tell_scale > normal.tell_scale > hard.tell_scale
    assert easy.tell_seconds > normal.tell_seconds > hard.tell_seconds


def test_lie_reveal_at_game_end():
    g = make(lies=AlwaysLie)
    ask_and_skip_ai(g)
    ask_and_skip_ai(g)
    g.player_ask("먹을 수 있어?")
    g.ai_act()
    g.player_answer("예")
    g.player_guess("바나나")
    assert "AI의 거짓말 공개" in g.events[-1].text


def test_ai_repeated_question_is_rejected():
    llm = FakeLLM()
    g = make(llm)
    g.player_ask("먹을 수 있어?")
    g.ai_act()
    asked = g.pending_ai_question
    g.player_answer("예")
    g.player_ask("살아있어?")
    llm.ai_action = {"action": "ask", "question": "", "guess": ""}
    llm.chat_json_orig = llm.chat_json
    # 계속 같은 질문을 돌려주면 재요청 후 범용 질문으로 대체한다 (정답 기회를 낭비하지 않음)
    llm.chat_json = lambda s, u, schema, temperature=0.3: (
        {"analysis": "", "candidates": ["물개"], "action": "ask", "question": asked, "guess": "", "line": ""}
        if "action" in schema["properties"]
        else llm.chat_json_orig(s, u, schema, temperature)
    )
    turn = g.ai_act()
    assert turn.action == "ask"
    assert turn.question == roles.FALLBACK_QUESTIONS[0]
    assert g.ai.guesses_left == 3


# ---------------------------------------------------------------- 3단계: 플레이어 거짓말 / 심판 / AI 의심


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def to_ai_question(g, q="먹을 수 있어?"):
    g.player_ask(q)
    g.ai_act()


def test_player_cannot_lie_in_first_two_answers():
    g = make()
    to_ai_question(g)
    assert not g.player_can_lie
    with pytest.raises(GameError):
        g.player_answer("예", lie=True)
    g.player_answer("예")
    to_ai_question(g)
    g.player_answer("예")
    to_ai_question(g)
    assert g.player_can_lie


def test_player_lie_uses_count_and_is_recorded():
    g = make()
    for _ in range(2):
        ask_and_skip_ai(g)
    to_ai_question(g)
    out = g.player_answer("아니오", lie=True)
    assert out.committed
    assert g.player_lies_left == 1
    assert g.player_answers[-1].lie
    assert g.events[-2].mood == "lie" or any(e.mood == "lie" for e in g.events)


def test_cannot_lie_with_ambiguous():
    g = make()
    for _ in range(2):
        ask_and_skip_ai(g)
    to_ai_question(g)
    with pytest.raises(GameError):
        g.player_answer("애매함", lie=True)


@pytest.mark.parametrize(
    "truth, answer, lie, warns",
    [
        ("예", "예", False, False),
        ("예", "아니오", False, True),  # 토글 없이 거짓말
        ("예", "아니오", True, False),  # 정상 거짓말
        ("예", "예", True, True),  # 토글 켰는데 사실
        ("애매함", "애매함", False, False),
        ("예", "애매함", False, True),  # 회피
        ("애매함", "예", False, False),
        ("무효", "아니오", False, False),  # 심판이 판단 못 하면 통과
    ],
)
def test_referee_table(truth, answer, lie, warns):
    g = make(FakeLLM(referee=truth))
    for _ in range(2):
        ask_and_skip_ai(g, reply="애매함" if truth == "애매함" else ("예" if truth != "아니오" else "아니오"))
    to_ai_question(g)
    assert (g.check_player_answer(answer, lie) is not None) == warns


def test_warning_does_not_change_state_unless_forced():
    g = make(FakeLLM(referee="예"))
    to_ai_question(g)
    out = g.player_answer("아니오")
    assert not out.committed and out.warning
    assert g.phase == Phase.AWAIT_PLAYER_ANSWER and g.ai.history == []
    out = g.player_answer("아니오", force=True)
    assert out.committed and g.ai.history[-1][1] == "아니오"


def test_lie_on_ambiguous_truth_not_counted():
    g = make(FakeLLM(referee="애매함"))
    for _ in range(2):
        ask_and_skip_ai(g, reply="애매함")
    to_ai_question(g)
    g.player_answer("예", lie=True)
    assert g.player_lies_left == 2
    assert not g.player_answers[-1].lie


def expecting(answer, conf=1.0):
    return {"action": "ask", "question": "살아있어?", "guess": "", "expected": answer, "confidence": conf}


def test_ai_catches_lie_when_answer_contradicts_expectation():
    llm = FakeLLM(referee="예", ai_action=expecting("예"))
    g = make(llm)
    for _ in range(2):
        ask_and_skip_ai(g)
    to_ai_question(g)
    ai_tokens = g.ai.tokens
    out = g.player_answer("아니오", lie=True)
    assert out.suspicion.accuse
    assert g.player_answers[-1].caught
    assert g.ai.history[-1][1] == "예"  # AI는 진실을 알게 됨
    assert g.ai_notes[-1].startswith("간파")
    assert g.ai.tokens == ai_tokens + 2
    assert g.stats["ai_challenge_ok"] == 1


def test_ai_false_accusation_costs_ai_token():
    llm = FakeLLM(referee="아니오", ai_action=expecting("예"))
    g = make(llm)
    for _ in range(2):
        ask_and_skip_ai(g, reply="아니오")
    to_ai_question(g)
    ai_tokens = g.ai.tokens
    out = g.player_answer("아니오")  # 진실인데 AI 예상과 다름
    assert out.suspicion.accuse
    assert g.ai.tokens == ai_tokens - 1
    assert g.stats["ai_challenge_fail"] == 1


def test_ai_does_not_suspect_first_two_answers():
    llm = FakeLLM(referee="무효", ai_action=expecting("예"))
    g = make(llm)
    to_ai_question(g)
    out = g.player_answer("아니오")
    assert out.suspicion.score == 0 and not out.suspicion.accuse


def test_slow_answer_raises_suspicion_and_marks_note():
    clock = FakeClock()
    llm = FakeLLM(ai_action=expecting("모름", 0.0))
    g = make(llm, clock=clock)
    for _ in range(3):
        to_ai_question(g)
        clock.t += 1.0
        g.player_answer("예")
    to_ai_question(g)
    clock.t += 10.0
    out = g.player_answer("예")
    assert any("답변 지연" in r for r in out.suspicion.reasons)
    assert out.suspicion.level in ("press", "trust")


def test_suspicious_note_is_sent_to_ai_prompt():
    llm = FakeLLM(ai_action=expecting("예", 0.5))  # 10 + 33 = 43 → 압박 단계
    g = make(llm)
    for _ in range(2):
        ask_and_skip_ai(g)
    to_ai_question(g)
    out = g.player_answer("아니오", force=True)
    assert out.suspicion.level == "press"
    assert g.ai_notes[-1] == "의심스러움"
    g.player_ask("과일이야?")
    llm.calls.clear()
    g.ai_act()
    assert "(의심스러움)" in llm.calls[0][1]


def test_ai_prompt_never_contains_referee_or_word():
    llm = FakeLLM(referee="예", ai_action=expecting("예"))
    g = make(llm)
    for _ in range(4):
        ask_and_skip_ai(g)
    ai_turn_calls = [(s, u) for s, u in llm.calls if "스무고개 게임의 플레이어" in s]
    assert ai_turn_calls
    assert all("펭귄" not in s + u for s, u in ai_turn_calls)


def test_note_markers_are_stripped_from_ai_question():
    llm = FakeLLM(ai_action={"action": "ask", "question": "물속에서만 살아? (간파: 진짜 답은 아니오)", "guess": ""})
    g = make(llm)
    g.player_ask("먹을 수 있어?")
    g.ai_act()
    assert "간파" not in g.pending_ai_question


def test_qwen_control_token_is_stripped():
    llm = FakeLLM(ai_action={"action": "ask", "question": "날개가 있어? /no_think", "guess": ""})
    g = make(llm)
    g.player_ask("먹을 수 있어?")
    g.ai_act()
    assert "/no" not in g.pending_ai_question


def test_ai_correct_guess_wins():
    llm = FakeLLM(ai_action={"action": "guess", "question": "", "guess": "펭귄"})
    g = make(llm)
    g.player_ask("먹을 수 있어?")
    g.ai_act()
    assert g.winner == "ai"


def test_ai_forced_to_guess_without_tokens():
    g = make(FakeLLM(), tokens=1)
    g.player_ask("먹을 수 있어?")  # 플레이어 토큰 0
    g.ai_act()
    g.player_answer("아니오")  # AI 토큰 0
    # 플레이어는 토큰이 없지만 정답 도전은 가능
    assert g.phase == Phase.PLAYER_TURN
    with pytest.raises(GameError):
        g.player_ask("살아있어?")
    g.player_guess("사과")
    turn = g.ai_act()
    assert turn.action == "guess"


WRONG_AI_GUESS = {"action": "guess", "question": "", "guess": "물개"}


def test_draw_when_both_exhausted():
    g = make(FakeLLM(ai_action=WRONG_AI_GUESS), tokens=0, guesses=1)
    g.player_guess("사과")
    g.ai_act()
    assert g.winner == "draw"


def test_skip_turn_of_side_that_cannot_act():
    g = make(FakeLLM(ai_action=WRONG_AI_GUESS), tokens=0, guesses=1)
    g.player_guess("사과")  # 플레이어 행동 불가
    g.ai.guesses_left = 2
    g.ai_act()  # AI 오답 → 플레이어는 못 움직이니 다시 AI 턴
    assert g.phase == Phase.AI_TURN


def test_wrong_turn_raises():
    g = make()
    with pytest.raises(GameError):
        g.ai_act()
    with pytest.raises(GameError):
        g.player_answer("예")


def test_wrong_guess_ai_says_wrong_before_its_turn():
    g = make(FakeLLM(guess_correct=False))
    g.player_guess("사과")
    last = g.events[-1]
    assert last.actor == "ai" and last.kind == "reaction" and last.mood == "guess_wrong"
    assert g.phase == Phase.AI_TURN  # 틀렸다고 말한 다음에 AI 차례


def test_ai_wrong_guess_has_no_wrong_reaction():
    g = make(FakeLLM(ai_action={"action": "guess", "question": "", "guess": "물개"}))
    g.player_ask("먹을 수 있어?")
    g.ai_act()
    assert not any(e.mood == "guess_wrong" for e in g.events)


def test_josa():
    from game.engine import was

    assert was("책") == "이었습니다"
    assert was("의자") == "였습니다"
    assert was("TV") == "였습니다"


def test_leak_filter():
    assert roles.leaks_word("바나나 맛있지", "바나나", [])
    assert roles.leaks_word("개야 그거", "강아지", ["개"])
    assert not roles.leaks_word("음, 그렇지.", "바나나", [])
