"""실제 Ollama로 LLM 역할 품질/속도를 확인하는 스크립트.

실행: .venv\\Scripts\\python.exe tools\\smoke_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game import roles  # noqa: E402
from game.llm import OllamaClient  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

llm = OllamaClient()


def timed(label, fn):
    t = time.perf_counter()
    out = fn()
    print(f"[{time.perf_counter() - t:5.1f}s] {label} → {out}")
    return out


# 워밍업 (모델 로딩)
timed("warmup", lambda: roles.answer_question(llm, "바나나", "음식", [], "먹을 수 있어?"))

print("\n== 답 판정 (제시어: 바나나) ==")
for q in [
    "먹을 수 있어?",
    "살아있는 동물이야?",
    "과일이야?",
    "비싸?",
    "무슨 색이야?",
    "바나나야?",
    "노란색이야?",
    "요리해서 먹어야 해?",
]:
    timed(q, lambda q=q: roles.answer_question(llm, "바나나", "음식", [], q))

print("\n== 반응 대사 (거짓말로 뒤집힌 답 포함) ==")
for q, a in [("먹을 수 있어?", "아니오"), ("살아있는 동물이야?", "예"), ("과일이야?", "아니오")]:
    timed(f"{q} / {a}", lambda q=q, a=a: roles.reaction_line(llm, q, a))

print("\n== 정답 판정 ==")
for word, syn, guess in [
    ("강아지", ["개", "멍멍이"], "댕댕이"),
    ("강아지", ["개", "멍멍이"], "늑대"),
    ("텔레비전", ["TV", "티비"], "텔레비젼"),
    ("바나나", [], "과일"),
]:
    timed(f"{word} vs {guess}", lambda w=word, s=syn, g=guess: roles.check_guess(llm, w, s, g))

print("\n== AI 턴 (AI 제시어: 펭귄, AI는 모름) ==")
history = [
    ("살아있는 거야?", "예"),
    ("하늘을 날 수 있어?", "아니오"),
    ("물에서 지낼 수 있어?", "예"),
]
for i in range(2):
    timed(f"턴 {i + 1}", lambda: roles.ai_turn(llm, "동물", history, [], 12, 3))
timed("토큰 0", lambda: roles.ai_turn(llm, "동물", history, ["물개"], 0, 2))
