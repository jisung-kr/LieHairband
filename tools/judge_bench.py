"""답 판정(= 심판) 정확도/속도 벤치마크.

실행: .venv\\Scripts\\python.exe tools\\judge_bench.py [--think] [--repeat N]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game import roles  # noqa: E402
from game.llm import OllamaClient  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# (카테고리, 제시어, 질문, 기대 답)
CASES = [
    # 상식/생물
    ("동물", "펭귄", "피가 있어?", "예"),
    ("동물", "펭귄", "알을 낳아?", "예"),
    ("동물", "펭귄", "날개가 있어?", "예"),
    ("동물", "펭귄", "하늘을 날 수 있어?", "아니오"),
    ("동물", "펭귄", "다리가 있어?", "예"),
    ("동물", "펭귄", "추운 곳에 살아?", "예"),
    ("동물", "돌고래", "물고기야?", "아니오"),
    ("동물", "돌고래", "새끼를 낳아?", "예"),
    ("동물", "뱀", "다리가 있어?", "아니오"),
    ("동물", "닭", "날개가 있어?", "예"),
    ("사물", "의자", "다리가 있어?", "예"),
    ("사물", "의자", "전기가 필요해?", "아니오"),
    ("사물", "냉장고", "전기가 필요해?", "예"),
    ("사물", "우산", "비 올 때 써?", "예"),
    ("음식", "바나나", "껍질을 벗겨 먹어?", "예"),
    ("음식", "김치", "매워?", "예"),
    ("음식", "수박", "씨가 있어?", "예"),
    ("직업", "의사", "병원에서 일해?", "예"),
    ("직업", "소방관", "제복을 입어?", "예"),
    ("직업", "어부", "바다에서 일해?", "예"),
    # 구어체 / 띄어쓰기 / 중의적 표현
    ("사물", "의자", "그럼 음식을 집을 때 쓰는거야??", "아니오"),
    ("사물", "의자", "식기류야??", "아니오"),
    ("사물", "의자", "집에있어?", "예"),
    ("사물", "의자", "앉을수있어??", "예"),
    ("사물", "가위", "종이 자를때 써?", "예"),
    ("음식", "김밥", "밥들어가?", "예"),
    ("동물", "토끼", "귀가 길어??", "예"),
    ("동물", "상어", "물에살아", "예"),
    ("직업", "의사", "아픈사람 고쳐줘?", "예"),
    ("사물", "열쇠", "문 열때 써?", "예"),
    ("사물", "시계", "배 위에서 써?", "아니오"),
    # 무효 / 애매함
    ("음식", "바나나", "무슨 색이야?", "무효"),
    ("음식", "바나나", "바나나야?", "무효"),
    ("음식", "바나나", "비싸?", "애매함"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    llm = OllamaClient(think=args.think)
    roles.answer_question(llm, "의자", "사물", [], "살아있어?")  # 워밍업

    times, ok, total = [], 0, 0
    for _ in range(args.repeat):
        for cat, word, q, exp in CASES:
            t = time.perf_counter()
            r = roles.answer_question(llm, word, cat, [], q)
            times.append(time.perf_counter() - t)
            total += 1
            ok += r.answer == exp
            if r.answer != exp:
                print(f"  BAD {word} / {q} → {r.answer} (기대 {exp})")
    print(
        f"think={args.think}: {ok}/{total} 정답 ({ok / total:.0%}) | "
        f"평균 {statistics.mean(times):.2f}s, 중앙값 {statistics.median(times):.2f}s, 최대 {max(times):.2f}s"
    )


if __name__ == "__main__":
    main()
