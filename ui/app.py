"""거짓말 헤드밴드 게임 화면 (Gradio).

디자인 기준: docs/ui-layout.html, 스타일: ui/style.css
  가로: 위 상태판 / 왼쪽 무대 + 커맨드 / 오른쪽 단서 영수증
  세로(820px 이하): 상태판 → 무대 → 커맨드 → 단서
페이지 전체를 화면 높이에 고정하고, 단서 영역만 안에서 스크롤한다.

실행: .venv\\Scripts\\python.exe ui\\app.py
"""

from __future__ import annotations

import functools
import html
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from game.engine import Event, Game, GameError, Phase, josa  # noqa: E402
from game.lie_system import DEFAULT_DIFFICULTY, DIFFICULTIES  # noqa: E402
from game.llm import LLMUnavailable, OllamaClient  # noqa: E402

ASSETS = ROOT / "assets"
gr.set_static_paths(paths=[ASSETS])

LLM = OllamaClient()

BG_W, BG_H = 1344, 768  # 배경 원본 크기 (무대 배치 기준)


def asset_url(rel: str) -> str:
    return f"/gradio_api/file={(ASSETS / rel).as_posix()}"

# ── 캐릭터: assets/characters/<ID>/meta.json 을 읽어서 동작한다 ──
#   name: 화면에 보이는 이름
#   stage.height/top: 무대 배치 (배경 px), stage.on_table: 테이블 위에 앉는 작은 캐릭터 (테이블 앞면 레이어 안 씀)
#   headband_card: 제시어 카드 위치 (스프라이트 px). printed=True 면 이미지에 카드가 그려져 있어서 글자만 얹는다
#   face_map: 엔진 표정 이름 → 파일 이름.  부위 조각: part_talk.png(말하기), part_tell.png(거짓말 버릇)
#   tell_fx: 버릇 조각 연출 ("spark" 지직거림 = 로봇 안테나, "twitch" 움찔 = 도하 귀). tell_origin: 연출 중심
#   lines: 고정 대사 (opening, game_over, challenge)
#   playable: false 면 무작위 선택에서 뺀다
# 새 게임마다 정식 캐릭터 중 하나를 무작위로 고른다. 하나도 없으면 임시 캐릭터(루미)로 시작한다.
# 개발용: 환경변수 HB_CHARACTER=<ID> 를 주면 항상 그 캐릭터로 시작한다.
CHARACTERS_DIR = ASSETS / "characters"
PLACEHOLDER_ID = "sly"  # 임시 캐릭터 (루미). 무작위 선택 대상이 아니다
BREATH_SECONDS = 3.6  # style.css 의 hb-breathe 주기와 같아야 한다
TALK_FLAP_MS = 260    # 말하기: 입 조각이 한 번 열렸다 닫히는 주기 (style.css hb-talk)

DEFAULT_LINES = {
    "opening": "준비 완료! 먼저 질문해 줘.",
    "game_over": {"player": "으으, 이번엔 내가 졌어.", "ai": "내가 이겼다!", "draw": "무승부네. 아쉽다."},
    "challenge": {
        "player_ok": "으윽, 들켰다...",
        "player_fail": "흥, 난 진실만 말했는데?",
        "ai_ok": "역시 거짓말이었어!",
        "ai_fail": "어라, 진짜였어? 미안...",
    },
}


@dataclass(frozen=True)
class Character:
    id: str
    name: str
    sprite_w: int
    sprite_h: int
    height: float          # 무대에서의 표시 높이 (배경 px)
    top: float             # 무대에서의 위쪽 위치 (배경 px)
    on_table: bool
    shadow: dict | None
    card: dict
    tell_origin: dict
    tell_fx: str
    face_map: dict
    lines: dict

    def has(self, name: str) -> bool:
        return _sprite_exists(self.id, name)

    def sprite_name(self, face: str) -> str:
        for name in (self.face_map.get(face, face), face):
            if self.has(name):
                return name
        return "idle"

    def url(self, name: str) -> str:
        return asset_url(f"characters/{self.id}/{name}.png")

    @property
    def sprite_box(self) -> str:
        """스프라이트 상자 배치 (배경 기준 %)."""
        return (
            f"top:{self.top / BG_H * 100:.3f}%;height:{self.height / BG_H * 100:.3f}%;"
            f"aspect-ratio:{self.sprite_w}/{self.sprite_h}"
        )


@lru_cache(maxsize=256)
def _sprite_exists(character_id: str, name: str) -> bool:
    return (CHARACTERS_DIR / character_id / f"{name}.png").exists()


@lru_cache(maxsize=16)
def load_character(character_id: str) -> Character:
    meta = json.loads((CHARACTERS_DIR / character_id / "meta.json").read_text(encoding="utf-8"))
    w, h = meta.get("size", [832, 1216])
    stage = meta.get("stage", {"height": 900, "top": -25, "on_table": False})
    lines = meta.get("lines", {})
    return Character(
        id=character_id,
        name=meta.get("name") or {"sly": "루미"}.get(character_id, character_id),
        sprite_w=w,
        sprite_h=h,
        height=stage["height"],
        top=stage["top"],
        on_table=stage.get("on_table", False),
        shadow=stage.get("shadow"),
        card=meta.get("headband_card", {"cx": 416, "cy": 174, "w": 230}),
        tell_origin=meta.get("tell_origin") or meta.get("antenna_tip") or {"cx": w / 2, "cy": h / 4},
        tell_fx=meta.get("tell_fx", "spark"),
        face_map={"ask": "idle", "gotcha": "smug", "oops": "caught", **meta.get("face_map", {})},
        lines={
            "opening": lines.get("opening", DEFAULT_LINES["opening"]),
            "game_over": {**DEFAULT_LINES["game_over"], **lines.get("game_over", {})},
            "challenge": {**DEFAULT_LINES["challenge"], **lines.get("challenge", {})},
        },
    )


def playable_characters() -> list[str]:
    """무작위 선택 대상: meta.json 과 idle.png 가 있고, 임시 캐릭터가 아니고, playable 이 false 가 아닌 캐릭터."""
    found = []
    for d in sorted(CHARACTERS_DIR.iterdir()) if CHARACTERS_DIR.exists() else []:
        if d.name == PLACEHOLDER_ID or not (d / "meta.json").exists() or not (d / "idle.png").exists():
            continue
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # 깨진 meta 는 건너뛴다
        if meta.get("playable", True):
            found.append(d.name)
    return found


def pick_character(rng: random.Random | None = None) -> str:
    forced = os.environ.get("HB_CHARACTER")
    if forced and (CHARACTERS_DIR / forced / "meta.json").exists():
        return forced
    candidates = playable_characters()
    return (rng or random).choice(candidates) if candidates else PLACEHOLDER_ID


def char_of(game: Game) -> Character:
    """게임에 정해진 상대 캐릭터. 새 게임을 만들 때 game.character_id 를 붙인다."""
    return load_character(getattr(game, "character_id", PLACEHOLDER_ID))

ANSWER_MARK = {"예": "O", "아니오": "X", "애매함": "△", "무효": "-"}
# 영수증(고정폭 글꼴)에서는 O 가 숫자 0 처럼 보여서 기호를 쓴다
RECEIPT_MARK = {"예": "○", "아니오": "×", "애매함": "△", "무효": "-"}

TYPE_MS = 45              # 대사 한 글자 간격
TYPE_MAX_TOTAL_MS = 2600  # 긴 대사도 이 시간 안에 다 찍히게


def esc(text) -> str:
    return html.escape(str(text))


ICON = {
    "q": '<path d="M4 5h16v11H9l-5 4z"/><path d="M10 9a2 2 0 1 1 2.6 1.9c-.4.1-.6.5-.6.9v.4"/><path d="M12 14.2h.01"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><path d="M12 12h.01"/>',
    "mask": '<path d="M4 6c3 1.5 13 1.5 16 0v5c0 5-4 8-8 8s-8-3-8-8z"/>'
    '<path d="M8 11.5c.8-.7 1.7-.7 2.5 0M13.5 11.5c.8-.7 1.7-.7 2.5 0"/><path d="M9.5 15.5c1.5 1 3.5 1 5 0"/>',
}



def icon(name: str) -> str:
    return f'<svg class="i" viewBox="0 0 24 24" aria-hidden="true">{ICON[name]}</svg>'


# ══════════════════════════════════════════════════════════════════ 렌더링


def render_hud(game: Game, busy: str) -> str:
    p, a = game.player, game.ai
    name = char_of(game).name
    if game.phase == Phase.GAME_OVER:
        turn, cls = {"player": "승리", "ai": "패배", "draw": "무승부"}[game.winner], "over"
    elif busy:
        turn, cls = "생각 중", "busy"
    elif game.phase == Phase.AWAIT_PLAYER_ANSWER:
        turn, cls = f"{name}의 질문", "ai"
    else:
        turn, cls = "내 차례", ""

    def ticks(n: int) -> str:
        total = max(15, n)
        cells = "".join(f'<span class="tick{"" if i < n else " off"}"></span>' for i in range(total))
        return f'<span class="ticks">{cells}</span>'

    return f"""
<div class="hud">
  <div class="side me">
    <div class="who"><span class="name">나</span><span class="word secret">??? · {esc(p.category)}</span></div>
    <div class="stats">
      <span class="stat" title="질문 토큰">{icon('q')}<b>{p.tokens}</b>{ticks(p.tokens)}</span>
      <span class="stat" title="정답 도전">{icon('target')}<b>{p.guesses_left}</b></span>
      <span class="stat" title="남은 거짓말">{icon('mask')}<b>{game.player_lies_left}</b></span>
    </div>
  </div>
  <div class="turn {cls}">{turn}</div>
  <div class="side ai">
    <div class="who"><span class="word" title="나만 볼 수 있는 {name}의 제시어">{esc(a.word)} · {esc(a.category)}</span><span class="name">{name}</span></div>
    <div class="stats">
      <span class="stat" title="남은 거짓말">{icon('mask')}<b>{game.lies.lies_left}</b></span>
      <span class="stat" title="정답 도전">{icon('target')}<b>{a.guesses_left}</b></span>
      <span class="stat" title="질문 토큰">{ticks(a.tokens)}{icon('q')}<b>{a.tokens}</b></span>
    </div>
  </div>
</div>"""


def type_step_ms(text: str) -> float:
    return min(TYPE_MS, TYPE_MAX_TOTAL_MS / max(1, len(text)))


def typewriter(text: str) -> str:
    """대사를 한 글자씩 찍히는 HTML로. gr.HTML 은 <script> 를 실행하지 않아서 CSS 애니메이션으로 처리한다.
    글자는 처음부터 자리를 차지하고 투명하다가 나타나므로 상자 크기가 흔들리지 않는다."""
    step = type_step_ms(text)
    chars = "".join(
        f'<span class="hb-ch" style="animation-delay:{i * step:.0f}ms">{esc(ch)}</span>' for i, ch in enumerate(text)
    )
    return chars + f'<span class="hb-next" style="animation-delay:{len(text) * step + 150:.0f}ms">▼</span>'


# 간파 결과 → 표정 (대사는 캐릭터 meta 의 lines.challenge)
CHALLENGE_FACE = {"player_ok": "caught", "player_fail": "smug", "ai_ok": "gotcha", "ai_fail": "oops"}
REACTION_FACE = {"trust": "idle", "press": "suspicious", "accuse": "accuse", "guess_wrong": "smug"}


def _sprite_layer(ch: Character, inner: str, cls: str = "", style: str = "") -> str:
    style_attr = f' style="{style}"' if style else ""
    return (
        f'<div class="hb-char hb-layer {cls}"{style_attr}>'
        f'<div class="hb-sprite" style="{ch.sprite_box}">{inner}</div></div>'
    )


def render_stage(game: Game) -> str:
    """무대. 마지막 AI 관련 이벤트만으로 만든다.
    내용이 바뀔 때만 DOM 이 새로 그려지고, 그때 텔/타자 애니메이션이 한 번 재생된다."""
    ch = char_of(game)
    SPRITE_W, SPRITE_H, CARD = ch.sprite_w, ch.sprite_h, ch.card
    face, tell, line, stamp, key = "idle", None, "", None, 0
    for i, e in enumerate(game.events):
        if e.actor == "ai" and e.kind == "answer":
            face, tell, line, key = "idle", e.tell, e.text, i
            stamp = e.answer if e.answer in ("예", "아니오", "애매함") else None
        elif e.actor == "ai" and e.kind in ("question", "guess"):
            face, tell, line, stamp, key = "ask", None, e.text, None, i
        elif e.actor == "ai" and e.kind == "reaction":
            face, tell, line, stamp, key = REACTION_FACE.get(e.mood, "idle"), None, e.text, None, i
            if e.mood == "guess_wrong":
                stamp = "아니오"  # 정답 도전 오답: X 도장
        elif e.kind == "challenge_result" and e.mood in CHALLENGE_FACE:
            face, line = CHALLENGE_FACE[e.mood], ch.lines["challenge"][e.mood]
            tell, stamp, key = None, None, i
    if game.phase == Phase.GAME_OVER:
        face = {"player": "lose", "ai": "win"}.get(game.winner, "idle")
        tell, stamp, line = None, None, ch.lines["game_over"][game.winner]
    elif not line:
        line = ch.lines["opening"]

    # 캐릭터는 늘 미세하게 숨 쉬고(hb-breathe), 대사가 찍히는 동안 입이 뻐끔거리고(part_talk),
    # 거짓말일 때만 고유 버릇 조각(part_tell)이 짧게 깜빡인다. 페이크 텔은 없다 (기획서 4.2).
    # 제시어 카드와 조각들은 몸과 같은 레이어 안에 넣어서 숨쉬기 때 같이 움직인다.
    printed = CARD.get("printed", False)
    card_style = f"left:{CARD['cx'] / SPRITE_W * 100:.2f}%;top:{CARD['cy'] / SPRITE_H * 100:.2f}%;" + (
        f"width:{CARD['w'] / SPRITE_W * 100:.2f}%" if printed
        else f"min-width:{CARD['w'] / SPRITE_W * 100 * 0.6:.2f}%;max-width:{CARD['w'] / SPRITE_W * 100 * 1.1:.2f}%"
    )
    if printed:
        # 그려진 카드 폭 안에 들어가도록 글자 수에 맞춰 크기를 줄인다 (cqw = 무대 폭의 1%)
        card_cqw = CARD["w"] * ch.height / SPRITE_H / BG_W * 100
        card_style += f";font-size:{min(card_cqw * 0.3, card_cqw * 0.9 / max(1, len(game.ai.word))):.2f}cqw"
    card = (
        f'<div class="hb-card{" printed" if printed else ""}" style="{card_style}" '
        f'title="{ch.name}의 제시어 (나만 볼 수 있음)">{esc(game.ai.word)}</div>'
    )
    name = ch.sprite_name(face)
    body = f'<img class="hb-body" src="{ch.url(name)}" alt="{ch.name}">'
    # 말하기: 대사가 다 찍힐 때까지만 입 조각을 깜빡인다. 표정 이미지가 떠 있을 때는 입이 안 맞아서 쓰지 않는다 (9.2)
    if line and name == "idle" and ch.has("part_talk"):
        flaps = max(1, round(len(line) * type_step_ms(line) / TALK_FLAP_MS))
        body += f'<img class="hb-talk" src="{ch.url("part_talk")}" alt="" style="--flaps:{flaps}">'
    if tell and ch.has("part_tell"):
        d, o = game.difficulty, ch.tell_origin
        body += (
            f'<img class="hb-tellfx {esc(ch.tell_fx)}" src="{ch.url("part_tell")}" alt="" '
            f'style="--tell-dur:{d.tell_seconds}s;--tell-amp:{d.tell_scale};'
            f'--ox:{o["cx"] / SPRITE_W * 100:.2f}%;--oy:{o["cy"] / SPRITE_H * 100:.2f}%">'
        )
    # 숨쉬기 위상을 서버 시각에 맞춘다 → 무대가 다시 그려져도 숨쉬기가 처음부터 다시 시작해 '툭' 튀지 않는다
    breath = f"animation-delay:-{time.time() % BREATH_SECONDS:.2f}s"
    layers = [_sprite_layer(ch, body + card, "hb-breathe", breath)]
    card = ""
    # 테이블 위에 앉는 캐릭터: 바닥 그림자를 깔고, 테이블 앞면 레이어는 쓰지 않는다
    shadow, table_fg = "", f'<img class="hb-layer hb-full" src="{asset_url("backgrounds/cafe_table_fg.png")}" alt="">'
    if ch.on_table:
        s = ch.shadow or {"cx": BG_W / 2, "cy": 600, "w": 300, "h": 30}
        shadow = (
            f'<div class="hb-layer hb-shadow" style="left:{(s["cx"] - s["w"] / 2) / BG_W * 100:.2f}%;'
            f'top:{(s["cy"] - s["h"] / 2) / BG_H * 100:.2f}%;width:{s["w"] / BG_W * 100:.2f}%;'
            f'height:{s["h"] / BG_H * 100:.2f}%"></div>'
        )
        table_fg = ""

    stamp_html = ""
    if stamp:
        cls = {"예": "yes", "아니오": "no", "애매함": "amb"}[stamp]
        stamp_html = f'<div class="hb-stamp {cls}"><span class="mk">{ANSWER_MARK[stamp]}</span><span class="lb">{stamp}</span></div>'
    dialog = (
        f'<div class="hb-dialog"><span class="hb-nameplate">{ch.name}</span>'
        f'<div class="hb-line">{typewriter(line) if line else "..."}</div></div>'
    )
    return f"""
<div class="hb-stage" data-k="{key}">
  <img class="hb-layer hb-full" src="{asset_url('backgrounds/cafe.png')}" alt="">
  {shadow}{''.join(layers)}{card}
  {table_fg}
  {stamp_html}{dialog}
</div>"""


def _receipt(sub: str, items: list[str], foot: tuple[str, str]) -> str:
    body = "".join(items) or '<div class="empty">아직 질문이 없어요.</div>'
    return f"""
<div class="receipt">
  <div class="head"><div class="t">CLUE RECEIPT</div><div class="s">{sub}</div></div>
  {body}
  <div class="foot"><span>{foot[0]}</span><span>{foot[1]}</span></div>
</div>"""


def _item(n: int, q: str, answer: str, cls: str = "", note: str = "", extra: str = "") -> str:
    a_cls = {"예": "o", "아니오": "x"}.get(answer, "")
    note_html = f'<span class="note">{note}</span>' if note else ""
    return (
        f'<div class="item {cls}"><span class="n">{n:02d}</span><span class="q"><span>{esc(q)}</span></span>'
        f'<span class="a {a_cls}">{RECEIPT_MARK.get(answer, esc(answer))}{extra}</span>{note_html}</div>'
    )


def render_my_clues(game: Game) -> str:
    by_index = {r.history_index: r for r in game.ai_answers}
    items = []
    for i, (q, a) in enumerate(game.player.history):
        rec = by_index.get(i)
        cls, note = "", ""
        if rec and rec.caught:
            cls, note = "caught", f"간파 성공 · 처음엔 {RECEIPT_MARK[rec.final]}라고 함"
        elif rec is not None and rec is game.challengeable and game.can_challenge:
            cls, note = "open", "간파 가능"
        items.append(_item(i + 1, q, a, cls, note))
    caught = sum(1 for r in game.ai_answers if r.caught)
    return _receipt(
        f"내 제시어 · {esc(game.player.category)} · 질문 {len(game.player.history)}회",
        items,
        (f"남은 질문 {game.player.tokens}", f"간파 성공 {caught}"),
    )


def render_ai_clues(game: Game) -> str:
    name = char_of(game).name
    items = []
    for i, ((q, a), rec) in enumerate(zip(game.ai.history, game.player_answers)):
        cls, note, extra = "", "", ""
        if rec.lie:
            extra = '<span class="mine" title="내가 한 거짓말">거짓</span>'
            if rec.caught:
                cls, note = "caught", f"들킴 · {name}{josa(name, '이', '가')} 진짜 답을 앎"
        items.append(_item(i + 1, q, a, cls, note, extra))
    fooled = sum(1 for r in game.player_answers if r.lie and not r.caught)
    return _receipt(
        f"{name}의 제시어 · {esc(game.ai.category)} · 질문 {len(game.ai.history)}회",
        items,
        (f"{name} 남은 질문 {game.ai.tokens}", f"속이기 성공 {fooled}"),
    )


def _chat_line(e: Event, ai_name: str) -> str:
    if e.actor == "system":
        return '<div class="chatline sys">' + "<br>".join(esc(t) for t in e.text.split("\n")) + "</div>"
    who, cls = ("나", "me") if e.actor == "player" else (ai_name, "")
    if e.kind == "answer" and e.actor == "ai":
        body = f"{ANSWER_MARK.get(e.answer, '')} {esc(e.text)}"
    elif e.kind == "answer":
        body = f"{ANSWER_MARK.get(e.answer, '')} {esc(e.answer)}"
        if e.mood == "lie":
            body += ' <span class="mine">거짓</span>'
    elif e.kind == "guess" and e.actor == "player":
        body = f"정답 도전: {esc(e.text)}"
    else:
        body = esc(e.text)
    return f'<div class="chatline {cls}"><b>{who}</b>{body}</div>'


def render_chat(game: Game, pending_question: str = "") -> str:
    name = char_of(game).name
    lines = [_chat_line(e, name) for e in game.events]
    if pending_question:
        lines.append(f'<div class="chatline me"><b>나</b>{esc(pending_question)}</div>')
    return '<div class="chat">' + "".join(reversed(lines)) + "</div>"  # 최신이 위로


def render_result(game: Game) -> str:
    title = {"player": "승리!", "ai": "패배...", "draw": "무승부"}[game.winner]
    s = game.stats
    my_lies = [a for a in game.player_answers if a.lie]
    fooled = sum(1 for a in my_lies if not a.caught)
    return f"""
<div class="result">
  <div class="title">{title}</div>
  <div class="rows">
    <span>내 제시어 <b>{esc(game.player.word)}</b></span>
    <span>간파 성공 <b>{s['challenge_ok']}</b> · 실패 <b>{s['challenge_fail']}</b></span>
    <span>내 거짓말 <b>{len(my_lies)}</b>번 중 <b>{fooled}</b>번 속임</span>
  </div>
</div>"""


# ══════════════════════════════════════════════════════════════════ 레이아웃 + 핸들러

# 규칙 설명과 탭 이름은 화면을 만들 때 한 번만 정해져서, 게임마다 바뀌는 캐릭터 이름 대신 "상대"라고 쓴다
RULES = """
- 상대는 새 게임마다 무작위로 정해집니다.
- 예/아니오로 답할 수 있는 질문으로 내 제시어를 맞히세요. 질문 1번에 토큰 1개를 씁니다.
- 애매함 답이나 무효 질문은 토큰을 쓰지 않습니다. 정답 도전은 3번, 틀리면 토큰 1개를 잃습니다.
- **상대는 가끔 거짓말을 합니다.** 거짓말할 때는 상대마다 다른 버릇이 잠깐 나옵니다.
- 직전 답이 거짓말 같으면 **간파**. 성공하면 진짜 답 공개와 토큰 +2, 실패하면 토큰 -1. 다음 질문 전까지만 할 수 있습니다.
- 나도 거짓말할 수 있습니다. 상대의 질문에 답할 때 "거짓말로 답하기"를 켜고 사실과 반대로 답하세요. 첫 2개 답은 거짓말할 수 없습니다.
- 상대는 내 답을 항상 의심합니다. 너무 오래 망설이면 의심받습니다.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="거짓말 헤드밴드") as demo:
        state = gr.State(None)
        warn_state = gr.State(None)  # (경고 문구, 답, 거짓말 여부)

        def H(**kw):
            return gr.HTML(apply_default_css=False, js_on_load=None, **kw)

        with gr.Column(elem_id="hb-app"):
            hud = H(elem_id="hb-hud")
            with gr.Row(elem_id="hb-body", equal_height=False):
                with gr.Column(elem_id="hb-left", scale=3, min_width=0):
                    stage = H(elem_id="hb-stage")
                    with gr.Column(elem_id="hb-cmd"):
                        # 내 차례
                        with gr.Column(visible=False, elem_classes="mode") as grp_my:
                            with gr.Row(elem_classes="perceive", visible=False) as perceive_row:
                                perceive_html = H(elem_classes="perceive-text")
                                perceive_btn = gr.Button("간파", elem_classes=["hb-btn", "primary"], scale=0, min_width=80)
                            with gr.Row(elem_classes="ask"):
                                q_box = gr.Textbox(
                                    placeholder="예/아니오로 답할 수 있는 질문 (예: 먹을 수 있어?)",
                                    show_label=False, container=False, lines=1, max_lines=1,
                                    scale=5, min_width=0, elem_id="hb-q",
                                )
                                ask_btn = gr.Button("질문", elem_classes=["hb-btn", "primary"], scale=0, min_width=64)
                                guess_btn = gr.Button("정답", elem_classes=["hb-btn"], scale=0, min_width=64)
                        # 루미의 질문에 답하기
                        with gr.Column(visible=False, elem_classes="mode", elem_id="hb-answer") as grp_answer:
                            ai_q_html = H()
                            with gr.Row():
                                lie_box = gr.Checkbox(label="거짓말로 답하기", elem_id="hb-lie", container=False, scale=3)
                                answer_perceive_btn = gr.Button(
                                    "직전 답 간파", elem_classes=["hb-btn", "quiet"], scale=0, min_width=100
                                )
                            with gr.Row():
                                yes_btn = gr.Button("O\n예", elem_classes=["hb-btn", "ox-btn", "yes"], min_width=0)
                                no_btn = gr.Button("X\n아니오", elem_classes=["hb-btn", "ox-btn", "no"], min_width=0)
                                amb_btn = gr.Button("△\n애매함", elem_classes=["hb-btn", "ox-btn", "amb"], min_width=0)
                        # 심판 경고
                        with gr.Column(visible=False, elem_classes="mode") as grp_warn:
                            warn_html = H()
                            with gr.Row():
                                retry_btn = gr.Button("다시 고르기", elem_classes=["hb-btn"], min_width=0)
                                force_btn = gr.Button("그대로 진행", elem_classes=["hb-btn", "quiet"], min_width=0)
                        # 생각 중
                        with gr.Column(visible=False, elem_classes="mode") as grp_busy:
                            busy_html = H()
                        # 게임 끝
                        with gr.Column(visible=False, elem_classes="mode") as grp_over:
                            over_html = H()
                            again_btn = gr.Button("한 판 더", elem_classes=["hb-btn", "primary"])
                with gr.Column(elem_id="hb-right", scale=1, min_width=0):
                    with gr.Tabs(elem_id="hb-tabs"):
                        with gr.Tab("내 단서"):
                            clues_html = H()
                        with gr.Tab("상대 단서"):
                            ai_clues_html = H()
                        with gr.Tab("대화"):
                            chat_html = H()
                    with gr.Accordion("설정 · 규칙", open=False, elem_id="hb-settings"):
                        difficulty = gr.Radio(list(DIFFICULTIES), value=DEFAULT_DIFFICULTY, label="난이도 (새 게임부터 적용)")
                        new_btn = gr.Button("새 게임", elem_classes=["hb-btn"])
                        gr.Markdown(RULES)

        outputs = [
            state, warn_state, hud, stage,
            grp_my, perceive_row, perceive_html, q_box, ask_btn, guess_btn,
            grp_answer, ai_q_html, lie_box, answer_perceive_btn,
            grp_warn, warn_html, grp_busy, busy_html, grp_over, over_html,
            clues_html, ai_clues_html, chat_html,
        ]

        def render(game: Game, busy: str = "", warning=None, pending_question: str = "") -> dict:
            phase, idle = game.phase, not busy
            mode = (
                "busy" if busy
                else "over" if phase == Phase.GAME_OVER
                else "warn" if warning
                else "answer" if phase == Phase.AWAIT_PLAYER_ANSWER
                else "my"
            )
            ch = game.challengeable
            perceive = idle and game.can_challenge and ch is not None
            if game.player_lies_left <= 0:
                lie_label = "거짓말로 답하기 (남은 거짓말 없음)"
            elif not game.player_can_lie:
                lie_label = "거짓말로 답하기 (첫 2개 답은 불가)"
            else:
                lie_label = f"거짓말로 답하기 (남은 {game.player_lies_left})"
            asked = (
                f'<div class="asked"><span class="who-tag">{char_of(game).name}</span>'
                f'<span class="text">{esc(game.pending_ai_question or "")}</span></div>'
            )
            return {
                state: game,
                warn_state: warning,
                hud: render_hud(game, busy),
                stage: render_stage(game),
                grp_my: gr.Column(visible=mode == "my"),
                perceive_row: gr.Row(visible=perceive),
                perceive_html: (
                    f'<span class="perceive-text">직전 답 <b>{esc(ch.question)} → {ANSWER_MARK[ch.final]}</b>'
                    f'<span class="hint"> · 다음 질문 전까지</span></span>' if perceive else ""
                ),
                q_box: gr.Textbox(value="", interactive=idle),
                ask_btn: gr.Button(interactive=idle and game.player.tokens > 0),
                guess_btn: gr.Button(interactive=idle and game.player.guesses_left > 0),
                grp_answer: gr.Column(visible=mode == "answer"),
                ai_q_html: asked,
                lie_box: gr.Checkbox(value=False, label=lie_label, interactive=game.player_can_lie),
                answer_perceive_btn: gr.Button(visible=perceive),
                grp_warn: gr.Column(visible=mode == "warn"),
                warn_html: asked + f'<div class="ref"><b>심판</b>{esc(warning[0])}</div>' if warning else "",
                grp_busy: gr.Column(visible=mode == "busy"),
                busy_html: f'<div class="busyline">{esc(busy)} 잠시 입력이 잠깁니다.</div>' if busy else "",
                grp_over: gr.Column(visible=mode == "over"),
                over_html: render_result(game) if phase == Phase.GAME_OVER else "",
                clues_html: render_my_clues(game),
                ai_clues_html: render_ai_clues(game),
                chat_html: render_chat(game, pending_question),
            }

        def subject(game: Game) -> str:
            """'도하가' / '로봇이' 처럼 이름 + 주격 조사."""
            name = char_of(game).name
            return name + josa(name, "이", "가")

        def run_ai(game: Game):
            while game.phase == Phase.AI_TURN:
                yield render(game, busy=f"{subject(game)} 질문을 고르는 중입니다.")
                game.ai_act()
                yield render(game)

        def guarded(fn):
            """GameError 와 Ollama 연결 실패는 경고 토스트로 보여준다."""

            @functools.wraps(fn)  # API 이름이 원래 함수 이름으로 남게
            def wrapper(game, *args):
                try:
                    if game is None:
                        raise GameError("먼저 새 게임을 시작하세요.")
                    yield from fn(game, *args)
                except (GameError, LLMUnavailable) as e:
                    gr.Warning(str(e))
                    if game is not None:
                        yield render(game)

            return wrapper

        def new_game(_game, diff):
            game = Game(LLM, difficulty=diff or DEFAULT_DIFFICULTY)
            # 상대 캐릭터를 무작위로 정한다 (정식 캐릭터가 없으면 임시 캐릭터 루미)
            game.character_id = pick_character()
            yield render(game)

        @guarded
        def on_ask(game: Game, question: str):
            if not question.strip():
                raise GameError("질문을 입력하세요.")
            yield render(game, busy=f"{subject(game)} 답을 생각하는 중입니다.", pending_question=question.strip())
            game.player_ask(question)
            yield render(game)
            yield from run_ai(game)

        @guarded
        def on_guess(game: Game, guess: str):
            if not guess.strip():
                raise GameError("입력창에 정답이라고 생각하는 단어를 쓰고 '정답'을 누르세요.")
            yield render(game, busy="정답을 확인하는 중입니다.")
            correct = game.player_guess(guess)
            yield render(game)
            if not correct and game.phase == Phase.AI_TURN:
                # 루미가 "틀렸어"라고 말하는 대사가 다 찍힐 때까지 기다린 뒤에 질문을 고른다
                last = game.events[-1].text if game.events[-1].kind == "reaction" else ""
                time.sleep(min(2.5, len(last) * TYPE_MS / 1000 + 0.8))
            yield from run_ai(game)

        @guarded
        def on_perceive(game: Game):
            game.player_challenge()
            yield render(game)
            yield from run_ai(game)

        def answer(game: Game, ans: str, lie: bool, force: bool):
            outcome = game.player_answer(ans, lie=lie, force=force)
            if not outcome.committed:
                yield render(game, warning=(outcome.warning, ans, lie))
                return
            yield render(game)
            yield from run_ai(game)

        def answer_handler(ans: str):
            @guarded
            def handler(game: Game, lie: bool):
                yield from answer(game, ans, bool(lie), force=False)

            return handler

        @guarded
        def on_force(game: Game, warning):
            if not warning:
                raise GameError("진행할 답이 없습니다.")
            _, ans, lie = warning
            yield from answer(game, ans, lie, force=True)

        @guarded
        def on_retry(game: Game):
            yield render(game)

        # Gradio 기본 진행 표시(주황 테두리, 로딩 표시)는 끈다. "생각 중" 은 상태판과 커맨드에서 직접 보여준다.
        ev = {"outputs": outputs, "show_progress": "hidden"}
        new_btn.click(new_game, [state, difficulty], **ev)
        again_btn.click(new_game, [state, difficulty], **ev)
        ask_btn.click(on_ask, [state, q_box], **ev)
        q_box.submit(on_ask, [state, q_box], **ev)
        guess_btn.click(on_guess, [state, q_box], **ev)
        perceive_btn.click(on_perceive, [state], **ev)
        answer_perceive_btn.click(on_perceive, [state], **ev)
        yes_btn.click(answer_handler("예"), [state, lie_box], **ev)
        no_btn.click(answer_handler("아니오"), [state, lie_box], **ev)
        amb_btn.click(answer_handler("애매함"), [state, lie_box], **ev)
        force_btn.click(on_force, [state, warn_state], **ev)
        retry_btn.click(on_retry, [state], **ev)
        demo.load(new_game, [state, difficulty], **ev)
    return demo


# 글꼴과 스타일은 <head> 로 넣는다 (launch(css=) 로 넣으면 @media 가 적용되지 않음)
HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Do+Hyeon'
    '&family=IBM+Plex+Sans+KR:wght@400;500;700&family=Nanum+Gothic+Coding:wght@400;700&display=swap">'
    f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>"
)


if __name__ == "__main__":
    build().launch(head=HEAD, allowed_paths=[str(ASSETS)], footer_links=[])
