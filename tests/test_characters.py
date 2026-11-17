"""상대 캐릭터 선택과 캐릭터별 화면 렌더링 테스트 (Ollama 없이)."""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

import app  # noqa: E402
from game.engine import Game  # noqa: E402
from test_engine import AlwaysLie, FakeLLM  # noqa: E402
from game.lie_system import DIFFICULTIES  # noqa: E402


def make_char_dir(base: Path, cid: str, meta: dict | None = None, idle: bool = True) -> None:
    d = base / cid
    d.mkdir(parents=True)
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if idle:
        (d / "idle.png").write_bytes(b"")


@pytest.fixture
def chars_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CHARACTERS_DIR", tmp_path)
    monkeypatch.delenv("HB_CHARACTER", raising=False)
    app.load_character.cache_clear()
    yield tmp_path
    app.load_character.cache_clear()


def test_picks_only_real_characters(chars_dir):
    make_char_dir(chars_dir, "sly", {"name": "루미"})
    make_char_dir(chars_dir, "honest", {"name": "도하"})
    make_char_dir(chars_dir, "robot", {"name": "로봇"})
    assert app.playable_characters() == ["honest", "robot"]
    picks = {app.pick_character(random.Random(i)) for i in range(40)}
    assert picks == {"honest", "robot"}  # 둘 다 나오고, 임시 캐릭터는 안 나온다


def test_falls_back_to_placeholder(chars_dir):
    make_char_dir(chars_dir, "sly", {"name": "루미"})
    assert app.playable_characters() == []
    assert app.pick_character() == "sly"


def test_skips_broken_or_incomplete_or_disabled(chars_dir):
    make_char_dir(chars_dir, "sly", {"name": "루미"})
    make_char_dir(chars_dir, "no_meta", None)
    make_char_dir(chars_dir, "no_idle", {"name": "x"}, idle=False)
    make_char_dir(chars_dir, "disabled", {"name": "x", "playable": False})
    (chars_dir / "broken").mkdir()
    (chars_dir / "broken" / "meta.json").write_text("{not json", encoding="utf-8")
    (chars_dir / "broken" / "idle.png").write_bytes(b"")
    assert app.playable_characters() == []
    assert app.pick_character() == "sly"


def test_env_override(chars_dir, monkeypatch):
    make_char_dir(chars_dir, "honest", {"name": "도하"})
    make_char_dir(chars_dir, "robot", {"name": "로봇"})
    monkeypatch.setenv("HB_CHARACTER", "robot")
    assert {app.pick_character(random.Random(i)) for i in range(10)} == {"robot"}


# ---------------------------------------------------------------- 실제 에셋으로 렌더링


def game_for(cid: str) -> Game:
    rng = random.Random(0)
    g = Game(FakeLLM(referee="무효"), words=(("음식", "바나나"), ("동물", "펭귄")), rng=rng,
             lie_system=AlwaysLie(DIFFICULTIES["보통"], rng))
    g.character_id = cid
    return g


@pytest.mark.parametrize("cid", ["honest", "robot", "sly"])
def test_render_each_character(cid):
    if not (ROOT / "assets" / "characters" / cid / "meta.json").exists():
        pytest.skip(f"{cid} 에셋 없음")
    app.load_character.cache_clear()
    g = game_for(cid)
    ch = app.char_of(g)
    html = app.render_stage(g)  # 첫 화면
    assert ch.name in html and "펭귄" in html
    # 두 라운드 + 거짓말이 나오는 세 번째 질문까지
    for _ in range(2):
        g.player_ask("먹을 수 있어?")
        g.ai_act()
        g.player_answer("예")
    g.player_ask("과일이야?")
    stage = app.render_stage(g)
    if ch.has("part_tell"):
        assert "hb-tellfx" in stage and ch.tell_fx in stage
    for fn in (app.render_hud, ):
        assert ch.name in fn(g, "")
    assert app.render_my_clues(g) and app.render_ai_clues(g) and app.render_chat(g)


def test_doha_settings():
    ch = app.load_character("honest")
    assert ch.name == "도하" and ch.tell_fx == "twitch"
    assert ch.sprite_name("win") == "correct" and ch.sprite_name("lose") == "wrong"
    assert ch.sprite_name("caught") == "caught" and ch.sprite_name("ask") == "idle"
    assert ch.lines["challenge"]["player_ok"].startswith("으, 미안")
