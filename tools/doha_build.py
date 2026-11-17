"""도하(정식 캐릭터) 에셋 후처리: 배경 제거 → idle.png, 헤드밴드 카드 좌표 측정 → meta.json.

사용법:
  python tools/doha_build.py idle       # assets/characters/honest/idle.png + meta.json
  python tools/doha_build.py preview    # 카페 배경 위 합성 미리보기 (카드 포함)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CAND = ASSETS / "_candidates" / "doha"
OUT_DIR = ASSETS / "characters" / "honest"

# 선택한 기본 이미지: 후보 A 를 한 번 더 아웃페인팅해서 몸통·어깨를 채운 A2-b
# (A 는 얼굴이 너무 커서 무대에 올리면 화면을 꽉 채웠다)
SOURCE = CAND / "outpaint" / "A2_out_1.png"
MASK = CAND / "mask" / "efd39c56_000.png"        # birefnet 배경 제거 마스크
BODY_SEED = (416, 900)                            # 몸 안의 한 점 (마스크에서 몸과 이어진 영역만 남길 때 기준)

# 무대 배치 (ui/app.py 와 같은 값). 배경 1344x768 기준.
BG_W, BG_H = 1344, 768
CHAR_HEIGHT, CHAR_TOP = 900, -25

META = {
    "id": "honest",
    "name": "도하",
    "type": "정직한 타입",
    "tell": "귀 움찔 + 귀 끝 붉어짐",
    "size": [832, 1216],
    "model": "animagine-xl-4.0-opt.safetensors",
    "base": {
        "workflow": "tools/workflows/sdxl_t2i.json (tools/doha_prompt.py, v2 프롬프트)",
        "seed": 9401, "batch_index": 3,
        "candidate": "assets/_candidates/doha/7689d317_003.png",
    },
    "outpaint": [
        {"script": "tools/doha_outpaint.py", "layout": "bottom", "scale": 0.82, "seed": 9601, "batch_index": 0,
         "result": "assets/_candidates/doha/outpaint/v2n4_out_0.png",
         "why": "Animagine 이 머리 위를 잘라서, 줄여서 아래에 놓고 위·양옆만 다시 그림 (후보 A)"},
        {"script": "tools/doha_outpaint.py --layout=top", "scale": 0.58, "seed": 9601, "batch_index": 1,
         "result": "assets/_candidates/doha/outpaint/A2_out_1.png",
         "why": "A 는 얼굴이 너무 커서, 줄여서 위에 놓고 몸통·어깨를 그림 (얼굴 크기를 임시 캐릭터와 맞춤)"},
    ],
    "mask": "assets/_candidates/doha/mask/efd39c56_000.png",
    "out_dir": "assets/characters/honest",
    # ── 화면 설정 (ui/app.py 가 읽는다) ──
    "stage": {"height": 900, "top": -25, "on_table": False},
    "tell_fx": "twitch",                      # 귀 조각(part_tell)을 붉게 + 두 번 움찔
    "tell_origin": {"cx": 428, "cy": 385},    # 두 귀 사이 (움찔 기준점)
    # 엔진 표정 → 파일. 정답(correct): 본인 정답 / 플레이어 오답 / 억울한 의심을 풀었을 때
    #                   오답(wrong): 본인 오답 / 플레이어 정답(패배) / 의심이 빗나갔을 때
    "face_map": {
        "ask": "idle", "suspicious": "idle", "accuse": "idle",
        "gotcha": "correct", "smug": "correct", "win": "correct",
        "oops": "wrong", "lose": "wrong",
    },
    # 정직한 타입: 순하고 솔직한 반말. 거짓말이 들키면 바로 미안해한다
    "lines": {
        "opening": "잘 부탁해! 먼저 물어봐도 돼.",
        "game_over": {
            "player": "아, 졌다... 너 진짜 잘한다.",
            "ai": "앗, 내가 맞혔다! 재밌었어.",
            "draw": "무승부네. 다음엔 꼭 맞혀 볼게.",
        },
        "challenge": {
            "player_ok": "으, 미안... 사실 거짓말이었어.",
            "player_fail": "아니야, 진짜야! 믿어 줘.",
            "ai_ok": "역시... 그거 거짓말이었지?",
            "ai_fail": "어, 진짜였구나. 의심해서 미안해.",
        },
    },
}


def keep_body_only(m: Image.Image) -> Image.Image:
    """마스크에서 몸과 이어진 영역만 남긴다 (아웃페인팅이 배경에 그린 조각 제거)."""
    solid = m.point(lambda v: 255 if v > 40 else 0)
    ImageDraw.floodfill(solid, BODY_SEED, 128)
    body = solid.point(lambda v: 255 if v == 128 else 0)
    # 반투명 가장자리는 원래 값을 살리되, 몸 영역을 살짝 넓힌 범위 안에서만
    from PIL import ImageFilter
    near = body.filter(ImageFilter.MaxFilter(9))
    return Image.composite(m, Image.new("L", m.size, 0), near)


def cutout() -> Image.Image:
    img = Image.open(SOURCE).convert("RGB")
    m = Image.open(MASK).convert("L").resize(img.size)
    m = m.point(lambda v: 0 if v < 12 else 255 if v > 243 else v)
    m = keep_body_only(m)
    rgba = img.copy()
    rgba.putalpha(m)
    return rgba


def find_headband_card(rgba: Image.Image) -> dict:
    """헤드밴드(빨간 천) 픽셀을 찾아 정면 중앙에 붙일 카드 위치를 정한다.
    가로 중심은 헤드밴드 좌우 끝의 가운데, 세로는 가운데 열들의 헤드밴드 위아래 중간."""
    px = rgba.load()
    w, h = rgba.size

    def red(p):
        r, g, b, a = p
        return a > 200 and r > 140 and r > g * 1.6 and r > b * 1.6

    # 볼 홍조·입술 같은 다른 붉은 부분이 섞이지 않게 머리 윗부분(이미지 위 27%)만 본다
    y_max = int(h * 0.27)
    pts = [(x, y) for y in range(0, y_max) for x in range(w) if red(px[x, y])]
    # 가로 범위는 행마다 붉은 픽셀이 충분히 많은 행들로 정한다 (옆으로 늘어진 매듭 끝 제외)
    rows: dict[int, list[int]] = {}
    for x, y in pts:
        rows.setdefault(y, []).append(x)
    band_rows = {y: xs for y, xs in rows.items() if len(xs) > 80}
    left = int(sorted(min(xs) for xs in band_rows.values())[len(band_rows) // 2])
    right = int(sorted(max(xs) for xs in band_rows.values())[len(band_rows) // 2])
    cx = (left + right) // 2
    col = sorted(y for x, y in pts if abs(x - cx) <= 40)
    top, bottom = col[len(col) // 10], col[len(col) * 9 // 10]  # 위아래 10% 는 머리카락 그림자 등으로 보고 제외
    cy = (top + bottom) // 2
    band_w = right - left
    return {"cx": cx, "cy": cy, "w": round(band_w * 0.57), "band": [left, top, right, bottom]}


# 자동 측정값 보정. 헤드밴드가 약간 비대칭(오른쪽 매듭)이라 자동 측정은 오른쪽·위로 치우친다.
# 미리보기를 보고 두 눈 사이 중심과 띠의 세로 가운데에 맞춘 값 (스프라이트 832x1216 픽셀).
CARD_OVERRIDE = {"cx": 428, "cy": 255, "w": 150}


def build_idle() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rgba = cutout()
    rgba.save(OUT_DIR / "idle.png")
    card = find_headband_card(rgba)
    meta_path = OUT_DIR / "meta.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta = dict(
        META,
        headband_card=CARD_OVERRIDE,
        headband_card_measured={k: card[k] for k in ("cx", "cy", "w")},
        headband_bbox=card["band"],
    )
    if "expressions" in old:  # 표정 기록(tools/doha_inpaint.py pick)은 유지한다
        meta["expressions"] = old["expressions"]
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("idle ->", OUT_DIR / "idle.png")
    print("headband_card", meta["headband_card"], "measured", meta["headband_card_measured"])


def preview(word: str = "펭귄") -> None:
    bg = Image.open(ASSETS / "backgrounds" / "cafe.png").convert("RGBA")
    fg = Image.open(ASSETS / "backgrounds" / "cafe_table_fg.png").convert("RGBA")
    ch = Image.open(OUT_DIR / "idle.png").convert("RGBA")
    meta = json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    scale = CHAR_HEIGHT / ch.height
    cw = round(ch.width * scale)
    ch_s = ch.resize((cw, CHAR_HEIGHT), Image.LANCZOS)
    ox = (bg.width - cw) // 2
    layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    layer.paste(ch_s, (ox, CHAR_TOP), ch_s)
    out = Image.alpha_composite(bg, layer)
    # 제시어 카드 (대략적인 미리보기)
    c = meta["headband_card"]
    ccx, ccy = ox + c["cx"] * scale, CHAR_TOP + c["cy"] * scale
    cw2, chh = c["w"] * scale * 0.7, 40
    d = ImageDraw.Draw(out)
    d.rectangle([ccx - cw2 / 2, ccy - chh / 2, ccx + cw2 / 2, ccy + chh / 2], fill=(251, 248, 239, 255))
    try:
        font = ImageFont.truetype("malgunbd.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    d.text((ccx, ccy), word, fill=(29, 31, 25), font=font, anchor="mm")
    out = Image.alpha_composite(out, fg)
    path = CAND / "preview_honest.png"
    out.convert("RGB").save(path)
    print(path)


if __name__ == "__main__":
    {"idle": build_idle, "preview": preview}[sys.argv[1]]()
