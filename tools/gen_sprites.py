"""스프라이트 에셋 기록 + 후처리 스크립트.

생성 자체는 ComfyUI(MCP)로 한다:
  - 배경/캐릭터: tools/workflows/bg_t2i.json, char_t2i.json  (Z-Image-Turbo int8 템플릿)
    `57.text`(프롬프트), `57.width/height`, `57.seed`, `57/13.batch_size` 슬롯을 바꿔서 실행
  - 배경 제거: tools/workflows/remove_bg.json  (core RemoveBackground + birefnet → 마스크 PNG)

이 스크립트는 고른 후보를 assets/ 로 옮기고, 후처리(투명 PNG, 테이블 앞쪽 레이어)와
재현용 메타데이터(meta.json) 기록을 맡는다.

사용법:
  python tools/gen_sprites.py background
  python tools/gen_sprites.py character
  python tools/gen_sprites.py preview      # 합성 미리보기 assets/_candidates/preview.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CAND = ASSETS / "_candidates"

STYLE = (
    "anime cel shading, clean lineart, soft warm lighting, "
    "visual novel style, original character"
)

# ---------------------------------------------------------------------------
# 에셋 스펙 (재현용: 같은 프롬프트 + 시드 + 배치 인덱스로 다시 뽑을 수 있다)
# ---------------------------------------------------------------------------
BACKGROUND = {
    "name": "cafe",
    "workflow": "tools/workflows/bg_t2i.json",
    "model": "z_image_turbo_int8_convrot.safetensors",
    "size": [1344, 768],
    "seed": 2001,
    "batch_size": 4,
    "batch_index": 2,
    "prompt": (
        "anime cel shading, clean lineart, soft warm lighting, visual novel style background art. "
        "Point-of-view shot from a person seated at a cafe table. The wooden tabletop is very close "
        "to the camera and fills the entire bottom quarter of the image edge to edge, with a coffee "
        "cup on the near right of the table. Directly across the table, in the exact center of the "
        "image, one single empty upholstered armchair faces the camera. No other chairs in the "
        "foreground, no people. Behind the armchair: cozy cafe interior, warm pendant lamps, wooden "
        "shelves with plants and coffee jars, window with soft afternoon light, gentle background "
        "blur. Eye-level camera, symmetrical, vibrant warm colors"
    ),
    "source": CAND / "bg" / "85866271_002.png",
    "out": ASSETS / "backgrounds" / "cafe.png",
    "fg_out": ASSETS / "backgrounds" / "cafe_table_fg.png",
    # 테이블 윗면(앞쪽 레이어) 다각형. 윗변 y=536, 양옆은 원근으로 아래로 기울어짐.
    "table_polygon": [(0, 695), (272, 536), (1152, 536), (1344, 626), (1344, 768), (0, 768)],
}

CHARACTER = {
    "id": "sly",
    "type": "능글맞은 타입",
    "workflow": "tools/workflows/char_t2i.json",
    "model": "z_image_turbo_int8_convrot.safetensors",
    "size": [832, 1216],
    "seed": 3001,
    "batch_size": 4,
    "batch_index": 1,
    "prompt": (
        "anime cel shading, clean lineart, soft warm lighting, visual novel character sprite, "
        "original character. Waist-up portrait of a young adult woman sitting and facing the camera "
        "directly, front view, symmetrical pose, looking straight into the viewer's eyes. Sly, "
        "teasing personality: a confident playful smirk with one corner of the mouth raised, slightly "
        "narrowed mischievous golden eyes, a small visible fang. Messy shoulder-length lavender-silver "
        "hair with a side-swept fringe, a small black hairpin. Wearing an oversized dark olive-green "
        "hoodie over a white collared shirt, sleeves slightly too long. Arms relaxed at her sides, "
        "hands out of frame. Plain flat solid light gray background, no shadows on background, no "
        "props, no text, full head and both shoulders fully visible, centered with margin above the head"
    ),
    "base_source": CAND / "char" / "5f3443f4_001.png",
    # 헤어밴드: 위 원본의 이마 영역(headband_mask.png)을 인페인팅 (tools/workflows/inpaint.json)
    "headband_inpaint": {
        "workflow": "tools/workflows/inpaint.json",
        "mask": CAND / "char" / "headband_mask.png",
        "seed": 4101, "batch_size": 4, "batch_index": 3, "denoise": 1.0,
    },
    "source": CAND / "band" / "a0049600_003.png",
    "mask": CAND / "mask" / "0995b9ba_000.png",
    "out_dir": ASSETS / "characters" / "sly",
    # 제시어 카드를 붙일 위치 (스프라이트 832x1216 기준 픽셀). 빨간 밴드 가운데.
    # 표정 스프라이트도 같은 원본에서 인페인팅하므로 이 위치를 공유한다.
    "headband_card": {"cx": 416, "cy": 174, "w": 230},
}


# ---------------------------------------------------------------------------
# 후처리
# ---------------------------------------------------------------------------
def cutout(src: Path, mask: Path, out: Path) -> Image.Image:
    """원본 + 배경제거 마스크 → 투명 PNG. 가장자리 반투명 노이즈를 살짝 정리한다."""
    img = Image.open(src).convert("RGB")
    m = Image.open(mask).convert("L").resize(img.size)
    # 거의 투명/거의 불투명한 값을 끝으로 밀어 가장자리 얼룩을 줄인다
    m = m.point(lambda v: 0 if v < 12 else 255 if v > 243 else v)
    rgba = img.copy()
    rgba.putalpha(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out)
    return rgba


def table_foreground(src: Path, polygon, out: Path) -> Image.Image:
    """배경에서 테이블 윗면만 남긴 투명 레이어. 캐릭터 앞에 깔아 '테이블 너머'로 보이게 한다."""
    bg = Image.open(src).convert("RGB")
    mask = Image.new("L", bg.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(0.8))  # 경계 계단 현상 완화
    fg = bg.copy()
    fg.putalpha(mask)
    out.parent.mkdir(parents=True, exist_ok=True)
    fg.save(out)
    return fg


def _meta(v):
    """Path → 프로젝트 기준 상대경로 문자열 (중첩 dict 포함)."""
    if isinstance(v, dict):
        return {k: _meta(x) for k, x in v.items()}
    if isinstance(v, Path):
        return v.relative_to(ROOT).as_posix()
    return v


def build_background() -> None:
    s = BACKGROUND
    s["out"].parent.mkdir(parents=True, exist_ok=True)
    Image.open(s["source"]).convert("RGB").save(s["out"])
    table_foreground(s["source"], s["table_polygon"], s["fg_out"])
    (s["out"].parent / "cafe.meta.json").write_text(
        json.dumps(_meta(s), ensure_ascii=False, indent=2), encoding="utf-8")
    print("background ->", s["out"], s["fg_out"])


def build_character() -> None:
    s = CHARACTER
    out = s["out_dir"] / "idle.png"
    cutout(s["source"], s["mask"], out)
    (s["out_dir"] / "meta.json").write_text(
        json.dumps(_meta(s), ensure_ascii=False, indent=2), encoding="utf-8")
    print("character ->", out)


# 화면 배치 (ui/app.py 와 같은 값). 배경 1344x768 기준 픽셀.
CHAR_HEIGHT = 900   # 캐릭터 스프라이트 표시 높이
CHAR_TOP = -25      # 캐릭터 스프라이트 상단 y (음수 = 화면 위로 살짝 넘침, 머리 위 여백만 잘림)


def preview() -> None:
    bg = Image.open(BACKGROUND["out"]).convert("RGBA")
    ch = Image.open(CHARACTER["out_dir"] / "idle.png")
    w = round(ch.width * CHAR_HEIGHT / ch.height)
    ch = ch.resize((w, CHAR_HEIGHT), Image.LANCZOS)
    layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    layer.paste(ch, ((bg.width - w) // 2, CHAR_TOP), ch)
    bg.alpha_composite(layer)
    bg.alpha_composite(Image.open(BACKGROUND["fg_out"]))
    out = CAND / "preview.png"
    bg.convert("RGB").save(out)
    print("preview ->", out)


# ---------------------------------------------------------------------------
# 표정 / 텔 스프라이트 (기본 헤어밴드 이미지에서 얼굴 일부만 인페인팅)
# ---------------------------------------------------------------------------
COMFY = "http://127.0.0.1:8188"
EXPR_DIR = CAND / "expr"
EXPR_SEED = 5001
EXPR_BATCH = 4

EXPR_BASE_PROMPT = (
    "anime cel shading, clean lineart, visual novel character sprite, same character, same art style. "
    "A young woman with messy lavender-silver hair, golden eyes, wearing a thick red headband, "
    "front view facing the viewer. "
)

# 마스크 도형: ("ellipse" | "rect", (x0, y0, x1, y1)) — 스프라이트 832x1216 기준
# 직사각형 하나로 얼굴을 덮으면 마스크 경계가 앞머리를 가로질러 머리카락 결이 끊긴다.
# → 눈·입·볼만 타원으로 잡고 크게 흐리게(부드러운 그라데이션) 해서 DifferentialDiffusion 으로 그린다.
#   가장자리로 갈수록 원본이 많이 남아서 머리카락이 자연스럽게 이어진다.
EYE_L = ("ellipse", (290, 292, 390, 376))
EYE_R = ("ellipse", (426, 282, 530, 370))
MOUTH = ("ellipse", (356, 380, 494, 448))
CHEEKS = [("ellipse", (300, 356, 384, 408)), ("ellipse", (448, 350, 534, 402))]
FACE = [EYE_L, EYE_R, MOUTH, *CHEEKS]
FACE_FEATHER = 16

# 텔(거짓말 버릇)은 표정 스프라이트가 아니라 UI 애니메이션(움찔 + 땀)으로 한다. 페이크 텔은 없다.
EXPRESSIONS: dict[str, dict] = {
    "ask": {"mask": FACE, "denoise": 1.0,
            "face": "thinking expression, eyes looking upward to the side, lips slightly pursed, curious"},
    "suspicious": {"mask": FACE, "denoise": 1.0,
                   "face": "suspicious skeptical expression, narrowed half-lidded eyes staring at the "
                           "viewer, slight doubtful frown, one eyebrow raised"},
    "accuse": {"mask": FACE, "denoise": 1.0,
               "face": "accusing expression, sharp wide eyes, eyebrows angled down, mouth open shouting"},
    "caught": {"mask": FACE, "denoise": 1.0,
               "face": "flustered caught-off-guard expression, wide startled eyes, heavy blush, "
                       "wavy nervous mouth"},
    "smug": {"mask": FACE, "denoise": 1.0,
             "face": "very smug teasing expression, half-closed eyes, wide confident grin showing a small fang"},
    "gotcha": {"mask": FACE, "denoise": 1.0,
               "face": "triumphant excited expression, sparkling eyes, big open-mouth grin with a small fang"},
    "oops": {"mask": FACE, "denoise": 1.0,
             "face": "awkward apologetic expression, eyebrows lowered, sheepish crooked smile, eyes looking away"},
    "win": {"mask": FACE, "denoise": 1.0,
            "face": "joyful laughing expression, eyes happily closed in upward arcs, wide open smile"},
    "lose": {"mask": FACE, "denoise": 1.0,
             "face": "sad defeated expression, teary eyes, pouting trembling mouth, eyebrows drooping"},
}


def _make_mask(shapes, size, feather: float = 6.0) -> Image.Image:
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for kind, box in shapes:
        if kind == "ellipse":
            d.ellipse(box, fill=255)
        else:
            d.rounded_rectangle(box, radius=24, fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather))


def _comfy_upload(path: Path) -> str:
    import httpx
    with open(path, "rb") as f:
        r = httpx.post(f"{COMFY}/upload/image", files={"image": (path.name, f, "image/png")},
                       data={"overwrite": "true"}, timeout=60)
    r.raise_for_status()
    return r.json()["name"]


def _comfy_run(workflow: dict, timeout: float = 300) -> list[Image.Image]:
    import io
    import time
    import uuid

    import httpx
    r = httpx.post(f"{COMFY}/prompt", json={"prompt": workflow, "client_id": uuid.uuid4().hex}, timeout=30)
    r.raise_for_status()
    pid = r.json()["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = httpx.get(f"{COMFY}/history/{pid}", timeout=30).json()
        if pid in h:
            status = h[pid].get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error: {status}")
            images = []
            for out in h[pid]["outputs"].values():
                for im in out.get("images", []):
                    v = httpx.get(f"{COMFY}/view", params=im, timeout=60)
                    images.append(Image.open(io.BytesIO(v.content)).convert("RGB"))
            return images
        time.sleep(1.0)
    raise TimeoutError(pid)


def generate_expressions(keys: list[str]) -> None:
    """각 표정마다 EXPR_BATCH 장 후보를 만들고 얼굴 확대 비교 시트를 저장한다."""
    base_path = CHARACTER["source"]
    base_name = _comfy_upload(base_path)
    base = Image.open(base_path).convert("RGB")
    wf_template = json.loads((ROOT / "tools/workflows/inpaint.json").read_text(encoding="utf-8"))
    for key in keys:
        spec = EXPRESSIONS[key]
        out_dir = EXPR_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        mask = _make_mask(spec["mask"], base.size, spec.get("feather", FACE_FEATHER))
        mask_path = out_dir / "mask.png"
        mask.save(mask_path)
        mask_name = _comfy_upload(mask_path)

        wf = json.loads(json.dumps(wf_template))
        wf["7"]["inputs"]["image"] = base_name
        wf["8"]["inputs"]["image"] = mask_name
        wf["5"]["inputs"]["text"] = EXPR_BASE_PROMPT + "Facial expression: " + spec["face"]
        wf["11"]["inputs"]["seed"] = EXPR_SEED
        wf["11"]["inputs"]["denoise"] = spec["denoise"]
        wf["15"]["inputs"]["amount"] = EXPR_BATCH
        wf["14"]["inputs"]["filename_prefix"] = f"headband/expr_{key}"
        # 부드러운 마스크를 픽셀별 강도로 쓰도록 DifferentialDiffusion 을 끼운다
        wf["16"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": ["4", 0], "strength": 1.0}}
        wf["11"]["inputs"]["model"] = ["16", 0]
        images = _comfy_run(wf)

        crops = []
        for i, im in enumerate(images):
            # 마스크 밖은 원본 픽셀로 되돌린다 → 바뀌는 건 마스크 부위뿐
            merged = Image.composite(im, base, mask)
            merged.save(out_dir / f"{i}.png")
            crops.append(merged.crop((250, 230, 590, 470)))
        sheet = Image.new("RGB", (340 * (len(crops) + 1), 260), "white")
        sheet.paste(base.crop((250, 230, 590, 470)), (0, 20))
        d = ImageDraw.Draw(sheet)
        d.text((4, 2), "base", fill=(0, 0, 0))
        for i, c in enumerate(crops):
            sheet.paste(c, (340 * (i + 1), 20))
            d.text((340 * (i + 1) + 4, 2), f"{key} #{i}", fill=(200, 0, 0))
        sheet.save(EXPR_DIR / f"{key}_sheet.png")
        print(key, "->", EXPR_DIR / f"{key}_sheet.png")


def pick_expression(key: str, index: int) -> None:
    """후보 하나를 골라 투명 스프라이트로 만들고 선택 기록을 남긴다."""
    s = CHARACTER
    merged = Image.open(EXPR_DIR / key / f"{index}.png").convert("RGB")
    alpha = Image.open(s["mask"]).convert("L").resize(merged.size)
    alpha = alpha.point(lambda v: 0 if v < 12 else 255 if v > 243 else v)
    merged.putalpha(alpha)  # 실루엣은 기본 이미지와 같으므로 같은 알파를 쓴다
    merged.save(s["out_dir"] / f"{key}.png")

    rec_path = s["out_dir"] / "expressions.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
    spec = EXPRESSIONS[key]
    rec[key] = {
        "base": _meta(s["source"]), "workflow": "tools/workflows/inpaint.json",
        "prompt": EXPR_BASE_PROMPT + "Facial expression: " + spec["face"],
        "mask": [{"shape": k, "box": list(b)} for k, b in spec["mask"]],
        "seed": EXPR_SEED, "batch_size": EXPR_BATCH, "batch_index": index, "denoise": spec["denoise"],
        "mask_feather": spec.get("feather", FACE_FEATHER), "differential_diffusion": True,
    }
    rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(key, "->", s["out_dir"] / f"{key}.png")


# ---------------------------------------------------------------------------
# 로봇 캐릭터: 얼굴 = 검은 디스플레이 화면. 표정은 화면 안쪽만 인페인팅해서 LED 얼굴로 뽑는다.
# ---------------------------------------------------------------------------
ROBOT = {
    "id": "robot",
    "name": "로봇",  # 임시 이름
    "workflow": "tools/workflows/char_t2i.json",
    "model": "z_image_turbo_int8_convrot.safetensors",
    "size": [832, 1216],
    "seed": 10001, "batch_size": 4, "batch_index": 3,
    "prompt": (
        "anime cel shading, clean lineart, soft warm lighting, visual novel character sprite, original character "
        "design. A cute small friendly robot sitting and facing the camera directly, front view, symmetrical, "
        "waist-up framing. Round head with a smooth glossy cream-white shell and a large rounded black glass display "
        "screen as its face, the face screen is completely blank and dark with no eyes and no mouth drawn. One short "
        "antenna with a small round tip on top of the head, small round ear discs on both sides of the head. A deep "
        "burgundy red cloth headband tied horizontally around its head above the face screen, with a blank plain "
        "white rectangular card tucked into the front center of the headband, landscape orientation, completely empty "
        "with no writing. A visible slim mechanical neck joint with segmented rings between head and body. Rounded "
        "compact body with cream-white shell and dark olive-green panels, small round shoulder joints, short arms "
        "relaxed at its sides, hands out of frame. Plain flat solid light gray background, no props, no text, "
        "centered, full head visible with margin above the antenna"
    ),
    "source": CAND / "v2_robot" / "19ad8f0b_003.png",
    "out_dir": ASSETS / "characters" / "robot",
    "screen_seed": (420, 640),  # 화면 안쪽 한 점 (화면 영역 찾기용)
    # ── UI 좌표 (스프라이트 픽셀) ──
    "headband_card": {"cx": 420, "cy": 392, "w": 300, "printed": True},  # 이미지에 카드가 그려져 있음 → 글자만 얹는다
    "antenna_tip": {"cx": 414, "cy": 174, "r": 21},  # 텔: 회로 이상 스파크 (part_tell.png 의 중심)
    # 무대 배치 (배경 1344x768 기준 px). 작은 로봇이라 의자가 아니라 테이블 위에 앉는다 → 테이블 앞면 레이어 안 씀
    "stage": {"height": 546, "top": 72, "on_table": True, "shadow": {"cx": 672, "cy": 606, "w": 300, "h": 30}},
    # 엔진 표정 이름 → 이 캐릭터의 파일 이름 (기획서 7.3: 정답/오답은 상황 여러 개를 표정 하나로)
    "face_map": {"ask": "thinking", "gotcha": "correct", "smug": "correct", "win": "correct",
                 "oops": "wrong", "lose": "wrong"},
}
ROBOT_DIR = CAND / "robot_expr"
ROBOT_EXPR_SEED = 21001

ROBOT_BASE_PROMPT = (
    "anime cel shading, clean lineart, soft warm lighting, visual novel character sprite, same character, same art "
    "style. A cute small round robot with a glossy cream-white shell head and a large black glass display screen "
    "face. The dark display screen shows a simple cute face made of softly glowing warm amber LED light: "
)
# 기획서 7.3 / 9.2: 전신 표정 이미지 (화면 안쪽만 인페인팅)
ROBOT_FACES = {
    "idle": "two calm rounded oval eyes, relaxed and friendly, no mouth",
    "correct": "two happy upward arc eyes like ^ ^ and a wide joyful smile, glad and relieved",
    "wrong": "two sad droopy eyes with small tear drops and a downturned frustrated wavy mouth",
    "caught": "two wide round startled eyes, a wavy nervous line mouth, small blush marks on the screen",
    "suspicious": "two flat narrow half-closed line eyes, skeptical squint, a small flat line mouth",
    "accuse": "two sharp angry slanted eyes pointing inward, and a small exclamation mark",
    "thinking": "two calm rounded oval eyes with the glowing pupils shifted up toward the upper right corner, "
                "looking upward while pondering, a small flat line mouth, and three small glowing dots floating "
                "near the top right",
}


def _screen_mask(img: Image.Image, seed: tuple[int, int]) -> Image.Image:
    """검은 화면 영역 마스크. 채도가 낮은 어두운 색만 잡아서 버건디 헤어밴드로 번지지 않게 한다."""
    from PIL import ImageChops

    W, H = img.size
    dark = Image.new("L", (W, H))
    dark.putdata([255 if sum(p) / 3 < 115 and max(p) - min(p) < 38 else 0 for p in img.getdata()])
    er = dark.filter(ImageFilter.MinFilter(7))
    ImageDraw.floodfill(er, seed, 128)
    scr = er.point(lambda v: 255 if v == 128 else 0).filter(ImageFilter.MaxFilter(7))
    inv = scr.copy()
    ImageDraw.floodfill(inv, (0, 0), 128)
    scr = ImageChops.lighter(scr, inv.point(lambda v: 255 if v == 0 else 0))  # 반사광 등 구멍 메우기
    return scr.filter(ImageFilter.GaussianBlur(5)).point(lambda v: 255 if v > 140 else 0)


def robot_base() -> None:
    """화면 마스크와 배경 제거용 원본을 준비한다. 배경 마스크는 remove_bg.json 으로 따로 만든다."""
    img = Image.open(ROBOT["source"]).convert("RGB")
    ROBOT_DIR.mkdir(parents=True, exist_ok=True)
    scr = _screen_mask(img, ROBOT["screen_seed"])
    scr.save(ROBOT_DIR / "screen.png")
    # 인페인팅 마스크: 테두리(검은 외곽선)를 지키도록 6px 안쪽, 경계는 살짝 부드럽게
    scr.filter(ImageFilter.MinFilter(13)).filter(ImageFilter.GaussianBlur(2)).save(ROBOT_DIR / "inpaint_mask.png")
    print("screen bbox", scr.getbbox())


def robot_expressions(keys: list[str]) -> None:
    base = Image.open(ROBOT["source"]).convert("RGB")
    base_name = _comfy_upload(ROBOT["source"])
    mask = Image.open(ROBOT_DIR / "inpaint_mask.png").convert("L")
    mask_name = _comfy_upload(ROBOT_DIR / "inpaint_mask.png")
    wf_template = json.loads((ROOT / "tools/workflows/inpaint.json").read_text(encoding="utf-8"))
    x0, y0, x1, y1 = mask.getbbox()
    crop_box = (x0 - 40, y0 - 170, x1 + 40, y1 + 30)  # 화면 + 카드까지 보이게
    for key in keys:
        wf = json.loads(json.dumps(wf_template))
        wf["7"]["inputs"]["image"] = base_name
        wf["8"]["inputs"]["image"] = mask_name
        wf["5"]["inputs"]["text"] = ROBOT_BASE_PROMPT + ROBOT_FACES[key]
        wf["11"]["inputs"]["seed"] = ROBOT_EXPR_SEED
        wf["11"]["inputs"]["denoise"] = 1.0
        wf["15"]["inputs"]["amount"] = EXPR_BATCH
        wf["14"]["inputs"]["filename_prefix"] = f"headband/robot_{key}"
        images = _comfy_run(wf)
        out_dir = ROBOT_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        crops = []
        for i, im in enumerate(images):
            merged = Image.composite(im, base, mask)  # 화면 밖은 원본 그대로
            merged.save(out_dir / f"{i}.png")
            crops.append(merged.crop(crop_box))
        cw, ch = crops[0].size
        sheet = Image.new("RGB", (cw * len(crops), ch + 20), "white")
        d = ImageDraw.Draw(sheet)
        for i, c in enumerate(crops):
            sheet.paste(c, (cw * i, 20))
            d.text((cw * i + 4, 4), f"{key} #{i}", fill=(200, 0, 0))
        sheet.save(ROBOT_DIR / f"{key}_sheet.png")
        print(key, "->", ROBOT_DIR / f"{key}_sheet.png")


def robot_pick(key: str, index: int, alpha_path: Path) -> None:
    """후보를 골라 투명 스프라이트로. 알파는 배경 제거 마스크(실루엣은 모든 표정이 같다)."""
    img = Image.open(ROBOT_DIR / key / f"{index}.png").convert("RGB")
    alpha = Image.open(alpha_path).convert("L").resize(img.size).point(lambda v: 0 if v < 12 else 255 if v > 243 else v)
    img.putalpha(alpha)
    out = ROBOT["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / f"{key}.png")
    rec_path = out / "expressions.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
    rec[key] = {"base": _meta(ROBOT["source"]), "workflow": "tools/workflows/inpaint.json",
                "mask": _meta(ROBOT_DIR / "inpaint_mask.png"), "prompt": ROBOT_BASE_PROMPT + ROBOT_FACES[key],
                "seed": ROBOT_EXPR_SEED, "batch_size": EXPR_BATCH, "batch_index": index, "denoise": 1.0,
                "alpha": _meta(alpha_path)}
    rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = _meta({k: v for k, v in ROBOT.items() if k != "out_dir"})
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(key, "->", out / f"{key}.png")


# 부위 조각 (기획서 9.2): idle 위에 겹치는 832x1216 투명 PNG
ROBOT_PARTS = {
    # 말하기: 화면 아래쪽 입만 벌린 LED 입으로
    "talk": {"mask": ("ellipse", (330, 622, 506, 716)), "clip_screen": True, "diff_alpha": False,
             "prompt": "exactly the same two calm glowing oval eyes, and a small open rounded glowing mouth as if "
                       "talking, same warm amber LED glow"},
    # 거짓말 버릇: 안테나 끝 구슬 주변에 작은 스파크 (회로 이상)
    # 안테나 자체(구슬 + 막대)는 마스크에서 빼서 그대로 두고, 그 주변에만 스파크를 그린다
    "tell": {"mask": ("ellipse", (334, 96, 494, 254)), "clip_screen": False, "diff_alpha": True,
             "exclude": [("ellipse", (382, 143, 446, 207)), ("rect", (400, 195, 430, 262))],
             "prompt": "the same small antenna with a round ball tip, with a few tiny bright yellow-white electric "
                       "sparks crackling around the antenna tip, small zigzag lightning sparks, a circuit glitch, "
                       "plain flat light gray background around it"},
}
ROBOT_IDLE_INDEX = 1  # 고른 idle 후보 (robot_expr/idle/<n>.png, 배경 있는 원본)


def robot_parts(keys: list[str]) -> None:
    base_path = ROBOT_DIR / "idle" / f"{ROBOT_IDLE_INDEX}.png"
    base = Image.open(base_path).convert("RGB")
    base_name = _comfy_upload(base_path)
    screen = Image.open(ROBOT_DIR / "inpaint_mask.png").convert("L")
    wf_template = json.loads((ROOT / "tools/workflows/inpaint.json").read_text(encoding="utf-8"))
    for key in keys:
        spec = ROBOT_PARTS[key]
        m = Image.new("L", base.size, 0)
        dm = ImageDraw.Draw(m)
        dm.ellipse(spec["mask"][1], fill=255)
        for kind, box in spec.get("exclude", []):
            (dm.ellipse if kind == "ellipse" else dm.rectangle)(box, fill=0)
        m = m.filter(ImageFilter.GaussianBlur(3 if spec.get("exclude") else 6))
        if spec["clip_screen"]:
            from PIL import ImageChops
            m = ImageChops.multiply(m, screen)
        out_dir = ROBOT_DIR / f"part_{key}"
        out_dir.mkdir(parents=True, exist_ok=True)
        m.save(out_dir / "mask.png")
        wf = json.loads(json.dumps(wf_template))
        wf["7"]["inputs"]["image"] = base_name
        wf["8"]["inputs"]["image"] = _comfy_upload(out_dir / "mask.png")
        wf["5"]["inputs"]["text"] = ROBOT_BASE_PROMPT + spec["prompt"]
        wf["11"]["inputs"]["seed"] = ROBOT_EXPR_SEED
        wf["11"]["inputs"]["denoise"] = 1.0
        wf["15"]["inputs"]["amount"] = EXPR_BATCH
        wf["14"]["inputs"]["filename_prefix"] = f"headband/robot_part_{key}"
        images = _comfy_run(wf)
        x0, y0, x1, y1 = spec["mask"][1]
        box = (x0 - 30, y0 - 30, x1 + 30, y1 + 30)
        crops = [base.crop(box)]
        for i, im in enumerate(images):
            merged = Image.composite(im, base, m)
            merged.save(out_dir / f"{i}.png")
            crops.append(merged.crop(box))
        cw, ch = crops[0].size
        sheet = Image.new("RGB", (cw * len(crops), ch + 20), "white")
        d = ImageDraw.Draw(sheet)
        for i, c in enumerate(crops):
            sheet.paste(c, (cw * i, 20))
            d.text((cw * i + 4, 4), "base" if i == 0 else f"{key} #{i - 1}", fill=(200, 0, 0))
        sheet.save(ROBOT_DIR / f"part_{key}_sheet.png")
        print(key, "->", ROBOT_DIR / f"part_{key}_sheet.png")


def robot_part_pick(key: str, index: int, alpha_path: Path) -> None:
    """조각 = 같은 캔버스 크기 투명 PNG. 말하기는 마스크 모양, 스파크는 원본과 달라진 픽셀만 남긴다."""
    from PIL import ImageChops

    spec = ROBOT_PARTS[key]
    out_dir = ROBOT_DIR / f"part_{key}"
    base = Image.open(ROBOT_DIR / "idle" / f"{ROBOT_IDLE_INDEX}.png").convert("RGB")
    img = Image.open(out_dir / f"{index}.png").convert("RGB")
    m = Image.open(out_dir / "mask.png").convert("L")
    if spec["diff_alpha"]:
        diff = ImageChops.difference(img, base).convert("L").point(lambda v: min(255, max(0, (v - 12) * 6)))
        alpha = ImageChops.multiply(diff.filter(ImageFilter.GaussianBlur(0.8)), m)
    else:
        sil = Image.open(alpha_path).convert("L").resize(img.size)
        alpha = ImageChops.multiply(m, sil)
    img.putalpha(alpha)
    img.save(ROBOT["out_dir"] / f"part_{key}.png")
    rec_path = ROBOT["out_dir"] / "expressions.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
    rec[f"part_{key}"] = {"base": _meta(ROBOT_DIR / "idle" / f"{ROBOT_IDLE_INDEX}.png"),
                          "mask": {"shape": spec["mask"][0], "box": list(spec["mask"][1])},
                          "prompt": ROBOT_BASE_PROMPT + spec["prompt"], "seed": ROBOT_EXPR_SEED,
                          "batch_size": EXPR_BATCH, "batch_index": index,
                          "alpha": "원본과 달라진 픽셀" if spec["diff_alpha"] else "마스크 × 실루엣"}
    rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(key, "->", ROBOT["out_dir"] / f"part_{key}.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "robot-parts":
        robot_parts(sys.argv[2:] or list(ROBOT_PARTS))
        sys.exit()
    if cmd == "robot-part-pick":  # python tools/gen_sprites.py robot-part-pick <alpha.png> talk 1 tell 2
        alpha_path, pairs = Path(sys.argv[2]).resolve(), sys.argv[3:]
        for k, i in zip(pairs[::2], pairs[1::2]):
            robot_part_pick(k, int(i), alpha_path)
        sys.exit()
    if cmd == "robot-base":
        robot_base()
        sys.exit()
    if cmd == "robot-expr":
        robot_expressions(sys.argv[2:] or list(ROBOT_FACES))
        sys.exit()
    if cmd == "robot-pick":  # python tools/gen_sprites.py robot-pick <alpha.png> idle 2 ask 0 ...
        alpha_path, pairs = Path(sys.argv[2]).resolve(), sys.argv[3:]
        for k, i in zip(pairs[::2], pairs[1::2]):
            robot_pick(k, int(i), alpha_path)
        sys.exit()
    if cmd == "expressions":
        generate_expressions(sys.argv[2:] or list(EXPRESSIONS))
        sys.exit()
    if cmd == "pick":  # python tools/gen_sprites.py pick tell 2 [fake1 0 ...]
        pairs = sys.argv[2:]
        for k, i in zip(pairs[::2], pairs[1::2]):
            pick_expression(k, int(i))
        sys.exit()
    if cmd in ("background", "all"):
        build_background()
    if cmd in ("character", "all"):
        build_character()
    if cmd in ("preview", "all"):
        preview()
