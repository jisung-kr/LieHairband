"""도하 표정·부위 조각 인페인팅 (SDXL Animagine, 부분 img2img).

기본 이미지(배경 있는 원본)의 해당 부위만 다시 그리고, 결과를 원본과 부드럽게 합성한 뒤
idle.png 의 투명도(배경 제거)를 그대로 적용한다. 같은 캔버스(832x1216)라 화면에서 좌표 계산 없이 겹친다.

  전신 표정: correct(기쁨), wrong(아쉬움), caught(간파당함)  → 얼굴 마스크
  부위 조각: part_talk(입 벌림) → 입 마스크,  part_tell(귀 끝 붉어짐) → 양쪽 귀 마스크

사용법:
  python tools/doha_inpaint.py prepare <키...>        → 마스크 + 워크플로 생성 (ComfyUI input 에 마스크 복사)
  (ComfyUI 실행)
  python tools/doha_inpaint.py composite <키> <결과 png...> → 후보 저장 (assets/_candidates/doha/expr/)
  python tools/doha_inpaint.py pick <키> <후보 png>    → assets/characters/honest/<파일> 로 확정
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doha_prompt import NEGATIVE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMFY_INPUT = Path(r"C:\Users\user\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
BASE_INPUT = "doha_base_A2.png"                       # ComfyUI input 에 있는 기본 이미지 (배경 있음)
BASE = ROOT / "assets" / "_candidates" / "doha" / "outpaint" / "A2_out_1.png"
IDLE = ROOT / "assets" / "characters" / "honest" / "idle.png"
WORK = ROOT / "assets" / "_candidates" / "doha" / "expr"
OUT_DIR = ROOT / "assets" / "characters" / "honest"
W, H = 832, 1216

# 캐릭터 정체성 태그 (표정 태그는 키마다 따로)
IDENTITY = (
    "1boy, solo, male focus, bishounen, handsome, young adult, upper body, facing viewer, straight-on, "
    "looking at viewer, almond eyes, long eyelashes, brown eyes, sharp jawline, "
    "very short hair, undercut, brown hair, both ears visible, human ears, "
    "red headband, white collared shirt, navy blue cardigan, simple background, "
    "anime coloring, clean lineart"
)
QUALITY = "masterpiece, high score, great score, absurdres"

# 얼굴 부위 좌표 (스프라이트 832x1216 픽셀, grid_head.png 로 확인)
FACE = [(300, 308, 545, 512)]                 # 눈썹~턱. 헤드밴드(위쪽 ~300)는 건드리지 않는다
MOUTH = [(380, 432, 470, 482)]
EARS = [(278, 340, 330, 432), (520, 340, 575, 432)]

SPECS = {
    "correct": {"file": "correct.png", "regions": FACE, "denoise": 0.7, "feather": 14,
                "tags": "happy, joyful, big smile, open mouth, :d, bright eyes, raised eyebrows"},
    "wrong": {"file": "wrong.png", "regions": FACE, "denoise": 0.7, "feather": 14,
              "tags": "disappointed, frustrated, frown, furrowed brow, pout, closed mouth, troubled eyebrows"},
    "caught": {"file": "caught.png", "regions": FACE, "denoise": 0.72, "feather": 14,
               "tags": "flustered, surprised, wide eyes, embarrassed, blush, sweatdrop, open mouth, nervous smile"},
    "talk": {"file": "part_talk.png", "regions": MOUTH, "denoise": 0.65, "feather": 8, "part": True,
             "tags": "talking, open mouth, gentle expression, calm eyes"},
    "tell": {"file": "part_tell.png", "regions": EARS, "denoise": 0.55, "feather": 6, "part": True,
             "tags": "red ears, blushing ears, ear blush, embarrassed, gentle smile, closed mouth"},
}


def region_mask(spec: dict, blur: bool = True) -> Image.Image:
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for box in spec["regions"]:
        d.ellipse(box, fill=255)
    return m.filter(ImageFilter.GaussianBlur(spec["feather"])) if blur else m


def prepare(keys: list[str]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    for key in keys:
        spec = SPECS[key]
        mask_name = f"doha_mask_{key}.png"
        region_mask(spec).convert("RGB").save(COMFY_INPUT / mask_name)
        positive = f"{IDENTITY}, {spec['tags']}, {QUALITY}"
        wf = {
            "1": {"class_type": "LoadImage", "inputs": {"image": BASE_INPUT}},
            "2": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
            "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "animagine-xl-4.0-opt.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["3", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["3", 1]}},
            "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["3", 2]}},
            "7": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["6", 0], "mask": ["2", 0]}},
            "8": {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["7", 0], "amount": 4}},
            "9": {"class_type": "KSampler", "inputs": {
                "seed": 9700, "steps": 28, "cfg": 5.0, "sampler_name": "euler_ancestral", "scheduler": "normal",
                "denoise": spec["denoise"], "model": ["3", 0], "positive": ["4", 0], "negative": ["5", 0],
                "latent_image": ["8", 0]}},
            "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 2]}},
            "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"doha/expr_{key}", "images": ["10", 0]}},
        }
        out = ROOT / "tools" / "workflows" / f"doha_inpaint_{key}.json"
        out.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
        print(out)


def build(key: str, gen_path: str) -> Image.Image:
    """생성 결과를 원본과 합성 + idle 투명도 적용. 부위 조각은 그 부위만 남긴 투명 PNG."""
    spec = SPECS[key]
    base = Image.open(BASE).convert("RGB")
    gen = Image.open(gen_path).convert("RGB").resize((W, H))
    m = region_mask(spec)
    merged = Image.composite(gen, base, m)
    alpha = Image.open(IDLE).getchannel("A")
    out = merged.convert("RGBA")
    if spec.get("part"):
        # 부위 조각: 마스크 영역만 남기고 나머지는 투명. 몸 밖(배경)은 idle 투명도로 지운다.
        part_alpha = Image.composite(alpha, Image.new("L", (W, H), 0), m)
        # 완전히 투명한 곳의 색 정보는 비워서 파일 크기를 줄인다 (브라우저에서 매번 불러오는 파일)
        out = Image.composite(out, Image.new("RGBA", (W, H), (0, 0, 0, 0)), part_alpha.point(lambda v: 255 if v else 0))
        out.putalpha(part_alpha)
    else:
        out.putalpha(alpha)
    return out


def composite(key: str, results: list[str]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        path = WORK / f"{key}_{i}.png"
        build(key, r).save(path)
        print(path)


def pick(key: str, cand: str) -> None:
    """후보를 확정 파일로 복사하고 meta.json 에 어떤 후보를 골랐는지 기록한다."""
    spec = SPECS[key]
    dst = OUT_DIR / spec["file"]
    shutil.copyfile(cand, dst)
    meta_path = OUT_DIR / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("expressions", {})[key] = {
        "file": spec["file"],
        "workflow": f"tools/workflows/doha_inpaint_{key}.json (tools/doha_inpaint.py)",
        "seed": 9700, "denoise": spec["denoise"], "tags": spec["tags"],
        "candidate": Path(cand).relative_to(ROOT).as_posix() if Path(cand).is_absolute() else cand,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(key, "->", dst)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "prepare":
        prepare(args or list(SPECS))
    elif cmd == "composite":
        composite(args[0], args[1:])
    elif cmd == "pick":
        pick(args[0], args[1])
