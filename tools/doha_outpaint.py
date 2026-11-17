"""도하 후보의 잘린 머리 위를 채우는 아웃페인팅 (SDXL, Animagine).

Animagine 은 정면 상반신을 너무 가깝게 잡아서 머리 위가 잘린다.
얼굴·귀·헤드밴드가 좋은 후보를 줄여서 캔버스 아래쪽에 놓고, 빈 공간(위·양옆)만 다시 그린 뒤
원본 영역은 원래 픽셀로 덮는다 (경계는 부드럽게).

사용법:
  python tools/doha_outpaint.py prepare <후보 png> <이름>   → ComfyUI input 에 패딩 이미지, 워크플로 생성
  (ComfyUI 실행)
  python tools/doha_outpaint.py composite <이름> <결과 png...> → 원본을 다시 덮은 결과 저장
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doha_prompt import NEGATIVE, POSITIVE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMFY_INPUT = Path(r"C:\Users\user\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
WORK = ROOT / "assets" / "_candidates" / "doha" / "outpaint"
W, H = 832, 1216
FEATHER = 24  # 원본 경계를 이 픽셀만큼 부드럽게 섞는다

# 배치 방식 (이름별). scale: 원본 축소 비율, anchor: 캔버스 어디에 붙일지
#   1차(v2n3, v2n4): 0.82 아래 붙임 → 잘린 머리 위를 채움
#   2차(A2): 1차 결과 A 를 0.58 로 줄여 위에 붙임 → 아래 몸통과 양옆 어깨를 채움 (얼굴 크기를 임시 캐릭터와 맞춤)
LAYOUTS = {"bottom": 0.82, "top": 0.58}
LAYOUT = "bottom"


def placement() -> tuple[int, int, int, int]:
    scale = LAYOUTS[LAYOUT]
    w, h = round(W * scale), round(H * scale)
    y = H - h if LAYOUT == "bottom" else 0
    return (W - w) // 2, y, w, h  # 가로 가운데


def keep_mask() -> Image.Image:
    """원본을 유지할 영역(흰색). 캔버스 끝에 닿지 않은 경계만 부드럽게 섞는다."""
    x, y, w, h = placement()
    m = Image.new("L", (W, H), 0)
    m.paste(255, (x + FEATHER, y + FEATHER, x + w - FEATHER, y + h - FEATHER))
    m = m.filter(ImageFilter.GaussianBlur(FEATHER / 2))
    # 캔버스 끝에 닿은 변은 흐리지 않는다
    if y == 0:
        m.paste(255, (x + FEATHER, 0, x + w - FEATHER, FEATHER * 2))
    if y + h == H:
        m.paste(255, (x + FEATHER, H - FEATHER * 2, x + w - FEATHER, H))
    return m


def prepare(src: str, name: str) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    x, y, w, h = placement()
    img = Image.open(src).convert("RGB").resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (230, 230, 230, 0))
    canvas.paste(img.convert("RGBA"), (x, y))
    # LoadImage 의 MASK 는 알파 0 인 곳이 1 (= 다시 그릴 곳). 원본 영역은 알파 255.
    alpha = Image.new("L", (W, H), 0)
    alpha.paste(255, (x, y, x + w, y + h))
    canvas.putalpha(alpha)
    in_name = f"doha_pad_{name}.png"
    canvas.save(COMFY_INPUT / in_name)
    canvas.convert("RGB").save(WORK / f"{name}_pad_preview.png")

    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": in_name}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "animagine-xl-4.0-opt.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": POSITIVE, "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 1]}},
        "5": {"class_type": "VAEEncodeForInpaint",
              "inputs": {"pixels": ["1", 0], "vae": ["2", 2], "mask": ["1", 1], "grow_mask_by": 24}},
        "6": {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["5", 0], "amount": 3}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": 9601, "steps": 28, "cfg": 5.0, "sampler_name": "euler_ancestral", "scheduler": "normal",
            "denoise": 1.0, "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"doha/outpaint_{name}", "images": ["8", 0]}},
    }
    out = ROOT / "tools" / "workflows" / f"doha_outpaint_{name}.json"
    out.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


def composite(name: str, results: list[str]) -> None:
    pad = Image.open(COMFY_INPUT / f"doha_pad_{name}.png").convert("RGB")
    keep = keep_mask()
    for i, r in enumerate(results):
        gen = Image.open(r).convert("RGB").resize((W, H))
        out = Image.composite(pad, gen, keep)
        path = WORK / f"{name}_out_{i}.png"
        out.save(path)
        print(path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].startswith("--layout="):
        LAYOUT = args.pop(0).split("=", 1)[1]
    sys.argv = [sys.argv[0], *args]
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    elif cmd == "composite":
        composite(sys.argv[2], sys.argv[3:])
