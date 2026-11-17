"""도하(정식 남자 캐릭터) 기본 이미지 생성용 SDXL 워크플로 준비.

모델: Animagine XL 4.0 opt (tools/workflows/sdxl_t2i.json, ComfyUI checkpoints 폴더)
Animagine 은 태그형 프롬프트로 학습돼서 태그 위주로 쓰고, 품질 태그는 맨 뒤에 둔다.

사용법:
  python tools/doha_prompt.py <seed> <출력 접두어>  →  tools/workflows/doha_sdxl_run.json 생성
  그 다음 ComfyUI(MCP run_workflow)로 실행
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "tools" / "workflows" / "sdxl_t2i.json"
OUT = ROOT / "tools" / "workflows" / "doha_sdxl_run.json"

# v2: v1 에서 머리가 길어 귀를 가리고, 헤드밴드가 얇고 기울고, 머리 위가 잘렸다 → 아래 태그로 보정
# v3: v2 에서 "ears" 를 동물 귀로 그린 경우, 여전히 머리 위가 잘림, 헤드밴드가 주황빛 → 사람 귀 / 허리 위 구도 / 다크 레드
POSITIVE = (
    "1boy, solo, male focus, bishounen, handsome, beautiful man, young adult, "
    "cowboy shot, waist up, head fully in frame, hair fully in frame, space above head, centered, "
    "looking at viewer, facing viewer, straight-on, symmetrical, head straight, "
    "gentle smile, closed mouth, kind eyes, almond eyes, long eyelashes, brown eyes, sharp jawline, "
    # "chestnut" 는 밤(열매)을 그려 넣는 경우가 있어서 쓰지 않는다
    "very short hair, undercut, short sides, brown hair, glossy hair, "
    "human ears, both ears visible, exposed ears, "
    "dark red headband, maroon headband, thick headband, wide headband, horizontal headband, headband across forehead, "
    "white collared shirt, navy blue cardigan, "
    "simple background, light grey background, visual novel sprite, anime coloring, clean lineart, "
    "masterpiece, high score, great score, absurdres"
)

NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digits, fewer digits, cropped, "
    "worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry, "
    "long hair, medium hair, sidelocks, hair over ears, covered ears, hair covering ears, "
    "thin headband, tilted headband, diagonal headband, hairband, "
    "cropped head, head out of frame, close-up, portrait, "
    "animal ears, kemonomimi, dog ears, fox ears, orange headband, "
    "hat, cap, hood, headphones, earrings, glasses, "
    "head tilt, looking away, profile, hands, multiple boys, female, breasts"
)


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    prefix = sys.argv[2] if len(sys.argv) > 2 else "doha/sdxl"
    wf = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    wf["6"]["inputs"]["text"] = POSITIVE
    wf["7"]["inputs"]["text"] = NEGATIVE
    wf["3"]["inputs"]["seed"] = seed
    wf["9"]["inputs"]["filename_prefix"] = prefix
    OUT.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
