# 거짓말 헤드밴드

로컬 LLM과 1:1로 하는 추리 대화 게임입니다. 헤드밴드 게임에 스무고개와 거짓말 간파를 섞었습니다.

- 서로 **상대의 제시어만 보면서** 예/아니오 질문을 주고받고, 먼저 자기 제시어를 맞히는 쪽이 이깁니다.
- **양쪽 모두 거짓말을 할 수 있습니다.** AI 캐릭터는 거짓말할 때 고유한 버릇(텔)을 아주 짧게 보입니다. 이 버릇을 알아채면 **간파**해서 손해를 되돌릴 수 있습니다.
- AI는 플레이어의 답을 항상 의심합니다. 너무 오래 망설이거나 앞뒤가 안 맞으면 거꾸로 간파당합니다.
- 질문 토큰(15개)과 정답 도전 기회(3번)가 한정되어 있어서, 속으면 손해를 보고 간파하면 만회합니다.

자세한 규칙과 설계는 [`docs/game-design.md`](docs/game-design.md), 화면 디자인 기준은 [`docs/ui-layout.html`](docs/ui-layout.html)에 있습니다.

## 구성

| 항목 | 내용 |
|---|---|
| 화면 | Python + Gradio 웹 (`ui/app.py`, `ui/style.css`) |
| AI | 로컬 LLM: [Ollama](https://ollama.com) + `qwen3:14b` |
| 아트 | ComfyUI로 생성한 오리지널 캐릭터 스프라이트 (`assets/`) |

```
LLM_Project/
├─ docs/               # 기획서, 화면 디자인
├─ game/               # 게임 로직 (UI를 모름)
│  ├─ engine.py        #   턴 진행, 토큰, 정답 도전, 승패
│  ├─ lie_system.py    #   AI 거짓말 결정, 텔, 난이도
│  ├─ suspicion.py     #   AI의 의심도, 간파 결정
│  ├─ roles.py         #   LLM 역할별 프롬프트 (답 판정, 심판, 반응 대사, AI 질문)
│  ├─ llm.py           #   Ollama 호출 (JSON 출력 강제)
│  └─ words.py         #   제시어와 카테고리
├─ ui/                 # Gradio 화면
├─ assets/             # 배경, 캐릭터 스프라이트 (characters/<ID>/meta.json 에 설정)
├─ tools/              # 스프라이트 생성(ComfyUI), LLM 점검 스크립트
└─ tests/              # pytest
```

## 요구 사항

- **Python 3.12 이상** (개발 환경: 3.14)
- **Ollama**와 `qwen3:14b` 모델 (약 9.3GB). GPU VRAM 12GB 정도면 충분합니다. GPU가 없어도 돌아가지만 답변이 느립니다.

## 가상환경 구축

아래는 Windows PowerShell 기준입니다. macOS/Linux는 괄호 안 명령을 쓰세요.

```powershell
# 1. 가상환경 만들기
python -m venv .venv

# 2. 활성화  (macOS/Linux: source .venv/bin/activate)
.venv\Scripts\Activate.ps1

# 3. 의존성 설치
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> PowerShell에서 "스크립트를 실행할 수 없습니다" 오류가 나면 한 번만 실행하세요:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> 활성화하지 않고 `.venv\Scripts\python.exe`로 바로 실행해도 됩니다.

스프라이트 생성 도구(`tools/gen_sprites.py`)까지 쓰려면 Pillow도 설치합니다.

```powershell
pip install pillow
```

## LLM 준비 (Ollama)

1. [ollama.com](https://ollama.com)에서 Ollama를 설치하고 실행합니다. 기본 주소는 `http://localhost:11434`입니다.
2. 모델을 받습니다.

```powershell
ollama pull qwen3:14b
```

3. (선택) 연결과 응답 품질을 확인합니다.

```powershell
python tools\smoke_test.py
```

## 실행

```powershell
python ui\app.py
```

브라우저에서 `http://127.0.0.1:7860`을 엽니다. 포트를 바꾸려면 실행 전에 환경 변수를 설정합니다.

```powershell
$env:GRADIO_SERVER_PORT = "7861"
python ui\app.py
```

### 상대 캐릭터 바꾸기

상대 캐릭터는 `assets/characters/<ID>/` 폴더 단위로 들어 있고, 환경 변수 `HB_CHARACTER`로 고릅니다 (기본값 `robot`).

```powershell
$env:HB_CHARACTER = "robot"
python ui\app.py
```

| ID | 캐릭터 | 상태 |
|---|---|---|
| `robot` | 작은 로봇 (버릇: 안테나 스파크) | 사용 중 |
| `honest` | 도하 (정직한 타입, 버릇: 귀 움찔) | 제작 중 |
| `sly` | 루미 | 화면 개발용 임시 캐릭터 |

캐릭터 이름, 무대 배치, 헤드밴드 카드 위치, 표정 연결은 각 폴더의 `meta.json`에 있습니다.

### 플레이 방법

1. **내 차례**: 내 제시어(나는 볼 수 없음)를 맞히기 위해 예/아니오로 답할 수 있는 질문을 입력하고 **[질문]**을 누릅니다.
2. 상대의 답이 거짓말 같으면, 다음 질문을 하기 전에 **[간파]**를 누릅니다. 성공하면 진짜 답이 공개되고 토큰을 돌려받으며, 실패하면 토큰 1개를 잃습니다.
3. **상대 차례**: 상대가 질문하면 **[예] [아니오] [애매함]**으로 답합니다. 거짓말하려면 **[거짓말로 답하기]**를 켜고 사실과 반대로 답합니다. 첫 2개 답은 거짓말할 수 없습니다.
4. 제시어를 알 것 같으면 단어를 입력하고 **[정답]**을 누릅니다 (3번까지, 틀리면 토큰 1개를 잃습니다).

난이도(쉬움/보통/어려움)는 오른쪽 아래 **설정 · 규칙**에서 바꿉니다. 어려울수록 상대의 버릇이 짧고 작아지고, 서로 쓸 수 있는 거짓말 횟수가 늘어납니다.

## 테스트

```powershell
python -m pytest -q
```

테스트는 가짜 LLM을 써서 Ollama 없이도 돌아갑니다. 실제 LLM으로 점검하려면 아래 스크립트를 씁니다.

```powershell
python tools\smoke_test.py               # LLM 역할별 품질과 속도
python tools\judge_bench.py --repeat 3   # 답 판정(심판) 정확도 벤치마크
```

## 스프라이트 다시 만들기 (선택)

캐릭터 에셋은 이미 `assets/`에 들어 있어서 게임 실행에는 필요 없습니다. 새로 만들거나 고칠 때만 로컬 ComfyUI(`127.0.0.1:8188`, Z-Image-Turbo 모델)를 띄우고 `tools/gen_sprites.py`를 씁니다. 생성 기록(프롬프트, 시드, 마스크)은 각 캐릭터 폴더의 `meta.json`과 `expressions.json`에 남아 있습니다.

```powershell
python tools\gen_sprites.py robot-expr              # 로봇 표정 후보 생성
python tools\gen_sprites.py robot-parts             # 말하기 / 버릇 조각 후보 생성
```

ComfyUI와 Ollama를 동시에 돌리면 VRAM이 부족할 수 있습니다. 스프라이트를 생성하는 동안에는 Ollama 모델을 내려 두세요.
