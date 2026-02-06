# ComfyUI 워크플로우

> 노드 기반 이미지 생성

## 개요

[IP-Adapter](./04-ip-adapter.md)까지 배운 모든 기술 — [LoRA](./01-lora.md), [DreamBooth](./02-dreambooth.md), [ControlNet](./03-controlnet.md), IP-Adapter — 을 **코드 없이 자유롭게 조합**할 수 있는 도구가 있습니다. 바로 **ComfyUI**입니다. 노드를 연결해서 워크플로우를 만드는 **비주얼 프로그래밍** 방식으로, 복잡한 이미지 생성 파이프라인을 직관적으로 설계할 수 있습니다.

**선수 지식**: [SD 아키텍처](../13-stable-diffusion/01-sd-architecture.md), [LoRA](./01-lora.md), [ControlNet](./03-controlnet.md)
**학습 목표**:
- ComfyUI의 노드 기반 인터페이스를 이해한다
- 기본 텍스트-투-이미지 워크플로우를 구성할 수 있다
- LoRA, ControlNet, IP-Adapter를 워크플로우에 통합할 수 있다
- 워크플로우를 저장하고 공유할 수 있다

## 왜 알아야 할까?

Python 코드로 파이프라인을 짜는 건 유연하지만, 매번 코드를 수정하고 실행하는 건 번거롭습니다. ComfyUI는 노드를 **드래그 앤 드롭**으로 연결해서 실시간으로 결과를 확인할 수 있어요. 또한 2025년 기준으로 **AI 비디오 생성**의 사실상 표준 도구가 되었고, SDXL, [FLUX](../13-stable-diffusion/05-flux.md), [SD3](../13-stable-diffusion/06-sd3-future.md) 같은 최신 모델도 가장 빠르게 지원합니다.

## 핵심 개념

### 개념 1: 노드 기반 인터페이스

> 💡 **비유**: ComfyUI는 **레고 블록**과 같습니다. 각 블록(노드)이 특정 기능을 담당하고, 블록들을 연결하면 원하는 결과물(이미지)이 나옵니다. 블록 순서를 바꾸거나 새 블록을 추가하면 완전히 다른 결과가 나오죠.

**ComfyUI의 기본 구성 요소:**

| 요소 | 역할 | 예시 |
|------|------|------|
| **노드(Node)** | 특정 기능을 수행하는 블록 | Load Checkpoint, CLIP Text Encode |
| **엣지(Edge)** | 노드 간 데이터 연결선 | 모델 → 샘플러 |
| **입력(Input)** | 노드의 왼쪽 연결점 | 모델, 조건, 잠재 이미지 |
| **출력(Output)** | 노드의 오른쪽 연결점 | 처리된 데이터 |
| **위젯(Widget)** | 노드 내 설정값 | 시드, 스텝 수, CFG 스케일 |

**기본 텍스트-투-이미지 워크플로우:**

> **Load Checkpoint** → **CLIP Text Encode (Positive)** ─┐
>                                                        ├→ **KSampler** → **VAE Decode** → **Save Image**
> **CLIP Text Encode (Negative)** ─────────────────────┘
>                                                        ↑
> **Empty Latent Image** ────────────────────────────────┘

### 개념 2: ComfyUI vs AUTOMATIC1111

두 도구 모두 Stable Diffusion UI지만, 철학이 다릅니다.

| 비교 항목 | ComfyUI | AUTOMATIC1111 |
|-----------|---------|---------------|
| **인터페이스** | 노드 기반 (플로우차트) | 웹 폼 기반 (슬라이더/버튼) |
| **학습 곡선** | **가파름** (초기 진입 어려움) | 완만함 (즉시 사용 가능) |
| **유연성** | **매우 높음** (노드 수준 제어) | 중간 (확장 스크립트로 추가) |
| **메모리 효율** | **우수** (동적 로딩/언로딩) | 보통 |
| **속도** | **10~20% 빠름** | 기준 |
| **비디오 생성** | **표준 도구** | 제한적 지원 |
| **최신 모델 지원** | **빠름** | 약간 느림 |
| **커뮤니티** | 빠르게 성장 중 | 대규모, 성숙함 |

> 🔥 **실무 팁**: 간단한 이미지 생성은 AUTOMATIC1111로, **복잡한 워크플로우**나 **비디오 생성**은 ComfyUI로 하는 조합이 효과적입니다.

### 개념 3: 핵심 노드 이해하기

**모델 로딩 노드:**

| 노드 | 기능 | 출력 |
|------|------|------|
| **Load Checkpoint** | SD 모델 로드 | MODEL, CLIP, VAE |
| **Load LoRA** | LoRA 가중치 적용 | MODEL, CLIP (수정됨) |
| **Load ControlNet** | ControlNet 모델 로드 | CONTROLNET |
| **Load VAE** | 별도 VAE 로드 | VAE |

**텍스트 인코딩 노드:**

| 노드 | 기능 | 입력 | 출력 |
|------|------|------|------|
| **CLIP Text Encode** | 프롬프트 → 조건 | CLIP, 텍스트 | CONDITIONING |
| **CLIP Text Encode (SDXL)** | SDXL용 인코딩 | CLIP, 텍스트, 해상도 | CONDITIONING |

**샘플링 노드:**

| 노드 | 기능 | 주요 설정 |
|------|------|----------|
| **KSampler** | 메인 이미지 생성 | 시드, 스텝, CFG, 샘플러, 스케줄러 |
| **KSampler Advanced** | 고급 설정 포함 | + 시작/종료 스텝 제어 |

**이미지 처리 노드:**

| 노드 | 기능 |
|------|------|
| **VAE Decode** | 잠재 → 이미지 |
| **VAE Encode** | 이미지 → 잠재 |
| **Save Image** | 이미지 저장 |
| **Preview Image** | 미리보기 (저장 안 함) |

> ⚠️ **흔한 오해**: "노드가 너무 많아서 복잡하다" — 처음엔 그렇게 느껴지지만, 대부분의 워크플로우는 **10개 미만의 노드**로 구성됩니다. 기본 패턴을 익히면 금방 익숙해져요.

### 개념 4: LoRA, ControlNet, IP-Adapter 통합

**LoRA 추가:**

> Load Checkpoint → **Load LoRA** → CLIP Text Encode → ...

```
Load LoRA 노드 설정:
- lora_name: "my_style_lora.safetensors"
- strength_model: 0.8  (모델에 적용할 강도)
- strength_clip: 0.8   (CLIP에 적용할 강도)
```

**ControlNet 추가:**

> Load Checkpoint → CLIP Text Encode ─┐
>                                      ├→ Apply ControlNet → KSampler → ...
> Load ControlNet ─┐                   │
>                  ├→ **Apply ControlNet**
> 조건 이미지 ─────┘

**IP-Adapter 추가:**

> Load Checkpoint → CLIP Text Encode ─┐
>                                      ├→ **Apply IPAdapter** → KSampler → ...
> Load IPAdapter ─┐                    │
>                 ├→ **Apply IPAdapter**
> 참조 이미지 ────┘

### 개념 5: 커스텀 노드 생태계

ComfyUI의 강점 중 하나는 **커스텀 노드** 생태계입니다. 커뮤니티가 만든 수천 개의 노드를 설치해서 기능을 확장할 수 있습니다.

**인기 있는 커스텀 노드 팩:**

| 노드 팩 | 기능 | 필수도 |
|---------|------|--------|
| **ComfyUI-Manager** | 노드 설치/관리 | ⭐⭐⭐ 필수 |
| **ComfyUI-Impact-Pack** | 얼굴 검출, 업스케일 | ⭐⭐⭐ |
| **ComfyUI-ControlNet-aux** | ControlNet 전처리기 | ⭐⭐⭐ |
| **ComfyUI-IPAdapter-plus** | IP-Adapter 확장 | ⭐⭐⭐ |
| **ComfyUI-AnimateDiff** | 비디오 생성 | ⭐⭐ |
| **ComfyUI-KJNodes** | 유틸리티 노드 모음 | ⭐⭐ |

> 💡 **알고 계셨나요?** ComfyUI-Manager 하나만 설치하면, GUI에서 다른 모든 커스텀 노드를 검색하고 설치할 수 있습니다. 마치 VS Code의 확장 마켓플레이스 같은 역할이에요.

## 실습: ComfyUI 워크플로우 구성

### 설치 및 실행

```bash
# 1. ComfyUI 클론
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 모델 배치
# models/checkpoints/ 에 SD 모델 (.safetensors) 배치
# models/loras/ 에 LoRA 파일 배치
# models/controlnet/ 에 ControlNet 모델 배치

# 4. 실행
python main.py

# 브라우저에서 http://127.0.0.1:8188 접속
```

### 기본 워크플로우 (텍스트 → 이미지)

ComfyUI 실행 후 기본 워크플로우가 로드됩니다. 주요 설정:

**Load Checkpoint 노드:**
- ckpt_name: 사용할 모델 선택 (예: sd_v1-5.safetensors)

**CLIP Text Encode (Positive):**
- text: "a beautiful sunset over the ocean, masterpiece, best quality"

**CLIP Text Encode (Negative):**
- text: "low quality, blurry, bad anatomy"

**Empty Latent Image:**
- width: 512 (SD 1.5) 또는 1024 (SDXL)
- height: 512 또는 1024
- batch_size: 1

**KSampler:**
- seed: 원하는 값 또는 랜덤
- steps: 20~30
- cfg: 7.0~7.5
- sampler_name: euler_ancestral
- scheduler: normal
- denoise: 1.0

**Queue Prompt** 버튼 클릭으로 생성 시작!

### LoRA 추가 워크플로우

```
워크플로우 구성:

1. [Load Checkpoint]
   → ckpt_name: "sd_v1-5.safetensors"
   → 출력: MODEL, CLIP, VAE

2. [Load LoRA]
   ← MODEL, CLIP 연결
   → lora_name: "anime_style.safetensors"
   → strength_model: 0.8
   → strength_clip: 0.8
   → 출력: MODEL, CLIP (LoRA 적용됨)

3. [CLIP Text Encode] × 2
   ← CLIP (LoRA 적용된 것) 연결
   → Positive: "1girl, smile, anime style"
   → Negative: "low quality"

4. [Empty Latent Image] → [KSampler] → [VAE Decode] → [Save Image]
```

### ControlNet + LoRA 조합 워크플로우

```
복합 워크플로우 구성:

1. [Load Checkpoint] → [Load LoRA] → MODEL, CLIP

2. [Load ControlNet Model]
   → control_net_name: "control_canny.safetensors"
   → 출력: CONTROLNET

3. [Load Image] → [Canny Edge Detection]
   → 출력: 에지 이미지

4. [Apply ControlNet]
   ← CONDITIONING (positive), CONTROLNET, 에지 이미지
   → strength: 1.0
   → 출력: CONDITIONING (ControlNet 적용됨)

5. [KSampler]
   ← MODEL, CONDITIONING (ControlNet), NEGATIVE, LATENT
   → 일반 설정

6. [VAE Decode] → [Save Image]
```

### 워크플로우 저장 및 공유

```
저장 방법:
1. 메뉴 > Save (JSON 파일로 저장)
2. 메뉴 > Save (API Format) - 프로그래밍 용

공유 방법:
1. JSON 파일 직접 공유
2. 이미지에 워크플로우 임베딩:
   - Save Image 노드에서 이미지 저장 시 자동 임베딩
   - 받은 사람이 이미지를 ComfyUI에 드래그하면 워크플로우 복원!

3. OpenArt, Civitai 등에 업로드
```

## 더 깊이 알아보기

### ComfyUI의 메모리 관리

ComfyUI가 AUTOMATIC1111보다 효율적인 이유 중 하나는 **동적 메모리 관리**입니다:

- 사용하지 않는 모델을 **자동으로 언로드**
- 필요할 때만 **GPU에 로드**
- 8GB VRAM에서도 SDXL 실행 가능

이런 최적화 덕분에 복잡한 워크플로우도 일반 소비자 GPU에서 실행할 수 있습니다.

### API 모드로 프로덕션 배포

ComfyUI는 API 서버로도 사용할 수 있습니다:

```python
# ComfyUI API 호출 예시
import json
import requests

# 워크플로우 JSON 로드
with open("workflow.json", "r") as f:
    workflow = json.load(f)

# 프롬프트 수정
workflow["3"]["inputs"]["text"] = "a cat in space"

# API 호출
response = requests.post(
    "http://127.0.0.1:8188/prompt",
    json={"prompt": workflow}
)

print(response.json())
```

이 방식으로 ComfyUI 워크플로우를 **프로덕션 파이프라인**에 통합할 수 있습니다.

### ComfyUI의 탄생

ComfyUI는 **comfyanonymous**라는 개발자가 2023년 초에 시작한 프로젝트입니다. AUTOMATIC1111이 이미 인기였지만, 그는 "더 **유연하고 효율적인** 도구가 필요하다"고 생각했죠. 노드 기반 접근법은 Blender, Nuke, Houdini 같은 3D/VFX 소프트웨어에서 영감을 받았습니다. 2024년에는 Stability AI가 ComfyUI 팀을 후원하기 시작하면서 개발이 더욱 가속화되었습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "ComfyUI는 코딩을 모르면 못 쓴다" — 전혀 아닙니다! 노드를 연결하는 것은 코딩이 아닌 **시각적 구성**입니다. 오히려 코드보다 직관적일 수 있어요.

> 🔥 **실무 팁**: 좋은 워크플로우를 처음부터 만들 필요 없습니다. **OpenArt**, **Civitai**, **ComfyUI Workflows** 같은 사이트에서 다른 사람이 만든 워크플로우를 다운받아 시작하세요.

> 💡 **알고 계셨나요?** ComfyUI로 저장한 이미지에는 **워크플로우가 임베딩**되어 있습니다. 다른 사람의 이미지를 ComfyUI에 드래그하면 그 이미지를 만든 정확한 설정을 복원할 수 있어요!

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **노드 기반 UI** | 블록을 연결해서 워크플로우 구성 |
| **핵심 노드** | Load Checkpoint, CLIP Text Encode, KSampler, VAE Decode |
| **커스텀 노드** | ComfyUI-Manager로 설치, 기능 무한 확장 |
| **메모리 효율** | 동적 로딩/언로딩으로 낮은 VRAM에서도 실행 |
| **워크플로우 공유** | JSON 저장 또는 이미지에 임베딩 |
| **vs A1111** | 유연성과 효율성 ↑, 학습 곡선 ↑ |

## 다음 섹션 미리보기

다음 [인페인팅과 아웃페인팅](./06-inpainting-outpainting.md)에서는 이미지의 **일부만 수정**하거나 **영역을 확장**하는 기법을 배웁니다. 기존 이미지에서 원하지 않는 요소를 제거하거나, 이미지 밖의 영역을 상상으로 채우는 기술이에요.

## 참고 자료

- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI) - 공식 저장소
- [Beginner's Guide to ComfyUI - Stable Diffusion Art](https://stable-diffusion-art.com/comfyui/) - 종합 초보자 가이드
- [ComfyUI Official Documentation](https://docs.comfy.org/) - 공식 문서
- [ComfyUI: Beginner to Advance Guide](https://www.stablediffusiontutorials.com/2024/04/comfyui-tutorial.html) - 단계별 튜토리얼
- [A1111 vs ComfyUI - Modal](https://modal.com/blog/a1111-vs-comfyui) - 상세 비교 분석
- [OpenArt Workflows](https://openart.ai/workflows) - 워크플로우 공유 사이트
- [ComfyUI Community Wiki](https://comfyui-wiki.com/) - 커뮤니티 위키
