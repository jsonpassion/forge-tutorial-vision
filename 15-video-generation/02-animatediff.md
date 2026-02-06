# AnimateDiff

> 이미지 모델의 비디오 확장

## 개요

[비디오 Diffusion 기초](./01-video-diffusion.md)에서 이미지를 시간 축으로 확장하는 원리를 배웠습니다. 하지만 처음부터 비디오 모델을 훈련하려면 엄청난 계산 비용이 듭니다. **AnimateDiff**는 **기존 이미지 모델에 작은 모듈만 추가**해서 비디오를 생성할 수 있게 만드는 획기적인 방법입니다. [LoRA](../14-generative-practice/01-lora.md)가 모델 전체를 재훈련하지 않고 스타일을 추가했듯이, AnimateDiff는 **Motion Module**만 학습해서 움직임을 추가합니다.

**선수 지식**: [비디오 Diffusion 기초](./01-video-diffusion.md), [SD 아키텍처](../13-stable-diffusion/01-sd-architecture.md), [LoRA](../14-generative-practice/01-lora.md)
**학습 목표**:
- AnimateDiff의 Motion Module 구조를 이해한다
- Domain Adapter와 MotionLoRA의 역할을 파악한다
- SD 1.5 및 SDXL에서 AnimateDiff를 활용할 수 있다
- ComfyUI와 Diffusers로 비디오 애니메이션을 생성할 수 있다

## 왜 알아야 할까?

Stable Diffusion 커뮤니티에서 만들어진 수많은 체크포인트와 LoRA가 있습니다. 애니메이션 스타일, 사실적 인물, 판타지 배경 등 다양한 스타일이 있죠. **AnimateDiff를 사용하면 이 모든 이미지 모델을 그대로 활용하면서 비디오를 생성**할 수 있습니다. 별도의 비디오 모델을 훈련하거나 찾을 필요 없이, 좋아하는 체크포인트에 Motion Module만 끼워넣으면 되니까요.

## 핵심 개념

### 개념 1: Motion Module — 움직임의 비밀

> 💡 **비유**: AnimateDiff는 **인형에 줄을 달아 조종사가 움직이게 만드는 것**과 같습니다. 인형(이미지 모델) 자체는 그대로 두고, 줄과 조종 메커니즘(Motion Module)만 추가해서 살아 움직이게 만드는 거죠.

**Motion Module의 핵심 아이디어:**

기존 Stable Diffusion U-Net은 2D 이미지를 처리합니다. AnimateDiff는 각 U-Net 블록 **뒤에** Temporal Transformer를 삽입합니다:

> **기존 SD 블록**: Spatial Conv → Spatial Attention → Cross Attention
>
> **AnimateDiff 블록**: Spatial Conv → Spatial Attention → Cross Attention → **Temporal Attention (Motion Module)**

**Motion Module 구조:**

| 구성 요소 | 역할 |
|-----------|------|
| **Projection In** | 피처를 Temporal Attention 입력으로 변환 |
| **Temporal Self-Attention** | 프레임 간 관계 학습 |
| **Position Encoding** | 프레임 순서 정보 제공 |
| **Projection Out** | 원래 차원으로 복원 |
| **Zero Init** | 학습 초기 안정성 위해 0으로 초기화 |

**Zero Initialization의 비밀:**

[ControlNet](../14-generative-practice/03-controlnet.md)에서 배운 Zero Convolution을 기억하시나요? AnimateDiff도 비슷한 트릭을 씁니다. Motion Module 출력을 **0으로 초기화**해서, 학습 초기에는 원래 이미지 모델처럼 동작하다가 점점 움직임을 학습합니다.

### 개념 2: 3단계 훈련 파이프라인

AnimateDiff는 세 단계로 훈련됩니다:

**Stage 1: Domain Adapter 훈련**

> 💡 **비유**: 영화 제작 전에 **배우가 작품 분위기에 적응하는 리허설** 단계입니다.

비디오 데이터셋과 이미지 생성 모델 사이의 **도메인 갭**을 줄이기 위해, 먼저 Domain Adapter를 훈련합니다. 비디오 데이터의 시각적 아티팩트(압축 노이즈, 블러 등)에 모델이 적응하게 만드는 거죠.

**Stage 2: Motion Module 훈련**

핵심 단계입니다. WebVid-10M 등의 비디오 데이터셋으로 **전이 가능한 움직임 패턴**을 학습합니다:

- 카메라 움직임 (팬, 틸트, 줌)
- 객체 움직임 (걷기, 날기, 흔들림)
- 자연 현상 (물결, 바람, 불꽃)

**Stage 3: MotionLoRA (선택)**

특정 움직임 패턴(예: 줌인, 롤링 카메라)을 추가로 학습하려면 **MotionLoRA**를 사용합니다. [LoRA](../14-generative-practice/01-lora.md)처럼 작은 파라미터만 학습해서 새로운 모션을 추가할 수 있죠.

### 개념 3: SD 1.5 vs SDXL Motion Module

**SD 1.5 Motion Module:**

- 해상도: 256×256 ~ 512×512
- 프레임: 16프레임 기준 훈련
- 파일: `mm_sd_v15_v2.ckpt` (약 1.5GB)
- 장점: 빠른 생성, 낮은 VRAM 요구

**SDXL Motion Module (Beta):**

- 해상도: 512×512 ~ 1024×1024
- 프레임: 16~32프레임
- 파일: `mm_sdxl_v10_beta.ckpt`
- 장점: 고해상도, 더 디테일한 움직임
- 단점: VRAM 16GB+ 필요

**호환성 표:**

| 체크포인트 | Motion Module | 호환 |
|------------|---------------|------|
| SD 1.5 계열 | mm_sd_v15_v2 | ✅ |
| SD 2.1 | mm_sd_v15_v2 | ⚠️ 부분적 |
| SDXL | mm_sdxl_v10_beta | ✅ |
| SD 1.5 + **SDXL Motion** | - | ❌ 불가 |

### 개념 4: AnimateDiff + ControlNet 조합

AnimateDiff의 강력한 점은 **기존 SD 생태계와 완전 호환**된다는 것입니다. ControlNet, LoRA, IP-Adapter 모두 함께 사용 가능합니다.

**ControlNet 조합 예시:**

> 입력 → OpenPose 시퀀스 (춤추는 사람) + AnimateDiff
> 출력 → 해당 동작을 따라하는 애니메이션 캐릭터

**조합 가능한 조건:**

| 컨트롤러 | 용도 | 비디오 적용 |
|----------|------|-------------|
| **OpenPose** | 인체 동작 | 각 프레임에 포즈 적용 |
| **Depth** | 카메라 움직임 | 깊이맵 시퀀스 |
| **Canny** | 선화 애니메이션 | 윤곽선 시퀀스 |
| **LoRA** | 스타일/캐릭터 | 그대로 적용 |
| **IP-Adapter** | 참조 이미지 | 첫 프레임 또는 전체 |

## 실습: AnimateDiff로 비디오 생성

### Diffusers로 기본 사용

```python
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

# 1. Motion Adapter 로드 (AnimateDiff 핵심)
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16
)

# 2. SD 1.5 체크포인트와 결합
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"  # 원하는 체크포인트
pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    torch_dtype=torch.float16
)

# 3. 스케줄러 설정
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config,
    beta_schedule="linear",
    steps_offset=1,
)

pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

# 4. 비디오 생성
prompt = "a beautiful woman walking in a flower garden, wind blowing hair, \
         high quality, detailed, cinematic lighting"
negative_prompt = "bad quality, blur, distorted"

# 16프레임 GIF 생성
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    height=512,
    width=512,
)

frames = output.frames[0]
export_to_gif(frames, "walking_woman.gif")
print("✅ GIF 저장 완료: walking_woman.gif")
```

### LoRA와 함께 사용

```python
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

# Motion Adapter 로드
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16
)

# 기본 모델 로드
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16
)

# 스타일 LoRA 로드 (예: 애니메이션 스타일)
pipe.load_lora_weights(
    "path/to/anime_style_lora",
    adapter_name="anime_style"
)
pipe.set_adapters(["anime_style"], adapter_weights=[0.8])

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# 애니메이션 스타일 비디오 생성
prompt = "1girl, long blue hair, magical girl transformation, \
         sparkles, dynamic pose, anime style"

output = pipe(
    prompt=prompt,
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
)

export_to_gif(output.frames[0], "anime_transformation.gif")
```

### MotionLoRA로 특정 모션 적용

```python
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

# Motion Adapter 로드
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16
)

pipe = AnimateDiffPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=adapter,
    torch_dtype=torch.float16
)

# MotionLoRA 로드 (예: 줌인 효과)
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-in",
    adapter_name="zoom_in"
)
# MotionLoRA 강도 조절
pipe.set_adapters(["zoom_in"], adapter_weights=[0.75])

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# 줌인 효과가 적용된 비디오
prompt = "a mystical forest with glowing mushrooms, foggy atmosphere"

output = pipe(
    prompt=prompt,
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
)

export_to_gif(output.frames[0], "forest_zoom_in.gif")
print("✅ 줌인 효과 비디오 생성 완료!")
```

### SDXL AnimateDiff

```python
import torch
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

# SDXL용 Motion Adapter
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-sdxl-beta",
    torch_dtype=torch.float16
)

# SDXL 파이프라인
pipe = AnimateDiffSDXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    motion_adapter=adapter,
    torch_dtype=torch.float16,
    variant="fp16",
)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# 고해상도 비디오 생성 (VRAM 16GB+ 필요)
prompt = "a majestic eagle soaring over snowy mountains, \
         golden hour lighting, cinematic, 4k quality"

output = pipe(
    prompt=prompt,
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=20,
    height=768,
    width=1024,
)

export_to_gif(output.frames[0], "eagle_flying_sdxl.gif")
print("✅ SDXL AnimateDiff 비디오 생성 완료!")
```

## 더 깊이 알아보기: AnimateDiff의 탄생

**2023년 — 상하이 AI Lab의 돌파구**

AnimateDiff는 2023년 상하이 AI Lab의 연구팀이 발표했습니다. 논문 제목 "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning"이 말해주듯, 핵심은 **기존 모델을 건드리지 않고 애니메이션 기능을 추가**하는 것이었죠.

당시 커뮤니티에는 수천 개의 SD 1.5 체크포인트가 있었지만, 비디오 모델은 거의 없었습니다. AnimateDiff는 이 문제를 "**Motion Module만 훈련하면 모든 이미지 모델이 비디오 모델이 된다**"는 우아한 해결책으로 풀었습니다.

**폭발적 인기의 비결:**

1. **즉시 사용 가능**: 기존 체크포인트, LoRA와 호환
2. **낮은 VRAM**: SD 1.5 기준 8GB GPU에서도 동작
3. **커뮤니티 지원**: ComfyUI, A1111 확장 빠르게 개발
4. **MotionLoRA**: 특정 모션 패턴 쉽게 추가 가능

**후속 발전:**

- **AnimateDiff v2**: 더 긴 영상 (32프레임+)
- **AnimateDiff SDXL Beta**: 고해상도 지원
- **SparseCtrl**: 키프레임 기반 제어
- **AnimateDiff-Lightning**: 4스텝 빠른 생성

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "AnimateDiff로 무한히 긴 영상을 만들 수 있다"
>
> 기본적으로 16~32프레임(약 1~2초)이 한계입니다. 더 긴 영상은 **영상 연장(Video Continuation)** 기법이나 **프레임 보간(Interpolation)**을 별도로 적용해야 합니다.

> 💡 **알고 계셨나요?**: AnimateDiff의 Motion Module은 **WebVid-10M** 데이터셋으로 훈련되었는데, 이 데이터셋은 유튜브 스톡 비디오 1천만 개를 수집한 것입니다. 그래서 자연스러운 카메라 움직임과 일반적인 동작은 잘 표현하지만, 특수한 애니메이션 동작은 MotionLoRA로 추가 학습이 필요합니다.

> 🔥 **실무 팁**: AnimateDiff 결과물의 **프레임 보간**을 적용하면 품질이 크게 향상됩니다. RIFE, FILM 등의 보간 모델로 16fps → 60fps로 올리면 훨씬 부드러운 영상이 됩니다.

> 🔥 **실무 팁**: 프롬프트에 **움직임 키워드**를 명시적으로 넣으세요. "walking", "running", "waving", "wind blowing" 등 동작을 설명하는 단어가 있어야 Motion Module이 적절한 움직임을 생성합니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **Motion Module** | 기존 SD 블록에 추가되는 Temporal Transformer |
| **Zero Init** | 학습 초기 안정성을 위한 0 초기화 |
| **Domain Adapter** | 비디오 데이터 도메인에 적응시키는 모듈 |
| **MotionLoRA** | 특정 모션 패턴을 학습한 경량 어댑터 |
| **호환성** | 기존 SD 체크포인트, LoRA, ControlNet과 호환 |
| **해상도** | SD1.5: ~512px, SDXL: ~1024px |

## 다음 섹션 미리보기

AnimateDiff는 텍스트나 기존 체크포인트로 비디오를 생성했습니다. 다음 섹션 [Stable Video Diffusion](./03-svd.md)에서는 **이미지 한 장을 입력으로 받아 비디오를 생성하는 Image-to-Video 모델**을 배웁니다. 이미 완성된 이미지에 생명을 불어넣는 기술이죠!

## 참고 자료

- [AnimateDiff GitHub](https://github.com/guoyww/AnimateDiff) - 공식 구현체
- [AnimateDiff 논문 (arXiv)](https://arxiv.org/abs/2307.04725) - 원본 논문
- [Diffusers AnimateDiff 가이드](https://huggingface.co/docs/diffusers/api/pipelines/animatediff) - HuggingFace 공식 문서
- [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) - ComfyUI 확장
- [Stable Diffusion Art - AnimateDiff](https://stable-diffusion-art.com/animatediff/) - 실습 튜토리얼
