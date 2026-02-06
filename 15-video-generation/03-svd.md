# Stable Video Diffusion

> 이미지-투-비디오 생성

## 개요

[AnimateDiff](./02-animatediff.md)에서 텍스트로부터 비디오를 생성하는 방법을 배웠습니다. 하지만 원하는 스타일과 구도를 텍스트만으로 정확히 표현하기는 어렵죠. **Stable Video Diffusion(SVD)**은 **이미지 한 장을 입력으로 받아** 그 이미지가 살아 움직이는 비디오를 생성합니다. 마치 해리 포터의 움직이는 사진처럼요!

**선수 지식**: [비디오 Diffusion 기초](./01-video-diffusion.md), [Latent Diffusion](../12-diffusion-models/06-latent-diffusion.md)
**학습 목표**:
- Image-to-Video 생성의 원리를 이해한다
- SVD의 아키텍처와 훈련 과정을 파악한다
- Motion Bucket과 FPS 파라미터를 활용할 수 있다
- Diffusers로 SVD 비디오를 생성할 수 있다

## 왜 알아야 할까?

이미지 생성 AI로 완벽한 한 장을 만들었다고 가정해 봅시다. 이제 이 이미지에 **자연스러운 움직임**을 추가하고 싶습니다. 머리카락이 바람에 날리고, 불꽃이 타오르고, 물결이 출렁이는 영상 말이죠. SVD는 이런 **Image-to-Video(I2V)** 작업의 표준 도구가 되었습니다. 광고, 소셜 미디어, 게임 시네마틱 등에서 정적인 이미지를 동적인 콘텐츠로 변환하는 데 널리 사용됩니다.

## 핵심 개념

### 개념 1: Image-to-Video — 조건부 비디오 생성

> 💡 **비유**: SVD는 **무성 영화에 배우를 캐스팅하는** 것과 같습니다. 배우(입력 이미지)가 정해지면, 감독(모델)이 그 배우가 어떻게 움직일지 연출합니다. 배우의 외모와 의상은 유지하면서 자연스러운 동작을 만들어내죠.

**I2V의 핵심 원리:**

1. 입력 이미지를 **첫 번째 프레임(조건)**으로 고정
2. 이후 프레임들을 노이즈에서 생성
3. 첫 프레임과 **시각적 일관성** 유지하면서 움직임 추가

**T2V(Text-to-Video) vs I2V 비교:**

| 구분 | Text-to-Video | Image-to-Video |
|------|---------------|----------------|
| **입력** | 텍스트 프롬프트 | 이미지 + (선택적 텍스트) |
| **제어력** | 낮음 (텍스트로 구도 표현 어려움) | 높음 (이미지가 시각적 기준) |
| **일관성** | 변동 큼 | 첫 프레임에 고정 |
| **활용** | 창의적 생성 | 기존 이미지 애니메이션 |

### 개념 2: SVD 아키텍처

> 💡 **비유**: SVD는 **SD의 뇌에 시간을 인식하는 영역을 추가한** 것과 같습니다. 공간만 보던 눈이 이제 시간의 흐름도 볼 수 있게 되었죠.

**아키텍처 구성:**

SVD는 Stable Diffusion의 U-Net을 확장한 **3D U-Net** 구조입니다:

> **기존 SD 블록**: 2D Conv → Spatial Attention → Cross Attention
>
> **SVD 블록**: 2D Conv → **Temporal Conv** → Spatial Attention → **Temporal Attention** → Cross Attention

**핵심 추가 요소:**

| 구성 요소 | 역할 |
|-----------|------|
| **Temporal Conv** | 시간 축 특징 혼합 (3×1×1 커널) |
| **Temporal Attention** | 프레임 간 어텐션 |
| **Image Conditioning** | 입력 이미지를 조건으로 주입 |
| **FPS Conditioning** | 프레임 속도 조건 (6~30 FPS) |
| **Motion Bucket** | 움직임 강도 조절 (1~255) |

**조건 주입 방식:**

입력 이미지는 여러 방식으로 조건에 반영됩니다:

1. **Latent 연결**: VAE로 인코딩된 이미지 latent를 첫 프레임에 직접 연결
2. **Cross Attention**: CLIP 이미지 인코더로 추출한 특징을 어텐션에 주입
3. **노이즈 마스킹**: 첫 프레임 영역은 노이즈 추가량 감소

### 개념 3: 훈련 파이프라인

Stability AI는 SVD를 **3단계**로 훈련했습니다:

**Stage 1: 이미지 사전 훈련**

대규모 이미지 데이터셋으로 Stable Diffusion을 훈련합니다. 이 단계에서 풍부한 시각적 표현을 학습하죠.

**Stage 2: 비디오 사전 훈련**

대규모 비디오 데이터셋(LVD-142M)으로 기본적인 시간적 일관성을 학습합니다:

- 1.4억 개 비디오 클립
- 580K 시간 분량
- **데이터 큐레이션**이 핵심 (품질 낮은 영상 필터링)

**Stage 3: 고품질 미세 조정**

소규모 고품질 데이터셋으로 미세 조정합니다:

- 높은 해상도
- 안정적인 카메라
- 깔끔한 동작

> 💡 **알고 계셨나요?**: SVD 훈련에는 **A100 80GB GPU 200,000시간**이 소요되었습니다. 단일 GPU로 계산하면 약 23년이 걸리는 분량이죠!

### 개념 4: Motion Bucket과 FPS — 움직임 제어

SVD의 강력한 기능 중 하나는 **움직임 강도와 속도를 조절**할 수 있다는 점입니다.

**Motion Bucket ID:**

움직임의 강도를 1~255 범위로 조절합니다:

| 값 범위 | 움직임 |
|---------|--------|
| **1~50** | 미세한 움직임 (바람에 흔들리는 정도) |
| **51~127** | 중간 움직임 (걷기, 고개 돌림) |
| **128~200** | 큰 움직임 (달리기, 춤) |
| **201~255** | 과격한 움직임 (주의: 품질 저하 가능) |

**FPS (Frames Per Second):**

프레임 속도를 조절합니다. SVD-XT 1.1은 **6 FPS로 고정** 훈련되어 있어, 6 근처 값이 가장 안정적입니다.

| FPS | 특징 |
|-----|------|
| **6** | 기본값, 가장 안정적 |
| **12~15** | 더 부드러운 동작, 약간의 아티팩트 |
| **24+** | 부자연스러운 결과 가능 |

### 개념 5: SVD 모델 변형

**SVD (14 프레임):**

- 576×1024 해상도
- 14프레임 생성
- 약 2초 영상

**SVD-XT (25 프레임):**

- 576×1024 해상도
- 25프레임 생성 (SVD를 미세 조정)
- 약 4초 영상

**SVD-XT 1.1 (개선판):**

- 고정 조건: 6 FPS, Motion Bucket 127
- **하이퍼파라미터 조정 없이** 일관된 결과
- 더 안정적인 출력

**해상도 가이드:**

| 해상도 | 권장 여부 |
|--------|-----------|
| **576×1024** | ✅ 최적 (훈련 해상도) |
| **768×1024** | ⚠️ 작동하나 품질 저하 |
| **1024×1024** | ⚠️ 정사각형은 비권장 |
| **기타** | ❌ 훈련 범위 벗어남 |

## 실습: SVD로 이미지 애니메이션

### 기본 Image-to-Video 생성

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# SVD 파이프라인 로드
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",  # 25프레임 버전
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")

# 메모리 최적화
pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking()

# 입력 이미지 로드 (576x1024 권장)
image = load_image("path/to/your/image.png")
image = image.resize((1024, 576))

# 비디오 생성
frames = pipe(
    image,
    decode_chunk_size=8,      # VRAM 절약
    num_frames=25,            # SVD-XT는 25프레임
    motion_bucket_id=127,     # 중간 정도 움직임
    noise_aug_strength=0.02,  # 노이즈 증강 (약간의 변화)
    num_inference_steps=25,
).frames[0]

# 비디오 저장
export_to_video(frames, "animated_image.mp4", fps=6)
print("✅ 비디오 저장 완료: animated_image.mp4")
```

### Motion Bucket으로 움직임 강도 조절

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1",  # 1.1 버전
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.enable_model_cpu_offload()

image = load_image("portrait.png").resize((1024, 576))

# 다양한 Motion Bucket 비교
motion_buckets = [50, 127, 200]

for mb in motion_buckets:
    frames = pipe(
        image,
        decode_chunk_size=8,
        motion_bucket_id=mb,
        noise_aug_strength=0.02,
        num_inference_steps=25,
    ).frames[0]

    export_to_video(frames, f"motion_{mb}.mp4", fps=6)
    print(f"Motion Bucket {mb}: motion_{mb}.mp4 저장")

# 결과 비교:
# - motion_50.mp4: 미세한 움직임 (눈 깜빡임, 미세한 표정)
# - motion_127.mp4: 적당한 움직임 (고개 돌림, 미소)
# - motion_200.mp4: 큰 움직임 (상체 움직임, 손 제스처)
```

### 시드 고정으로 일관된 결과

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.enable_model_cpu_offload()

image = load_image("landscape.png").resize((1024, 576))

# 시드 고정으로 재현 가능한 결과
seed = 42
generator = torch.Generator(device="cuda").manual_seed(seed)

frames = pipe(
    image,
    decode_chunk_size=8,
    motion_bucket_id=100,
    noise_aug_strength=0.02,
    num_inference_steps=30,
    generator=generator,
).frames[0]

export_to_video(frames, "consistent_video.mp4", fps=6)
print(f"시드 {seed}로 생성: consistent_video.mp4")
```

### 프레임 보간으로 부드러운 영상

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import cv2
import numpy as np

# SVD로 기본 비디오 생성
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.enable_model_cpu_offload()

image = load_image("action_scene.png").resize((1024, 576))

frames = pipe(
    image,
    decode_chunk_size=8,
    motion_bucket_id=150,
    num_inference_steps=25,
).frames[0]

# 6fps → 24fps로 프레임 보간 (간단한 선형 보간)
def interpolate_frames(frames, target_fps_multiplier=4):
    """프레임 사이에 보간 프레임 삽입"""
    interpolated = []
    for i in range(len(frames) - 1):
        interpolated.append(frames[i])
        # 중간 프레임 생성 (간단한 블렌딩)
        for j in range(1, target_fps_multiplier):
            alpha = j / target_fps_multiplier
            # PIL Image를 numpy로 변환
            f1 = np.array(frames[i])
            f2 = np.array(frames[i + 1])
            blended = (f1 * (1 - alpha) + f2 * alpha).astype(np.uint8)
            from PIL import Image
            interpolated.append(Image.fromarray(blended))
    interpolated.append(frames[-1])
    return interpolated

# 프레임 보간 적용
smooth_frames = interpolate_frames(frames, target_fps_multiplier=4)

# 24fps로 저장
export_to_video(smooth_frames, "smooth_video.mp4", fps=24)
print(f"보간 적용: {len(frames)} → {len(smooth_frames)} 프레임, 24fps")
```

## 더 깊이 알아보기: SVD의 데이터 비밀

**데이터 큐레이션의 중요성**

SVD 논문의 핵심 기여 중 하나는 **데이터 품질이 모델 성능을 결정한다**는 발견입니다. Stability AI는 대규모 비디오 데이터셋을 만들면서 다음을 철저히 필터링했습니다:

1. **정적 비디오 제거**: 움직임이 없는 영상 제외
2. **과도한 텍스트 제거**: 워터마크, 자막 많은 영상 제외
3. **저해상도 제거**: 480p 미만 필터링
4. **급격한 장면 전환 제거**: 컷 편집이 많은 영상 제외

**LVD-142M 데이터셋:**

| 항목 | 수치 |
|------|------|
| 총 클립 수 | 1.4억 개 |
| 총 시간 | 580,000 시간 |
| 원본 소스 | 공개 웹 비디오 |
| 필터링 후 | 약 2,500만 개 사용 |

> ⚠️ **흔한 오해**: "더 많은 데이터 = 더 좋은 모델"
>
> SVD 연구진은 반대 결과를 발견했습니다. **품질 높은 소량의 데이터**로 미세 조정했을 때 오히려 성능이 향상되었습니다. 빅데이터의 양보다 큐레이션의 질이 중요한 거죠.

**왜 Image-to-Video가 중요한가?**

Text-to-Video는 멋지지만 제어가 어렵습니다. 특정 인물, 특정 구도, 특정 스타일을 정확히 텍스트로 표현하기 어렵죠. I2V는 이 문제를 해결합니다:

1. **완벽한 첫 프레임 생성** (SD, SDXL, FLUX 등으로)
2. **SVD로 애니메이션 적용**
3. 필요시 **ControlNet + AnimateDiff로 추가 제어**

이 워크플로우가 현재 프로 크리에이터들이 가장 많이 사용하는 방식입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "SVD는 어떤 이미지든 잘 애니메이션한다"
>
> 그렇지 않습니다. **사진 같은 사실적인 이미지**에서 가장 잘 동작하고, 추상적이거나 만화 스타일 이미지는 결과가 불안정할 수 있습니다. 입력 이미지 품질이 출력 품질을 결정합니다.

> 💡 **알고 계셨나요?**: SVD의 "motion_bucket_id" 파라미터는 훈련 시 사용된 **광학 흐름(Optical Flow)의 크기**를 기반으로 합니다. 높은 값일수록 훈련 데이터에서 빠른 움직임의 비디오가 많았다는 의미죠.

> 🔥 **실무 팁**: 인물 사진을 애니메이션할 때는 **정면을 바라보는 포즈**가 가장 안정적입니다. 옆모습이나 특이한 각도는 얼굴 왜곡이 발생할 수 있습니다.

> 🔥 **실무 팁**: SVD 결과물이 마음에 들지 않으면, **noise_aug_strength를 0.0~0.1 사이로 조절**해 보세요. 이 값이 높으면 원본 이미지에서 더 많이 벗어나고, 낮으면 원본에 더 가깝게 유지됩니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **Image-to-Video** | 이미지를 첫 프레임으로 고정하고 비디오 생성 |
| **Temporal Conv/Attention** | 시간 축 처리를 위한 추가 레이어 |
| **Motion Bucket** | 움직임 강도 조절 (1~255, 기본 127) |
| **FPS Conditioning** | 프레임 속도 조건 (6 FPS 권장) |
| **SVD-XT** | 25프레임 버전, 약 4초 영상 |
| **SVD-XT 1.1** | 고정 하이퍼파라미터로 안정적 출력 |

## 다음 섹션 미리보기

SVD로 몇 초짜리 영상을 생성했습니다. 하지만 영화나 광고에 쓰려면 **더 길고, 더 복잡한 동작**이 필요하죠. 다음 섹션 [Sora와 대규모 비디오 모델](./04-sora.md)에서는 OpenAI의 Sora를 비롯한 **최신 대규모 비디오 생성 모델**을 살펴봅니다. U-Net에서 Diffusion Transformer로의 패러다임 전환, 시공간 패치(Spacetime Patches) 등 비디오 생성의 미래를 엿봅니다.

## 참고 자료

- [Stable Video Diffusion 논문 (arXiv)](https://arxiv.org/abs/2311.15127) - 원본 논문
- [SVD - Stability AI](https://stability.ai/stable-video) - 공식 페이지
- [Hugging Face SVD-XT 1.1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) - 모델 카드
- [OpenVINO SVD 튜토리얼](https://docs.openvino.ai/2024/notebooks/stable-video-diffusion-with-output.html) - Intel 최적화 가이드
- [Civitai SVD 가이드](https://education.civitai.com/quickstart-guide-to-stable-video-diffusion/) - 실습 튜토리얼
