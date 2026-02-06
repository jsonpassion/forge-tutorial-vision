# SD 1.5 vs SDXL

> 모델 버전별 차이점

## 개요

Stable Diffusion은 1.x에서 SDXL로 진화하면서 이미지 품질, 해상도, 텍스트 이해력이 크게 향상되었습니다. 이번 섹션에서는 두 모델의 아키텍처 차이를 기술적으로 비교하고, 실전에서 어떤 모델을 선택해야 하는지 가이드합니다.

**선수 지식**: [Stable Diffusion 아키텍처](./01-sd-architecture.md)
**학습 목표**:
- SD 1.5와 SDXL의 아키텍처 차이를 이해한다
- SDXL의 듀얼 텍스트 인코더와 리파이너의 역할을 파악한다
- 각 모델의 장단점과 적합한 사용 시나리오를 안다

## 왜 알아야 할까?

커뮤니티에서는 여전히 SD 1.5와 SDXL이 공존합니다. SD 1.5는 [LoRA](../14-generative-practice/01-lora.md)와 커스텀 모델의 생태계가 풍부하고, SDXL은 기본 품질이 뛰어나죠. 각 모델의 강점을 알아야 상황에 맞는 선택을 할 수 있습니다.

## 핵심 개념

### 개념 1: 핵심 스펙 비교

| 항목 | SD 1.5 | SDXL |
|------|--------|------|
| **U-Net 파라미터** | 8.6억 | **26억** (3배) |
| **기본 해상도** | 512×512 | **1024×1024** (4배 픽셀) |
| **텍스트 인코더** | CLIP ViT-L/14 (1개) | **OpenCLIP ViT-G + CLIP ViT-L** (2개) |
| **텍스트 임베딩 차원** | 768 | **2048** (768+1280) |
| **VAE 잠재 채널** | 4 | 4 |
| **리파이너** | 없음 | **별도 리파이너 모델** |
| **GPU 메모리 (최소)** | ~4GB | ~8GB |
| **생성 속도 (512²)** | ~1.2초 | ~3.5초 |

> 💡 **비유**: SD 1.5가 **콤팩트 카메라**라면, SDXL은 **풀프레임 DSLR**입니다. DSLR이 화질은 뛰어나지만 무겁고 비싸듯이, SDXL은 품질은 좋지만 더 많은 리소스를 요구하죠.

### 개념 2: 듀얼 텍스트 인코더 — SDXL의 비밀 무기

SD 1.5는 CLIP ViT-L/14 하나만 사용하지만, SDXL은 **두 개의 텍스트 인코더**를 동시에 사용합니다:

- **OpenCLIP ViT-bigG**: 큰 모델로 풍부한 의미 포착 (1280차원)
- **CLIP ViT-L/14**: SD 1.5와 동일한 인코더 (768차원)

두 인코더의 출력을 **concat**하여 2048차원의 텍스트 임베딩을 만듭니다. 이 덕분에 SDXL은 복잡한 프롬프트를 더 정확하게 이해합니다. 테스트 결과 프롬프트 준수율이 SD 1.5의 71%에서 SDXL의 89%로 향상되었죠.

### 개념 3: 리파이너 — 2단계 생성 파이프라인

SDXL은 **Base + Refiner** 2단계 파이프라인을 지원합니다:

1. **Base 모델**: 전체 디노이징 스텝 중 초반 80%를 담당 — 전체적인 구조와 구도 생성
2. **Refiner 모델**: 나머지 20%를 담당 — 세밀한 디테일과 질감 개선

리파이너는 선택사항이지만, 사용하면 특히 **피부 질감, 눈동자, 세밀한 패턴**에서 품질이 눈에 띄게 향상됩니다.

### 개념 4: 실전 선택 가이드

| 상황 | 추천 모델 | 이유 |
|------|----------|------|
| **GPU 메모리 6GB 이하** | SD 1.5 | SDXL은 메모리 부족 가능 |
| **최고 품질 필요** | SDXL + Refiner | 디테일과 해상도 우위 |
| **빠른 프로토타이핑** | SD 1.5 | 생성 속도 2~3배 빠름 |
| **커스텀 LoRA 다양성** | SD 1.5 | 생태계가 훨씬 풍부 |
| **텍스트 정확도 중요** | SDXL | 듀얼 인코더의 장점 |
| **최신 모델 사용** | FLUX | SD 생태계를 넘어선 품질 |

> ⚠️ **흔한 오해**: "SDXL이 항상 SD 1.5보다 좋다" — 기본 품질은 SDXL이 뛰어나지만, 잘 튜닝된 SD 1.5 커스텀 모델(예: DreamShaper, Realistic Vision)이 특정 스타일에서는 SDXL을 능가할 수 있습니다. 생태계의 성숙도도 중요한 요소예요.

## 실습: SD 1.5와 SDXL 비교 실행

```python
# SD 1.5와 SDXL을 동일 프롬프트로 비교
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch

prompt = "a majestic lion in a savanna at golden hour, photorealistic"
negative = "blurry, low quality, distorted"

# === SD 1.5 ===
pipe_15 = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image_15 = pipe_15(
    prompt=prompt,
    negative_prompt=negative,
    num_inference_steps=30,
    guidance_scale=7.5,
    width=512, height=512,                # SD 1.5 기본 해상도
).images[0]
image_15.save("lion_sd15.png")

# === SDXL ===
pipe_xl = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

image_xl = pipe_xl(
    prompt=prompt,
    negative_prompt=negative,
    num_inference_steps=30,
    guidance_scale=7.5,
    width=1024, height=1024,              # SDXL 기본 해상도
).images[0]
image_xl.save("lion_sdxl.png")

print("두 이미지를 비교해보세요!")
print(f"SD 1.5: 512×512, U-Net 8.6억 파라미터")
print(f"SDXL:   1024×1024, U-Net 26억 파라미터")
```

## 더 깊이 알아보기

### SD 2.x — 잊혀진 중간 세대

SD 1.5와 SDXL 사이에 **SD 2.0/2.1**이 존재했습니다. SD 2.0은 OpenCLIP을 텍스트 인코더로 교체하고, $v$-prediction을 도입했지만, 커뮤니티에서 외면받았죠. 이유는:
- SD 1.5용 LoRA/모델과 **호환 불가**
- NSFW 필터링이 강화되어 예술적 자유도 제한
- 품질 향상이 호환성 파괴를 정당화할 만큼 크지 않음

이 실패 경험이 SDXL 설계에 반영되어, SDXL은 SD 1.5의 CLIP ViT-L도 함께 포함하게 되었습니다.

> 💡 **알고 계셨나요?**: SD 1.5는 사실 RunwayML에서 학습한 모델입니다. 원래 CompVis/Stability AI/RunwayML의 3자 협업이었지만, 이후 RunwayML은 별도로 Gen-1/Gen-2 영상 생성 모델을 개발하며 독자 노선을 걸었죠.

## 흔한 오해와 팁

> 🔥 **실무 팁**: SDXL에서 해상도를 1024×1024가 아닌 다른 크기로 생성할 때는, SDXL이 학습된 해상도 버킷(1024×1024, 1152×896, 1216×832 등)에 가까운 크기를 사용하세요. 학습 해상도와 너무 다르면 품질이 떨어집니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| SD 1.5 | 512² 기본, 8.6억 파라미터, 가볍고 생태계 풍부 |
| SDXL | 1024² 기본, 26억 파라미터, 고품질/고해상도 |
| 듀얼 텍스트 인코더 | OpenCLIP ViT-G + CLIP ViT-L 결합으로 텍스트 이해 향상 |
| 리파이너 | SDXL의 2단계 파이프라인, 디테일 향상용 |
| $v$-prediction | SD 2.x에서 도입된 예측 대상 변경, SDXL에도 적용 |

## 다음 섹션 미리보기

모델의 성능을 최대로 끌어내려면 **어떻게 말하느냐**가 중요합니다. 다음 섹션 [프롬프트 엔지니어링](./03-prompting.md)에서는 효과적인 프롬프트 작성법을 체계적으로 알아봅니다.

## 참고 자료

- [Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis" (2023)](https://arxiv.org/abs/2307.01952) - SDXL 논문
- [SDXL vs SD 1.5 Comparison](https://sdxlturbo.ai/blog-SDXL-10-vs-Stable-Diffusion-15-Handson-Comparison-1518) - 실전 비교 가이드
- [stabilityai/stable-diffusion-xl-base-1.0 (HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) - SDXL 공식 모델
- [Comparing Stable Diffusion Models (Medium)](https://medium.com/@promptingpixels/comparing-stable-diffusion-models-2c1dc9919ab7) - 버전별 상세 비교
