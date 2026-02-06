# SD3와 미래 방향

> MMDiT 아키텍처와 최신 동향

## 개요

[FLUX 모델](./05-flux.md)에서 Black Forest Labs의 차세대 아키텍처를 살펴봤습니다. 이번 섹션에서는 같은 MM-DiT 뿌리에서 출발한 **Stable Diffusion 3(SD3)** 시리즈를 분석하고, FLUX와 비교합니다. 나아가 이미지 생성 AI가 비디오, 3D, 멀티모달로 확장되는 **미래 방향**까지 전망합니다.

**선수 지식**: [SD 아키텍처](./01-sd-architecture.md), [FLUX 모델](./05-flux.md), [Latent Diffusion](../12-diffusion-models/06-latent-diffusion.md)
**학습 목표**:
- SD3의 MM-DiT 아키텍처를 이해하고 FLUX와 차이를 파악한다
- SD3.5 변형들(Large, Medium, Turbo)의 특징을 안다
- 이미지 생성 모델의 미래 발전 방향을 전망한다
- SD3를 HuggingFace Diffusers로 실행할 수 있다

## 왜 알아야 할까?

SD3와 FLUX는 **같은 뿌리(MM-DiT)**에서 출발했지만 서로 다른 설계 철학을 선택했습니다. 두 모델을 비교하면 현재 이미지 생성 기술의 설계 트레이드오프를 깊이 이해할 수 있어요. 또한 DiT 아키텍처는 이미지를 넘어 비디오 생성(Sora, Veo), 3D 생성 등으로 확장되고 있어서, 이 흐름을 파악하는 것은 CV 전체의 미래를 읽는 열쇠가 됩니다.

## 핵심 개념

### 개념 1: SD3의 MM-DiT — 순수 이중 스트림 설계

> 💡 **비유**: SD3와 FLUX의 차이는 **오케스트라 편성**의 차이와 같습니다. SD3는 현악과 관악을 **끝까지 따로 유지하며** 협연하는 방식이고, FLUX는 처음에는 따로 연주하다가 **후반부에 합주**로 전환하는 방식이에요. 둘 다 아름다운 음악을 만들지만, 접근법이 다른 거죠.

SD3는 2024년 2월 Stability AI가 발표한 논문 "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"에서 소개되었습니다. [FLUX](./05-flux.md)와 마찬가지로 MM-DiT를 사용하지만, 블록 구성에서 핵심적인 차이가 있습니다.

**SD3 vs FLUX 아키텍처 비교:**

| 항목 | SD3 | FLUX.1 |
|------|-----|--------|
| **이중 스트림 블록** | **24개 (전부)** | 19개 |
| **단일 스트림 블록** | **0개** | 38개 |
| **총 블록 수** | 24개 | 57개 |
| **총 파라미터** | ~20억 | ~119억 |
| **텍스트 인코더** | CLIP ViT-L + OpenCLIP ViT-G + **T5-XXL** | CLIP ViT-L + **T5-XXL** |
| **학습 방식** | Rectified Flow | Rectified Flow |
| **QK 정규화** | **RMSNorm** 적용 | 적용 |

**핵심 차이**: SD3는 **모든 블록이 이중 스트림**입니다. 텍스트와 이미지가 처음부터 끝까지 독립적인 가중치를 유지하면서 어텐션에서만 정보를 교환하죠. 반면 FLUX는 19개 이중 스트림 후 38개 단일 스트림으로 전환하여 **효율성**을 높였습니다.

SD3가 텍스트 인코더를 **3개** 사용한다는 점도 주목할 만합니다. CLIP ViT-L, OpenCLIP ViT-G, 그리고 T5-XXL까지 — 텍스트 이해에 최대한 투자한 설계입니다.

> ⚠️ **흔한 오해**: "SD3는 FLUX보다 열등하다" — 파라미터 수가 적다고 나쁜 것은 아닙니다. SD3.5 Medium(25억)은 소비자 하드웨어에서 실행 가능하면서도 높은 품질을 제공합니다. 용도에 따라 올바른 모델이 다른 거죠.

### 개념 2: SD3.5 시리즈 — 실용성을 위한 변형

2024년 10월, Stability AI는 SD3.5 시리즈를 공개하며 다양한 하드웨어 환경을 지원했습니다:

| 모델 | 파라미터 | 스텝 | VRAM | 특징 |
|------|---------|------|------|------|
| **SD3.5 Large** | 80억 | 28~50 | ~24GB | 최고 품질, 1MP 해상도 |
| **SD3.5 Large Turbo** | 80억 | **4** | ~24GB | 디스틸레이션으로 4스텝 생성 |
| **SD3.5 Medium** | **25억** | 28~50 | **~10GB** | 소비자 GPU에서 실행 가능 |

SD3.5의 핵심 개선 사항:

- **MMDiT-X 아키텍처** (Medium): 기존 MM-DiT를 개선하여 적은 파라미터로 효율적 생성
- **QK Normalization**: Transformer 블록에 Query-Key 정규화를 적용하여 학습 안정성 향상, **파인튜닝이 쉬워짐**
- **다양한 스타일**: 3D, 사진, 회화, 선화 등 광범위한 스타일 생성 가능
- **커뮤니티 라이선스**: 연매출 100만 달러 미만 기업은 무료 상용 사용 가능

> 🔥 **실무 팁**: 파인튜닝이나 [LoRA](../14-generative-practice/01-lora.md) 학습을 계획한다면 SD3.5 Medium이 가장 실용적입니다. 25억 파라미터라 학습이 빠르고, QK Normalization 덕분에 학습이 안정적이에요.

### 개념 3: SD3 vs FLUX — 실전 비교

같은 뿌리에서 출발한 두 모델, 실전에서는 어떤 차이가 있을까요?

| 비교 항목 | SD3.5 Large | FLUX.1 Dev |
|-----------|-------------|------------|
| **이미지 품질** | 우수 | **매우 우수** |
| **텍스트 렌더링** | 양호 | **뛰어남** |
| **인체 정확도** | 양호 | **뛰어남** |
| **프롬프트 충실도** | 우수 | **매우 우수** |
| **생성 속도** | 빠름 (80억) | 느림 (119억) |
| **VRAM 요구** | ~24GB | ~32GB+ |
| **커스터마이징 생태계** | 성장 중 | **빠르게 확대** |
| **라이선스** | 커뮤니티 라이선스 | 비상용 (Dev) |

FLUX가 대부분의 품질 지표에서 앞서지만, SD3.5는 **접근성과 라이선스** 면에서 장점이 있습니다. 특히 SD3.5 Medium은 소비자 GPU에서도 실행 가능하다는 큰 강점이 있죠.

### 개념 4: DiT의 확장 — 이미지를 넘어 비디오로

> 💡 **비유**: DiT(Diffusion Transformer)가 이미지 생성에서 성공한 것은 **활판 인쇄의 발명**과 같습니다. 처음에는 책(이미지)만 찍었지만, 같은 원리로 신문(비디오), 포스터(3D), 광고(멀티모달)까지 확장된 것처럼, DiT라는 핵심 아키텍처가 다양한 생성 과제로 확장되고 있습니다.

FLUX와 SD3에서 검증된 DiT 아키텍처는 이미 비디오, 3D, 멀티모달 생성의 핵심 엔진으로 자리잡았습니다:

**비디오 생성:**
- **Sora** (OpenAI): DiT 기반 비디오 생성, 물리 법칙을 이해하는 장면 생성
- **Veo 3** (Google): 4K 포토리얼리즘과 통합 오디오 생성
- **Kling** (Kuaishou): 립싱크와 모션에 특화된 비디오 생성
- **Runway Gen-3**: 초현실적 10초 비디오 클립 생성

**통합 멀티모달 생성:**
- 텍스트, 이미지, 오디오, 비디오를 **하나의 모델**에서 처리
- 참조 이미지 + 텍스트 + 음악을 입력하면 완성된 비디오 출력
- FLUX.2의 멀티 레퍼런스(최대 10장 참조) 기능도 이 방향의 일부

**핵심 트렌드:**

| 방향 | 설명 | 대표 사례 |
|------|------|-----------|
| **스케일링** | 더 큰 모델, 더 많은 데이터 | FLUX.2 (320억), Sora |
| **효율화** | 적은 스텝, 작은 모델 | FLUX.2 Klein (40억), SD3.5 Medium |
| **멀티모달** | 이미지+비디오+오디오+3D 통합 | Veo 3, 멀티모달 플랫폼 |
| **제어 가능성** | 더 정밀한 사용자 제어 | [ControlNet](../14-generative-practice/03-controlnet.md), IP-Adapter |
| **개인화** | 적은 데이터로 스타일 학습 | [LoRA](../14-generative-practice/01-lora.md), [DreamBooth](../14-generative-practice/02-dreambooth.md) |

## 실습: SD3.5 실행해보기

```python
# SD3.5 Medium으로 이미지 생성 (소비자 GPU 친화적)
import torch
from diffusers import StableDiffusion3Pipeline

# SD3.5 Medium 파이프라인 로드
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# 이미지 생성
prompt = "A serene Japanese garden with a koi pond, cherry blossoms falling"
negative_prompt = "blurry, low quality, distorted"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=28,       # Medium 권장 스텝
    height=1024,
    width=1024,
    guidance_scale=5.0,
).images[0]

image.save("sd35_medium_result.png")
print(f"SD3.5 Medium 생성 완료! 크기: {image.size}")
```

```python
# SD3.5 Large Turbo로 초고속 생성 (4스텝)
pipe_turbo = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo",
    torch_dtype=torch.bfloat16
)
pipe_turbo.to("cuda")

prompt = "An astronaut riding a horse on Mars, cinematic lighting"

# Turbo는 4스텝이면 충분, CFG도 낮게
image = pipe_turbo(
    prompt=prompt,
    num_inference_steps=4,        # Turbo 최적 스텝
    height=1024,
    width=1024,
    guidance_scale=0.0,           # Turbo는 CFG 불필요
).images[0]

image.save("sd35_turbo_result.png")
print("4스텝 초고속 생성 완료!")
```

```python
# SD3.5 vs FLUX 비교 생성 (같은 프롬프트로)
import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from PIL import Image

prompt = "A wooden sign that reads 'Welcome to AI World', forest background"

# SD3.5 Medium 생성
pipe_sd3 = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
)
pipe_sd3.enable_model_cpu_offload()

img_sd3 = pipe_sd3(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=5.0,
).images[0]

# FLUX Schnell 생성
pipe_flux = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)
pipe_flux.enable_model_cpu_offload()

img_flux = pipe_flux(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]

# 나란히 비교
comparison = Image.new('RGB', (2048, 1024))
comparison.paste(img_sd3, (0, 0))
comparison.paste(img_flux, (1024, 0))
comparison.save("sd3_vs_flux_comparison.png")
print("비교 이미지 저장 완료! 왼쪽: SD3.5, 오른쪽: FLUX")
```

## 더 깊이 알아보기

### U-Net에서 DiT로 — 패러다임 전환의 의미

[SD 아키텍처](./01-sd-architecture.md)에서 배운 U-Net은 2015년부터 이미지 생성의 핵심 백본이었습니다. 하지만 2023년 DiT(Diffusion Transformer) 논문 이후, 이미지 생성의 패러다임이 바뀌기 시작했죠.

| 세대 | 디노이저 | 대표 모델 | 시기 |
|------|---------|-----------|------|
| **1세대** | U-Net | SD 1.x, SD 2.x | 2022 |
| **2세대** | 확대된 U-Net | SDXL | 2023 |
| **3세대** | MM-DiT (Transformer) | SD3, FLUX.1 | 2024 |
| **4세대** | 확대된 DiT + 멀티모달 | FLUX.2, Sora | 2025 |

이 전환의 핵심 이유는 **스케일링 법칙(Scaling Law)**입니다. Transformer는 파라미터를 늘릴수록 성능이 예측 가능하게 향상되는 특성이 있는데, U-Net은 이런 스케일링이 어려웠거든요. LLM 분야에서 Transformer의 스케일링이 증명된 후, 이미지 생성도 같은 길을 따라가게 된 것입니다.

### SD3의 공개와 논란

SD3는 흥미로운 역사를 가지고 있습니다. 2024년 2월에 논문이 공개되었을 때 커뮤니티의 기대가 컸지만, 6월 실제 모델이 공개되었을 때는 기대에 미치지 못했다는 평가가 많았습니다. 특히 인체 생성 품질에서 SDXL보다 떨어진다는 비판이 있었죠. 이후 10월에 공개된 SD3.5 시리즈에서 대부분의 문제가 해결되었지만, 이 시기에 이미 FLUX가 커뮤니티의 주류가 된 상태였습니다. 기술의 완성도만큼 **타이밍**이 중요하다는 교훈을 보여주는 사례입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "SD3는 실패한 모델이다" — SD3의 초기 공개가 아쉬웠던 것은 사실이지만, SD3.5 Medium은 10GB VRAM에서 실행 가능한 **가장 접근성 높은 DiT 모델** 중 하나입니다. 용도에 맞는 모델 선택이 중요합니다.

> 💡 **알고 계셨나요?** SD3와 FLUX는 같은 연구팀의 작업물에서 갈라져 나왔습니다. SD3 논문의 공동 저자들 중 Robin Rombach, Patrick Esser 등이 이후 Black Forest Labs를 설립하고 FLUX를 만들었죠. 형제 모델인 셈입니다.

> 🔥 **실무 팁**: 2025년 기준으로 새 프로젝트를 시작한다면, **품질 우선이면 FLUX.1 Dev**, **접근성 우선이면 SD3.5 Medium**, **속도 우선이면 FLUX.1 Schnell** 또는 **FLUX.2 Klein**을 추천합니다. 어떤 것을 선택하든 [LoRA](../14-generative-practice/01-lora.md)로 커스터마이징할 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **SD3 MM-DiT** | 24개 이중 스트림 블록만 사용하는 순수 MM-DiT 설계 |
| **SD3.5 변형** | Large(80억), Large Turbo(4스텝), Medium(25억, 10GB VRAM) |
| **SD3 vs FLUX** | 같은 MM-DiT 뿌리, SD3는 접근성, FLUX는 품질에 강점 |
| **DiT 확장** | 이미지 → 비디오(Sora/Veo) → 멀티모달로 확장 중 |
| **스케일링 트렌드** | Transformer의 스케일링 법칙이 이미지 생성에도 적용 |
| **미래 방향** | 스케일링 + 효율화 + 멀티모달 + 제어 가능성 + 개인화 |

## 다음 섹션 미리보기

축하합니다! Chapter 13 Stable Diffusion 심화를 모두 마쳤습니다. 다음 [Ch14. 생성 AI 실전](../14-generative-practice/01-lora.md)에서는 지금까지 배운 모델들을 **실제로 커스터마이징**하는 방법을 다룹니다. LoRA로 스타일을 학습하고, DreamBooth로 특정 인물/객체를 생성하며, ControlNet으로 구도를 정밀하게 제어하는 실전 기법들을 배워보겠습니다.

## 참고 자료

- [Scaling Rectified Flow Transformers (SD3 논문)](https://stability.ai/news/stable-diffusion-3-research-paper) - SD3의 원본 연구 논문
- [Introducing Stable Diffusion 3.5 — Stability AI](https://stability.ai/news/introducing-stable-diffusion-3-5) - SD3.5 시리즈 공식 발표
- [Stable Diffusion 3 & FLUX: Complete Guide to MMDiT Architecture](https://blog.sotaaz.com/post/sd3-flux-architecture-en) - SD3와 FLUX의 MM-DiT 아키텍처 상세 비교
- [SD3.5 Architecture and Inference — LearnOpenCV](https://learnopencv.com/stable-diffusion-3/) - SD3.5 아키텍처와 추론 과정 해설
- [Stable Diffusion 3.5 vs. Flux — Modal](https://modal.com/blog/best-text-to-image-model-article) - SD3.5와 FLUX의 실전 비교 분석
- [Demystifying Flux Architecture (arXiv 2507.09595)](https://arxiv.org/abs/2507.09595) - FLUX 아키텍처 분석 논문
- [HuggingFace Video Generation Blog](https://huggingface.co/blog/video_gen) - 오픈소스 비디오 생성 모델 현황
