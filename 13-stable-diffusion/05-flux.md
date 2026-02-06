# FLUX 모델

> 차세대 Diffusion Transformer

## 개요

[SD 1.5 vs SDXL](./02-sd15-vs-sdxl.md)에서 Stable Diffusion이 U-Net 기반으로 진화해온 과정을 살펴봤는데요, FLUX는 여기서 한 단계 더 나아갑니다. **U-Net을 완전히 버리고 Transformer만으로** 이미지를 생성하는 차세대 모델이거든요. 119억 파라미터의 이 거대한 모델은 2024년 공개되자마자 이미지 품질, 텍스트 렌더링, 프롬프트 충실도 면에서 기존 모델들을 압도하며 새로운 기준을 세웠습니다.

**선수 지식**: [SD 아키텍처](./01-sd-architecture.md), [SD 1.5 vs SDXL](./02-sd15-vs-sdxl.md), [Vision Transformer](../09-vision-transformer/03-vit.md)
**학습 목표**:
- FLUX의 MM-DiT 아키텍처를 이해한다
- Rectified Flow가 기존 디퓨전과 다른 점을 파악한다
- FLUX.1의 세 가지 변형(Pro/Dev/Schnell)의 차이를 안다
- FLUX.2의 발전 방향과 Klein 모델을 이해한다
- HuggingFace Diffusers로 FLUX를 실행할 수 있다

## 왜 알아야 할까?

2024년 8월, Stability AI의 핵심 개발자였던 Robin Rombach가 세운 **Black Forest Labs**가 FLUX.1을 공개했습니다. SD 1.5와 SDXL의 시대를 열었던 바로 그 사람이 만든 "다음 세대" 모델인 셈이죠. FLUX는 텍스트 렌더링 능력이 특히 뛰어나고, 인체 해부학적 정확도도 크게 개선되었습니다. [LoRA](../14-generative-practice/01-lora.md)와 [ControlNet](../14-generative-practice/03-controlnet.md) 등 Stable Diffusion 생태계의 기법들이 FLUX에도 적용되고 있어, 이 아키텍처를 이해하는 것은 이미지 생성의 현재와 미래를 파악하는 데 필수적입니다.

## 핵심 개념

### 개념 1: FLUX의 탄생 — Stable Diffusion 창시자의 새 도전

> 💡 **비유**: Stable Diffusion이 "디지털 카메라의 등장"이었다면, FLUX는 "미러리스 카메라로의 전환"과 같습니다. 핵심 원리(빛으로 이미지 만들기)는 같지만, 내부 구조를 완전히 새로 설계해서 성능과 효율을 동시에 잡은 거죠.

Black Forest Labs(BFL)는 2024년 초 Robin Rombach, Andreas Blattmann, Patrick Esser가 Stability AI를 떠나 설립한 회사입니다. 이들은 Stable Diffusion의 기반이 된 Latent Diffusion Model(LDM) 논문의 저자들이기도 하죠. 자신들이 만든 모델의 한계를 가장 잘 아는 사람들이 새로운 아키텍처를 설계한 것입니다.

FLUX와 기존 SD의 가장 큰 차이는 **디노이징 백본**입니다:

| 항목 | SD 1.5 / SDXL | FLUX.1 |
|------|---------------|--------|
| **디노이징 백본** | U-Net | **MM-DiT (Transformer)** |
| **파라미터** | 8.6억 / 26억 | **119억** |
| **텍스트 인코더** | CLIP (1~2개) | **CLIP + T5-XXL** (2개) |
| **학습 방식** | DDPM (전통 디퓨전) | **Rectified Flow** |
| **위치 인코딩** | 절대 위치 | **RoPE** (회전 위치 인코딩) |
| **기본 해상도** | 512 / 1024 | **다양한 해상도 지원** |

### 개념 2: MM-DiT — 멀티모달 Diffusion Transformer

> 💡 **비유**: 기존 SD의 U-Net이 **통역사를 통해 대화하는** 것이라면, MM-DiT는 **텍스트와 이미지가 직접 같은 테이블에 앉아 대화하는** 것과 같습니다. 중간 번역 없이 서로의 맥락을 바로 이해하니까 소통이 훨씬 정확해지죠.

FLUX의 핵심은 **MM-DiT(Multimodal Diffusion Transformer)** 아키텍처입니다. [Vision Transformer](../09-vision-transformer/03-vit.md)에서 배운 Transformer를 디퓨전 모델의 디노이저로 사용하는 건데, 텍스트와 이미지를 **모두 토큰으로** 처리한다는 게 핵심이에요.

**FLUX의 57개 Transformer 블록 구성:**

**1단계: 이중 스트림 블록 (Double-Stream Blocks) — 19개**

이중 스트림 블록에서는 텍스트 토큰과 이미지 토큰이 **각각 독립적인 가중치**로 처리됩니다. 하지만 어텐션 연산에서는 두 모달리티의 투영(projection)을 **결합(concatenate)하여 하나의 어텐션**을 수행합니다. 이렇게 하면 텍스트→이미지, 이미지→텍스트 양방향으로 정보가 흐르게 됩니다.

> 텍스트 토큰 ──┐
>                ├──→ **통합 어텐션** ──→ 텍스트 출력 / 이미지 출력 (각각 독립 FFN)
> 이미지 토큰 ──┘

**2단계: 단일 스트림 블록 (Single-Stream Blocks) — 38개**

단일 스트림 블록에서는 텍스트와 이미지 토큰을 **하나로 결합**하여 동일한 가중치로 처리합니다. 이중 스트림에서 충분히 양방향 정보를 교환한 후이므로, 통합 처리해도 각 모달리티의 특성이 유지됩니다.

> (텍스트 + 이미지) 결합 토큰 ──→ **통합 Transformer 블록** ──→ 출력

| 블록 유형 | 개수 | 블록당 파라미터 | 특징 |
|-----------|------|-----------------|------|
| 이중 스트림 | 19개 | ~3.4억 | 모달리티별 독립 가중치 + 통합 어텐션 |
| 단일 스트림 | 38개 | ~1.4억 | 결합 토큰을 통합 가중치로 처리 |

> ⚠️ **흔한 오해**: "Transformer 기반이면 U-Net보다 무조건 무겁다" — 실제로 FLUX의 단일 스트림 블록은 이중 스트림보다 **파라미터가 60% 적습니다**. 하이브리드 설계로 효율성과 성능을 동시에 잡은 거죠.

### 개념 3: Rectified Flow — 직선으로 가는 디퓨전

> 💡 **비유**: 기존 DDPM/DDIM이 **구불구불한 산길**을 따라 하산한다면, Rectified Flow는 **직선 케이블카**를 타는 것과 같습니다. 출발점(노이즈)에서 도착점(이미지)까지 가장 직선에 가까운 경로를 학습하니까, 적은 스텝으로도 목적지에 도착할 수 있어요.

[DDPM](../12-diffusion-models/02-ddpm.md)은 노이즈를 점진적으로 추가/제거하는 **확률적 과정(SDE)**을 사용합니다. 반면 Rectified Flow는 **노이즈와 데이터 사이의 직선 경로**를 학습하는 ODE 기반 접근법입니다.

**학습 과정의 차이:**

- **DDPM**: 노이즈 스케줄에 따라 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ 형태로 보간
- **Rectified Flow**: 단순 선형 보간 $x_t = (1-t) x_0 + t \epsilon$ 으로 직선 경로 학습

여기서 $t$는 시간 스텝(0=원본, 1=순수 노이즈), $x_0$는 원본 이미지, $\epsilon$는 순수 노이즈입니다.

Rectified Flow의 핵심 아이디어는 **"경로를 최대한 직선으로 만들자"**입니다. 직선에 가까울수록 적은 스텝으로 정확하게 샘플링할 수 있거든요. 이것이 FLUX [schnell]이 단 1~4스텝만으로 고품질 이미지를 생성할 수 있는 비결입니다.

### 개념 4: 듀얼 텍스트 인코더 — CLIP + T5

SDXL이 CLIP 2개를 사용했다면, FLUX는 **CLIP + T5-XXL**이라는 조합을 선택했습니다.

| 인코더 | 역할 | 특징 |
|--------|------|------|
| **CLIP ViT-L/14** | 시각-언어 정렬 | 이미지-텍스트 유사도에 특화 |
| **T5-XXL (4.7B)** | 텍스트 이해 | 복잡한 문장 구조, 긴 프롬프트, **텍스트 렌더링** 능력 |

T5는 Google의 대규모 언어 모델로, CLIP보다 훨씬 풍부한 텍스트 이해 능력을 갖추고 있습니다. 특히 "Write 'Hello World' on the sign"처럼 **이미지 안에 특정 텍스트를 렌더링**하는 작업에서 T5의 기여가 결정적이에요. SDXL에서 거의 불가능했던 정확한 텍스트 렌더링이 FLUX에서 가능해진 핵심 이유입니다.

> 💡 **알고 계셨나요?** FLUX의 텍스트 인코더만 해도 약 50억 파라미터(T5-XXL 47억 + CLIP 4억)로, SD 1.5의 **전체 모델**보다 큽니다. 텍스트를 정확히 이해하는 데 이만큼의 투자가 필요한 셈이죠.

### 개념 5: FLUX.1 변형 — Pro, Dev, Schnell

FLUX.1은 세 가지 변형으로 출시되었습니다:

| 변형 | 라이선스 | 스텝 | 특징 | 적합한 용도 |
|------|---------|------|------|-------------|
| **Pro** | 상용 (API) | 25~50 | 최고 품질, 가장 정확한 프롬프트 충실도 | 상업적 프로젝트 |
| **Dev** | 비상용 | 20~30 | **가이던스 디스틸레이션** 적용, Pro에 근접한 품질 | 연구/개인 프로젝트 |
| **Schnell** | Apache 2.0 | **1~4** | **적대적 디스틸레이션** 적용, 초고속 | 실시간 애플리케이션 |

**디스틸레이션(Distillation)**이란 큰 모델(teacher)의 지식을 작은 모델 또는 적은 스텝의 모델(student)로 전달하는 기술입니다:

- **가이던스 디스틸레이션** (Dev): CFG(Classifier-Free Guidance) 없이도 CFG를 사용한 것 같은 효과를 내도록 학습. 추론 시 네트워크를 1번만 실행하면 됨
- **적대적 디스틸레이션** (Schnell): GAN의 판별자를 활용해 1~4스텝만에 고품질 이미지 생성이 가능하도록 학습

> 🔥 **실무 팁**: 프로토타이핑 단계에서는 Schnell(1~4스텝)로 빠르게 구도를 잡고, 최종 결과물은 Dev(20~30스텝)로 생성하는 **2단계 워크플로우**가 효율적입니다.

## 실습: FLUX 실행해보기

```python
# FLUX.1-schnell로 이미지 생성 (가장 빠른 변형)
import torch
from diffusers import FluxPipeline

# 파이프라인 로드 (schnell은 Apache 2.0 라이선스)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16  # FLUX는 bfloat16 권장
)
pipe.to("cuda")

# 이미지 생성 (schnell은 1~4스텝으로 충분)
prompt = "A cute robot reading a book in a cozy library, warm lighting"
image = pipe(
    prompt=prompt,
    num_inference_steps=4,        # schnell은 4스텝이면 충분
    height=1024,
    width=1024,
    guidance_scale=0.0,           # schnell은 CFG 불필요
).images[0]

image.save("flux_schnell_result.png")
print(f"이미지 생성 완료! 크기: {image.size}")
```

```python
# FLUX.1-dev로 고품질 이미지 생성
pipe_dev = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe_dev.to("cuda")

# Dev는 가이던스 디스틸레이션 덕분에 guidance_scale=3.5 정도로 충분
prompt = "A neon sign that reads 'HELLO FLUX' on a rainy street at night"
image = pipe_dev(
    prompt=prompt,
    num_inference_steps=28,       # Dev는 20~30스텝 권장
    height=1024,
    width=1024,
    guidance_scale=3.5,           # Dev의 권장 CFG 값
).images[0]

image.save("flux_dev_result.png")
print(f"텍스트 렌더링 테스트 완료!")
```

```python
# VRAM이 부족할 때 — CPU 오프로딩 활용
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)

# 모델 전체를 GPU에 올리지 않고, 필요한 부분만 GPU로 이동
pipe.enable_model_cpu_offload()

# 메모리 효율적 어텐션 활성화
pipe.enable_xformers_memory_efficient_attention()

prompt = "A beautiful landscape painting in the style of Monet"
image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    height=768,                   # 해상도를 줄이면 VRAM 절약
    width=768,
    guidance_scale=0.0,
).images[0]

image.save("flux_memory_efficient.png")
print("메모리 효율 모드로 생성 완료!")
```

## 더 깊이 알아보기

### FLUX.2 — 2025년의 진화

2025년 11월, BFL은 FLUX.2 시리즈를 공개했습니다. 주요 발전 사항:

| 특징 | FLUX.1 | FLUX.2 |
|------|--------|--------|
| **파라미터** | 119억 | **320억** |
| **멀티 레퍼런스** | 미지원 | **최대 10장 참조 가능** |
| **최대 해상도** | ~2MP | **4MP** |
| **타이포그래피** | 우수 | **세밀한 텍스트/UI/인포그래픽** |
| **VAE** | FLUX.1 VAE | **새로운 VAE (처음부터 재학습)** |

FLUX.2의 가장 주목할 변형은 **Klein** 모델입니다:
- **FLUX.2 [klein] 4B**: 40억 파라미터로 경량화, ~13GB VRAM에서 실행 가능
- **Apache 2.0 라이선스**: 완전 오픈소스
- 1초 이내 이미지 생성이 가능하여 **실시간 애플리케이션**에 적합
- RTX 3090/4070급 이상에서 실행 가능

### RoPE — 회전 위치 인코딩

FLUX는 [ViT](../09-vision-transformer/03-vit.md)의 고정된 위치 인코딩 대신 **RoPE(Rotary Position Embedding)**를 사용합니다. RoPE는 토큰의 상대적 위치 관계를 회전 행렬로 인코딩하는 방식으로, **다양한 해상도에 유연하게 대응**할 수 있다는 장점이 있습니다. 이것이 FLUX가 고정 해상도에 얽매이지 않는 이유입니다.

### Robin Rombach의 여정

FLUX의 탄생 배경에는 흥미로운 이야기가 있습니다. Robin Rombach는 2022년 Latent Diffusion Model(LDM) 논문으로 Stable Diffusion의 기반을 만든 장본인입니다. 이후 Stability AI에서 일하다가 2024년 독립하여 Black Forest Labs를 설립했죠. "Black Forest"라는 이름은 이들의 연구 기반인 독일 프라이부르크 근처의 **슈바르츠발트(Black Forest, 검은 숲)** 지역에서 따왔습니다. 자신이 만든 모델의 한계를 가장 잘 아는 사람이 그 한계를 넘는 새 모델을 만든 셈이죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "FLUX는 VRAM이 너무 많이 필요해서 못 쓴다" — FLUX.1-schnell은 `enable_model_cpu_offload()`을 사용하면 8GB VRAM에서도 실행 가능합니다. FLUX.2 [klein] 4B는 13GB면 충분하고요.

> 🔥 **실무 팁**: FLUX에서 텍스트 렌더링을 원하면 프롬프트에 **정확한 텍스트를 따옴표로 감싸세요**. 예: `A sign that says "OPEN 24/7"`. T5 인코더가 따옴표 안의 텍스트를 특별히 잘 인식합니다.

> 💡 **알고 계셨나요?** FLUX.1 [schnell]의 "schnell"은 독일어로 **"빠른"**이라는 뜻입니다. Black Forest Labs가 독일 회사라서 독일어 이름을 사용한 거죠. "Flux"도 물리학에서 **흐름/유동**을 의미하는데, Flow Matching 기반이라는 기술적 특성을 반영한 이름입니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **MM-DiT** | 텍스트와 이미지를 모두 토큰으로 처리하는 멀티모달 Transformer 아키텍처 |
| **이중/단일 스트림** | 19개 이중 스트림(모달리티별 독립) + 38개 단일 스트림(통합 처리)의 하이브리드 |
| **Rectified Flow** | 노이즈→이미지의 직선 경로를 학습하여 적은 스텝으로 효율적 생성 |
| **CLIP + T5** | CLIP(시각-언어 정렬) + T5-XXL(텍스트 이해)로 정확한 프롬프트 반영 |
| **Pro/Dev/Schnell** | 최고 품질 / 연구용 / 초고속(1~4스텝)의 세 변형 |
| **FLUX.2** | 320억 파라미터, 멀티 레퍼런스, 4MP 해상도, Klein(4B) 오픈소스 모델 |

## 다음 섹션 미리보기

다음 [SD3와 미래 방향](./06-sd3-future.md)에서는 Stability AI의 SD3/SD3.5와 FLUX를 비교하고, 이미지 생성 모델이 앞으로 어떤 방향으로 발전할지 전망합니다. 같은 MM-DiT 아키텍처를 공유하면서도 다른 선택을 한 두 모델의 비교가 흥미로울 거예요.

## 참고 자료

- [FLUX GitHub Repository](https://github.com/black-forest-labs/flux) - Black Forest Labs 공식 추론 코드
- [FLUX.1-dev on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev) - 모델 카드와 사용법
- [FLUX.1-schnell on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-schnell) - Apache 2.0 오픈소스 모델
- [Demystifying Flux Architecture (arXiv 2507.09595)](https://arxiv.org/abs/2507.09595) - FLUX 아키텍처 상세 분석 논문
- [How does Flux work? - Marcos V. Conde](https://medium.com/@drmarcosv/how-does-flux-work-the-new-image-generation-ai-that-rivals-midjourney-7f81f6f354da) - FLUX 작동 원리 해설
- [FLUX.2 공식 블로그](https://bfl.ai/blog/flux-2) - FLUX.2 시리즈 소개
- [Stable Diffusion 3 & FLUX: Complete Guide to MMDiT Architecture](https://blog.sotaaz.com/post/sd3-flux-architecture-en) - MM-DiT 아키텍처 상세 가이드
