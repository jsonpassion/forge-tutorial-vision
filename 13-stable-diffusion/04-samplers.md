# 샘플러 가이드

> Euler, DPM++, UniPC 비교

## 개요

Stable Diffusion에서 이미지를 생성할 때 반드시 선택해야 하는 것이 **샘플러(Sampler)**입니다. [DDIM과 샘플링 가속](../12-diffusion-models/03-ddim.md)에서 이론을 배웠는데, 이번 섹션에서는 실전에서 자주 사용하는 샘플러들의 특성과 최적 설정을 정리합니다.

**선수 지식**: [DDIM과 샘플링 가속](../12-diffusion-models/03-ddim.md), [SD 아키텍처](./01-sd-architecture.md)
**학습 목표**:
- 주요 샘플러의 특성과 차이를 이해한다
- 스텝 수와 품질의 관계를 파악한다
- 상황별 최적의 샘플러를 선택할 수 있다
- Karras 노이즈 스케줄의 효과를 안다

## 왜 알아야 할까?

같은 프롬프트, 같은 시드라도 샘플러에 따라 결과가 크게 달라집니다. 적절한 샘플러를 선택하면 더 적은 스텝으로 더 좋은 결과를 얻을 수 있어요.

## 핵심 개념

### 개념 1: 주요 샘플러 분류

> 💡 **비유**: 샘플러는 **등산 하산 방법**과 같습니다. 같은 산 정상(노이즈)에서 같은 계곡(이미지)으로 내려가지만, 경로가 다른 거죠. 지그재그(Euler)로 갈 수도 있고, 최적 경로(DPM-Solver)를 계산하여 직선에 가깝게 갈 수도 있습니다.

| 샘플러 | 유형 | 스텝 | 특징 |
|--------|------|------|------|
| **Euler** | 1차 ODE | 20~30 | 가장 기본, 안정적 |
| **Euler a** | 1차 SDE | 20~40 | Ancestral (확률적), 다양성 높음 |
| **Heun** | 2차 ODE | 15~25 | Euler보다 정확, 2배 느림 |
| **DPM++ 2M** | 2차 ODE | 20~30 | **가장 추천**, 빠르고 고품질 |
| **DPM++ 2M Karras** | 2차 ODE | 20~30 | Karras 스케줄 적용, **최고 인기** |
| **DPM++ SDE** | SDE | 20~30 | 확률적, 디테일 풍부 |
| **DPM++ 2M SDE Karras** | SDE | 20~30 | SDE + Karras 조합 |
| **UniPC** | 다차 | 10~20 | 매우 빠름, 적은 스텝에서 강함 |
| **LCM** | 특수 | 4~8 | 초고속, 별도 학습 필요 |

### 개념 2: ODE vs SDE — 결정론적 vs 확률적

[DDIM](../12-diffusion-models/03-ddim.md)에서 배운 개념의 실전 적용:

**ODE 샘플러** (Euler, DPM++ 2M 등):
- 결정론적 — 같은 시드에서 항상 같은 결과
- 일반적으로 더 깨끗한 결과
- 재현성이 중요할 때 선호

**SDE 샘플러** (Euler a, DPM++ SDE 등):
- 확률적 — 같은 시드에서도 약간 다른 결과
- 더 많은 디테일과 텍스처
- 예술적 변화를 원할 때 유용

**"a"가 붙은 샘플러** (Euler a, DPM++ 2S a):
- "Ancestral" — 각 스텝에서 랜덤 노이즈를 다시 추가
- **수렴하지 않음** — 스텝을 늘려도 결과가 계속 변함

> ⚠️ **흔한 오해**: "스텝을 많이 할수록 항상 좋다" — ODE 샘플러는 일정 스텝 이상에서 **수렴**하여 더 이상 변하지 않지만, Ancestral 샘플러는 수렴하지 않고 계속 변합니다. DPM++ 2M Karras는 20 스텝이면 대부분 충분해요.

### 개념 3: Karras 노이즈 스케줄

"Karras"가 붙은 샘플러는 Tero Karras([StyleGAN](../11-generative-basics/04-gan-variants.md)의 창시자!)가 제안한 노이즈 스케줄을 사용합니다:

- **Linear 스케줄**: 스텝 간격이 균등
- **Karras 스케줄**: 마지막 단계(낮은 노이즈)에 더 많은 스텝 할당

Karras 스케줄은 이미지의 **세밀한 디테일** 단계에 더 많은 연산을 투자하므로, 같은 스텝 수에서 더 좋은 결과를 냅니다. 대부분의 상황에서 Karras 버전을 사용하는 것이 좋아요.

### 개념 4: 실전 추천 설정

**일반적인 상황 (SD 1.5 / SDXL)**:
- 샘플러: **DPM++ 2M Karras**
- 스텝: **20~25**
- CFG Scale: 7~8

**속도 우선**:
- 샘플러: **UniPC** 또는 **LCM**
- 스텝: 10~15 (UniPC) / 4~8 (LCM)
- CFG Scale: 5~7

**최고 품질**:
- 샘플러: **DPM++ 2M SDE Karras**
- 스텝: 25~35
- CFG Scale: 7~8

**FLUX 모델**:
- 샘플러: **Euler** (FLUX에 최적화)
- 스텝: 20~28
- CFG Scale: 3.5 (FLUX는 낮은 CFG 권장)

## 실습: 샘플러별 비교

```python
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompt = "a fantasy castle on a floating island, dramatic lighting"
seed = 42

# 샘플러별 비교
schedulers = {
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M Karras": DPMSolverMultistepScheduler,
    "UniPC": UniPCMultistepScheduler,
}

for name, SchedulerClass in schedulers.items():
    pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config)
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    image.save(f"castle_{name.replace(' ', '_')}.png")
    print(f"{name}: 20 스텝으로 생성 완료")
```

## 더 깊이 알아보기

### LCM과 Turbo — 초고속 생성의 시대

**LCM (Latent Consistency Model, 2023)**: [Consistency Models](../12-diffusion-models/03-ddim.md)의 아이디어를 Latent Diffusion에 적용하여, **4~8 스텝**으로 이미지를 생성합니다. 별도의 학습(Distillation)이 필요하지만, 실시간에 가까운 생성이 가능해요.

**SDXL Turbo (2023)**: Adversarial Diffusion Distillation으로 **1~4 스텝**만에 이미지를 생성합니다. 실시간 이미지 편집의 가능성을 열었죠.

## 흔한 오해와 팁

> 💡 **알고 계셨나요?**: "DPM++"의 "DPM"은 Diffusion Probabilistic Model의 약자이고, "++"는 가이던스가 있는 확장 버전을 의미합니다. 칭화대학교의 Lu Cheng 팀이 개발했으며, 수학적으로는 지수 적분(exponential integrator)을 활용한 고차 ODE 해법입니다.

> 🔥 **실무 팁**: 새로운 프롬프트를 실험할 때는 먼저 **Euler 20 스텝**으로 빠르게 여러 시드를 테스트한 뒤, 마음에 드는 시드를 찾으면 **DPM++ 2M Karras 25 스텝**으로 고품질 생성하세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| ODE 샘플러 | 결정론적, 수렴함 (Euler, DPM++ 2M) |
| SDE 샘플러 | 확률적, 다양성 높음 (Euler a, DPM++ SDE) |
| Karras 스케줄 | 마지막 단계에 더 많은 스텝 할당, 디테일 향상 |
| DPM++ 2M Karras | 실전 가장 인기 있는 범용 샘플러 |
| LCM/Turbo | 4~8 스텝 초고속 생성, 별도 학습 필요 |

## 다음 섹션 미리보기

SD 생태계를 넘어, 다음 섹션 [FLUX 모델](./05-flux.md)에서는 Stable Diffusion의 창시자들이 만든 차세대 모델을 만나봅니다. 120억 파라미터의 Diffusion Transformer가 어떻게 기존의 한계를 뛰어넘었는지 알아볼까요?

## 참고 자료

- [Lu et al., "DPM-Solver++" (2022)](https://arxiv.org/abs/2211.01095) - DPM-Solver++ 논문
- [Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (2022)](https://arxiv.org/abs/2206.00364) - Karras 스케줄 논문
- [Luo et al., "Latent Consistency Models" (2023)](https://arxiv.org/abs/2310.04378) - LCM 논문
- [Stable Diffusion Samplers Guide](https://stable-diffusion-art.com/samplers/) - 샘플러 실전 가이드
