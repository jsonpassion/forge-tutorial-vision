# Classifier-Free Guidance

> 조건부 생성과 가이던스 스케일

## 개요

"고양이 사진을 그려줘"라고 했을 때, Diffusion 모델이 정말 고양이를 그리게 만드는 핵심 기술이 바로 **Classifier-Free Guidance(CFG)**입니다. 이 기법 없이는 텍스트와 무관한 이미지가 생성될 수 있어요. 이번 섹션에서는 CFG의 원리와 가이던스 스케일이 생성 품질에 미치는 영향을 다룹니다.

**선수 지식**: [DDPM](./02-ddpm.md), [U-Net 아키텍처](./04-unet-architecture.md)
**학습 목표**:
- 조건부(Conditional) Diffusion 모델의 개념을 이해한다
- Classifier Guidance와 Classifier-Free Guidance의 차이를 파악한다
- 가이던스 스케일($w$)이 생성 결과에 미치는 영향을 안다
- CFG의 학습과 추론 방식을 이해한다

## 왜 알아야 할까?

Stable Diffusion에서 "Guidance Scale" 또는 "CFG Scale"이라는 슬라이더를 보셨을 거예요. 이 값을 7.5로 할지 15로 할지에 따라 결과가 크게 달라지죠. 이 슬라이더의 정체가 바로 CFG입니다. 원리를 알면 최적의 값을 선택할 수 있게 됩니다.

## 핵심 개념

### 개념 1: 조건부 생성의 필요성

> 💡 **비유**: 기본 DDPM은 **눈을 감고 그림을 그리는 화가**와 같습니다. 무언가를 만들어내긴 하지만, 원하는 것을 지정할 수 없죠. CFG는 이 화가에게 **귀를 열어주는 것** — "빨간 자동차를 그려줘"라는 말을 듣고 그에 맞게 그릴 수 있게 됩니다.

[DDPM](./02-ddpm.md)에서 학습한 모델은 **비조건부(Unconditional)** 모델입니다. 랜덤 노이즈에서 이미지를 만들지만, "어떤" 이미지를 만들지 제어할 수 없어요.

**조건부 모델**은 텍스트, 클래스, 이미지 등의 **조건 $c$**를 추가로 입력받아, 그 조건에 맞는 이미지를 생성합니다:

$$\epsilon_\theta(x_t, t, c) \quad \text{← 조건 } c \text{가 추가됨!}$$

### 개념 2: Classifier Guidance — 분류기의 힘을 빌리기

CFG 이전에는 **Classifier Guidance**라는 방법이 있었습니다:

1. Diffusion 모델과 별도로 **이미지 분류기(Classifier)**를 학습
2. 생성 과정에서 분류기의 그래디언트를 이용하여 원하는 클래스 방향으로 유도

$$\hat{\epsilon}(x_t, t) = \epsilon_\theta(x_t, t) - w \cdot \sigma_t \nabla_{x_t} \log p_\phi(c | x_t)$$

문제점: 별도의 분류기가 필요하고, 노이즈 이미지에 대해서도 작동하는 특수한 분류기를 학습해야 합니다. 번거롭죠.

### 개념 3: Classifier-Free Guidance — 분류기 없이!

> 💡 **비유**: Classifier Guidance가 **외부 코치**를 고용하여 방향을 잡아주는 것이라면, CFG는 화가 자신이 "내가 아무 생각 없이 그리면 어떻게 되고, 고양이를 생각하며 그리면 어떻게 되는지" 비교하여 **스스로 방향을 잡는 것**입니다.

Ho & Salimans(2022)의 핵심 아이디어: 분류기 대신 **조건부/비조건부 예측의 차이**를 이용하자!

$$\hat{\epsilon}(x_t, t, c) = \underbrace{\epsilon_\theta(x_t, t, \varnothing)}_{\text{비조건부 예측}} + w \cdot \left(\underbrace{\epsilon_\theta(x_t, t, c)}_{\text{조건부 예측}} - \underbrace{\epsilon_\theta(x_t, t, \varnothing)}_{\text{비조건부 예측}}\right)$$

정리하면:

$$\hat{\epsilon} = (1 - w) \cdot \epsilon_\theta(x_t, t, \varnothing) + w \cdot \epsilon_\theta(x_t, t, c)$$

> - $\epsilon_\theta(x_t, t, c)$: 조건(텍스트)을 반영한 노이즈 예측
> - $\epsilon_\theta(x_t, t, \varnothing)$: 조건 없이 예측한 노이즈 (빈 텍스트)
> - $w$: **가이던스 스케일** — 조건을 얼마나 강하게 반영할지

직관적으로: "조건이 있을 때와 없을 때의 차이를 $w$배 증폭한다"는 의미입니다.

### 개념 4: 가이던스 스케일의 효과

$w$ 값에 따라 생성 결과가 극적으로 달라집니다:

| $w$ 값 | 효과 | 특징 |
|--------|------|------|
| $w = 1$ | 가이던스 없음 | 다양하지만 텍스트와 무관할 수 있음 |
| $w = 3\sim5$ | 약한 가이던스 | 자연스럽고 다양성 높음 |
| $w = 7\sim8$ | 표준 가이던스 | **품질과 텍스트 일치의 균형** |
| $w = 12\sim15$ | 강한 가이던스 | 텍스트에 매우 충실, 과포화 경향 |
| $w > 20$ | 과도한 가이던스 | 색상 과포화, 아티팩트 발생 |

> 🔥 **실무 팁**: Stable Diffusion에서 기본값 7.5는 대부분의 상황에서 잘 작동합니다. 사진 같은 리얼리즘을 원하면 5~7, 만화/일러스트 스타일이면 8~12가 적합해요. SDXL에서는 5~8 범위를 추천합니다.

### 개념 5: CFG의 학습 — 조건 드롭아웃

CFG를 쓰려면 하나의 모델이 조건부와 비조건부 예측을 **둘 다** 할 수 있어야 합니다. 이를 위해 학습 시 **조건 드롭아웃(Condition Dropout)**을 사용합니다:

- 학습 데이터의 약 10~20%에서 조건 $c$를 **빈 값 $\varnothing$**으로 대체
- 모델이 조건이 있을 때와 없을 때를 모두 학습하게 됨

이것은 [Dropout 정규화](../04-cnn-fundamentals/04-regularization.md)와 비슷한 아이디어예요. 일부를 의도적으로 가려서 모델의 능력을 다양화시키는 거죠.

> ⚠️ **흔한 오해**: "CFG는 추론 시에만 적용된다" — 추론 시 가이던스 스케일 $w$를 적용하는 것은 맞지만, **학습 시에도 조건 드롭아웃이 필수**입니다. 드롭아웃 없이 학습하면 비조건부 예측을 할 수 없어서 CFG가 작동하지 않아요.

## 실습: CFG가 적용된 샘플링

```python
import torch
import torch.nn.functional as F

class CFGSampler:
    """Classifier-Free Guidance가 적용된 DDIM 샘플러"""
    def __init__(self, model, alpha_cumprod, device='cpu'):
        self.model = model
        self.alpha_cumprod = alpha_cumprod.to(device)
        self.device = device

    @torch.no_grad()
    def sample(self, shape, condition, guidance_scale=7.5,
               num_steps=50, null_condition=None):
        """
        CFG 샘플링
        - condition: 텍스트 임베딩 (B, seq_len, dim)
        - guidance_scale: 가이던스 스케일 w
        - null_condition: 빈 텍스트의 임베딩 (비조건부용)
        """
        # 순수 노이즈에서 시작
        x = torch.randn(shape, device=self.device)

        # 서브시퀀스 (DDIM 스타일)
        step_size = 1000 // num_steps
        timesteps = list(range(999, -1, -step_size))[:num_steps]

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)

            # ====== CFG의 핵심 ======
            # 1. 조건부 예측: 텍스트 조건을 넣어서 예측
            noise_cond = self.model(x, t_batch, condition)

            # 2. 비조건부 예측: 빈 텍스트로 예측
            noise_uncond = self.model(x, t_batch, null_condition)

            # 3. 가이던스 적용: 두 예측의 차이를 w배 증폭
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            # ========================

            # DDIM 업데이트
            alpha_bar_t = self.alpha_cumprod[t]
            alpha_bar_prev = (self.alpha_cumprod[timesteps[i+1]]
                              if i + 1 < len(timesteps)
                              else torch.tensor(1.0, device=self.device))

            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            direction = torch.sqrt(1 - alpha_bar_prev) * noise_pred
            x = torch.sqrt(alpha_bar_prev) * x0_pred + direction

        return x


# 가이던스 스케일 비교
print("CFG Scale 비교:")
print(f"  w=1.0  → 가이던스 없음 (다양하지만 텍스트 무시)")
print(f"  w=7.5  → 표준 (품질과 텍스트 일치의 균형)")
print(f"  w=15.0 → 강한 가이던스 (텍스트 충실, 과포화 주의)")
print(f"  w=30.0 → 과도 (아티팩트 발생)")
```

## 더 깊이 알아보기

### Negative Prompt의 원리

Stable Diffusion의 **네거티브 프롬프트(Negative Prompt)**가 작동하는 원리도 CFG입니다:

$$\hat{\epsilon} = \epsilon_\theta(x_t, t, c_{neg}) + w \cdot (\epsilon_\theta(x_t, t, c_{pos}) - \epsilon_\theta(x_t, t, c_{neg}))$$

비조건부 예측 $\epsilon(x_t, t, \varnothing)$ 대신 **네거티브 프롬프트의 예측** $\epsilon(x_t, t, c_{neg})$를 사용하는 거예요. "blurry, low quality"를 네거티브로 넣으면, 그 방향에서 **멀어지도록** 가이드하는 셈이죠.

### CFG의 발견 — Jonathan Ho의 또 다른 기여

CFG를 제안한 사람은 DDPM의 저자인 Jonathan Ho입니다! DDPM(2020) 이후 2년 만에 CFG(2022)를 발표했죠. 이 두 기여만으로도 현대 이미지 생성의 근간을 놓았다고 할 수 있습니다. CFG 논문은 간결하면서도 강력한 아이디어를 담고 있어, "좋은 연구란 단순해야 한다"는 교훈을 줍니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "CFG Scale을 높이면 항상 좋다" — 아닙니다! Scale이 너무 높으면 색상이 과포화되고, 아티팩트가 생기며, 다양성이 사라집니다. 7~8이 대부분의 경우 최적입니다.

> 💡 **알고 계셨나요?**: CFG는 추론 시 **2번의 forward pass**가 필요합니다 (조건부 1번 + 비조건부 1번). 이는 생성 속도를 2배 느리게 만들죠. 이를 해결하기 위해 배치 차원으로 합쳐서 한 번에 처리하는 기법이 사용됩니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| Classifier Guidance | 별도의 분류기 그래디언트로 생성 방향 유도 |
| Classifier-Free Guidance | 조건부/비조건부 예측의 차이를 증폭하여 유도 |
| 가이던스 스케일 $w$ | 조건 반영 강도 (7~8이 표준) |
| 조건 드롭아웃 | 학습 시 10~20% 확률로 조건을 제거 |
| 네거티브 프롬프트 | 비조건부 대신 원치 않는 조건을 사용하여 회피 |

## 다음 섹션 미리보기

CFG로 "무엇을" 생성할지 제어하는 법을 배웠습니다. 다음 섹션 [Latent Diffusion](./06-latent-diffusion.md)에서는 "어디서" Diffusion을 수행할지의 혁신을 다룹니다. 픽셀 공간이 아닌 **잠재 공간**에서 Diffusion을 수행함으로써, 일반 GPU로도 고해상도 이미지를 생성할 수 있게 된 핵심 아이디어를 알아봅시다.

## 참고 자료

- [Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)](https://arxiv.org/abs/2207.12598) - CFG 원논문
- [Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021)](https://arxiv.org/abs/2105.05233) - Classifier Guidance 원논문
- [Classifier Free Diffusion Guidance (Blog)](https://mbottoni.github.io/2024/12/15/cfg.html) - CFG의 직관적 해설
- [Stable Diffusion with Diffusers (HuggingFace)](https://huggingface.co/blog/stable_diffusion) - CFG 실전 적용
