# Diffusion 이론

> 전방/역방향 프로세스의 수학

## 개요

Diffusion 모델은 현재 이미지 생성 분야의 최강자입니다. Stable Diffusion, DALL-E, Midjourney 모두 이 원리를 기반으로 합니다. 이번 섹션에서는 Diffusion 모델의 핵심 직관과 수학적 기초인 **전방 프로세스(노이즈 추가)**와 **역방향 프로세스(노이즈 제거)**를 깊이 있게 다룹니다.

**선수 지식**: [생성 모델 개론](../11-generative-basics/01-generative-intro.md), [VAE](../11-generative-basics/02-vae.md), [손실 함수와 옵티마이저](../03-deep-learning-basics/04-loss-optimizer.md)
**학습 목표**:
- Diffusion 모델의 핵심 아이디어를 직관적으로 이해한다
- 전방 프로세스(Forward Process)의 수학적 정의를 파악한다
- 역방향 프로세스(Reverse Process)가 왜 가능한지 이해한다
- 노이즈 스케줄의 역할과 설계 원리를 안다

## 왜 알아야 할까?

[GAN](../11-generative-basics/03-gan-basics.md)이 "위조범과 감정사의 대결"이었다면, Diffusion은 완전히 다른 철학을 가지고 있습니다. "이미지를 천천히 파괴하는 과정을 학습하면, 그 반대로 이미지를 만들 수 있지 않을까?" — 이 단순한 아이디어가 GAN의 불안정한 학습, 모드 붕괴 문제를 한 번에 해결했죠.

2020년 DDPM 이후 불과 2년 만에, Diffusion 모델은 GAN을 제치고 이미지 생성의 주류가 되었습니다. Stable Diffusion, DALL-E 2, Midjourney, FLUX — 현재 우리가 사용하는 거의 모든 이미지 생성 도구가 Diffusion 기반이에요.

## 핵심 개념

### 개념 1: Diffusion의 핵심 직관 — 잉크 한 방울

> 💡 **비유**: 투명한 물컵에 잉크 한 방울을 떨어뜨려 보세요. 처음에는 또렷한 한 점이지만, 시간이 지나면 잉크가 물 전체에 **균일하게 퍼져** 원래 어디에 떨어졌는지 알 수 없게 됩니다. 이것이 **전방 프로세스**(노이즈 추가)입니다.

> 이제 마법 같은 질문을 해봅시다 — "균일하게 퍼진 잉크물을 보고, 원래 잉크가 어디에 떨어졌는지 **역추적**할 수 있을까?" 놀랍게도 Diffusion 모델은 이것을 해냅니다. 이것이 **역방향 프로세스**(노이즈 제거)이죠.

Diffusion 모델의 작동 원리를 두 단계로 요약하면:

1. **전방 프로세스 (Forward / Diffusion)**: 깨끗한 이미지에 가우시안 노이즈를 **점진적으로** 추가하여, 최종적으로 순수한 랜덤 노이즈로 만듦
2. **역방향 프로세스 (Reverse / Denoising)**: 순수한 노이즈에서 시작하여, 한 단계씩 노이즈를 **제거**해가며 깨끗한 이미지를 복원

> 깨끗한 이미지 $x_0$ → 약간 노이즈 $x_1$ → ... → 완전 노이즈 $x_T$

> 완전 노이즈 $x_T$ → 약간 정리 $x_{T-1}$ → ... → 깨끗한 이미지 $x_0$

### 개념 2: 전방 프로세스 — 체계적인 파괴

전방 프로세스는 **마르코프 체인(Markov Chain)**으로 정의됩니다. 각 단계에서 이전 상태에만 의존하여 노이즈를 추가하죠:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t \cdot I)$$

> - $x_t$: 시간 $t$에서의 노이즈가 추가된 이미지
> - $\beta_t$: 시간 $t$에서의 노이즈 강도 (노이즈 스케줄)
> - $\mathcal{N}$: 가우시안(정규) 분포

직관적으로 해석하면: "이전 이미지를 $\sqrt{1-\beta_t}$만큼 **축소**하고, $\beta_t$만큼의 **랜덤 노이즈**를 더한다"는 뜻입니다.

놀라운 성질이 하나 있습니다 — 중간 단계를 건너뛰고, **임의의 시점 $t$의 노이즈 이미지를 한 번에 계산**할 수 있어요:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \cdot I)$$

> - $\alpha_t = 1 - \beta_t$
> - $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ (누적 곱)

이를 **Reparameterization Trick**([VAE](../11-generative-basics/02-vae.md)에서 배운 것과 같은 아이디어!)으로 표현하면:

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

> 🔥 **실무 팁**: 이 "한 번에 점프" 공식 덕분에 학습 시 1000단계를 순서대로 밟을 필요 없이, 랜덤한 시점 $t$를 뽑아 바로 $x_t$를 만들 수 있습니다. 이것이 학습을 효율적으로 만드는 핵심이에요.

### 개념 3: 노이즈 스케줄 — 파괴의 속도 조절

$\beta_t$ 값의 시퀀스를 **노이즈 스케줄(Noise Schedule)**이라 합니다. 이것이 "얼마나 빠르게 이미지를 파괴할 것인가"를 결정하죠.

| 스케줄 | 설명 | 특징 |
|--------|------|------|
| **Linear** | $\beta_1$에서 $\beta_T$까지 일정하게 증가 | DDPM 원논문에서 사용, 간단 |
| **Cosine** | 코사인 함수 기반, 부드러운 전환 | 초반에 천천히, 후반에 빠르게 |
| **Scaled Linear** | Linear의 변형, 범위 조정 | Stable Diffusion에서 사용 |

일반적으로 $\beta_1 = 0.0001$ (매우 작은 노이즈)에서 시작하여 $\beta_T = 0.02$ (강한 노이즈)까지 증가합니다. 총 스텝 수 $T$는 보통 1000입니다.

> ⚠️ **흔한 오해**: "Diffusion 모델은 1000단계를 거쳐야 해서 느리다" — 학습 시에는 랜덤 시점 하나만 선택하므로 느리지 않습니다. 느린 것은 **생성(추론)** 단계인데, 이것도 [DDIM](./03-ddim.md) 같은 가속 기법으로 20~50 스텝으로 줄일 수 있습니다.

### 개념 4: 역방향 프로세스 — 학습된 복원

전방 프로세스가 "파괴"였다면, 역방향 프로세스는 "복원"입니다. 핵심은: **역방향의 각 단계도 가우시안 분포로 근사할 수 있다**는 것이죠.

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

> 신경망(파라미터 $\theta$)이 현재 노이즈 이미지 $x_t$와 시점 $t$를 보고, 한 단계 전의 이미지 $x_{t-1}$의 **평균 $\mu_\theta$**를 예측합니다.

여기서 핵심 통찰이 등장합니다 — $\mu_\theta$를 직접 예측하는 대신, **추가된 노이즈 $\epsilon$을 예측**하는 것이 더 잘 작동한다는 것입니다! 이것이 DDPM의 핵심 아이디어예요.

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

> 신경망 $\epsilon_\theta$가 해야 할 일은 단 하나: **"이 노이즈 이미지에 어떤 노이즈가 추가되었는지 맞춰라"**

### 개념 5: 왜 작동하는가 — 확률론적 직관

> 💡 **비유**: 1000겹의 얇은 안개를 하나씩 걷어내는 것을 상상해보세요. 한꺼번에 모든 안개를 걷어내는 것은 거의 불가능하지만, 한 겹씩 걷어내는 것은 각 단계가 **아주 작은 변화**이기 때문에 가능합니다.

Diffusion 모델이 작동하는 수학적 이유:
1. $\beta_t$가 충분히 작으면, 전방 프로세스의 역(reverse)도 가우시안으로 근사 가능
2. 각 단계의 변화량이 작아 신경망이 학습하기 쉬움
3. 1000단계를 거치면서 복잡한 데이터 분포를 **점진적으로** 구축

## 실습: 전방 프로세스 직접 구현하기

```python
import torch
import torch.nn.functional as F

class DiffusionSchedule:
    """Diffusion 노이즈 스케줄 및 전방 프로세스"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # Linear 노이즈 스케줄
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        # 누적 곱: ᾱ_t = α_1 × α_2 × ... × α_t
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_process(self, x_0, t, noise=None):
        """
        전방 프로세스: 깨끗한 이미지 x_0에 시점 t만큼의 노이즈 추가
        x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 시점 t에 해당하는 ᾱ_t 값 가져오기
        alpha_bar_t = self.alpha_cumprod[t]
        # 배치 차원에 맞게 reshape
        while alpha_bar_t.dim() < x_0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        # 전방 프로세스 공식 적용
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise


# 전방 프로세스 테스트
schedule = DiffusionSchedule(num_timesteps=1000)

# 가상의 깨끗한 이미지 (배치 4, 3채널, 32x32)
x_0 = torch.randn(4, 3, 32, 32)

# 다양한 시점에서의 노이즈 추가
for t_val in [0, 250, 500, 750, 999]:
    t = torch.tensor([t_val] * 4)
    x_t, noise = schedule.forward_process(x_0, t)
    alpha_bar = schedule.alpha_cumprod[t_val].item()
    signal_ratio = alpha_bar
    noise_ratio = 1 - alpha_bar
    print(f"t={t_val:4d} | ᾱ_t={alpha_bar:.4f} | "
          f"신호 비율={signal_ratio:.1%} | 노이즈 비율={noise_ratio:.1%}")

# 출력 예시:
# t=   0 | ᾱ_t=0.9999 | 신호 비율=100.0% | 노이즈 비율=0.0%
# t= 250 | ᾱ_t=0.7647 | 신호 비율=76.5% | 노이즈 비율=23.5%
# t= 500 | ᾱ_t=0.4180 | 신호 비율=41.8% | 노이즈 비율=58.2%
# t= 750 | ᾱ_t=0.1140 | 신호 비율=11.4% | 노이즈 비율=88.6%
# t= 999 | ᾱ_t=0.0047 | 신호 비율=0.5% | 노이즈 비율=99.5%
```

> 💡 **알고 계셨나요?**: t=0에서는 거의 원본이고, t=999에서는 99.5%가 노이즈입니다. 이 부드러운 전환이 Diffusion 모델이 학습할 수 있는 비결이에요. 각 단계의 변화가 너무 크지 않아서 신경망이 "이 시점에서 어떤 노이즈가 추가되었는지"를 충분히 학습할 수 있죠.

## 더 깊이 알아보기

### Diffusion의 물리학적 기원

"Diffusion"이라는 이름은 물리학에서 온 것입니다. 1827년 식물학자 Robert Brown이 물 위에 떠 있는 꽃가루 입자가 무작위로 움직이는 것을 관찰했고, 이것을 **브라운 운동(Brownian Motion)**이라 불렀습니다. 1905년 아인슈타인이 이를 수학적으로 설명했죠.

Diffusion 모델의 전방 프로세스는 바로 이 브라운 운동과 같은 원리입니다 — 데이터에 랜덤 노이즈를 점진적으로 추가하는 것은 입자가 확산(diffusion)하는 과정과 수학적으로 동일해요. 2015년 Sohl-Dickstein 등이 이 물리적 직관을 생성 모델에 처음 적용했고, 2020년 Ho 등의 DDPM이 이를 실용적인 수준으로 끌어올렸습니다.

### 확률 미분 방정식(SDE) 관점

Diffusion 모델은 **확률 미분 방정식(Stochastic Differential Equation, SDE)**의 이산화(discretization)로 이해할 수 있습니다. Song et al.(2021)의 "Score SDE" 논문은 DDPM, Score Matching 등 다양한 생성 모델을 SDE라는 통합 프레임워크로 설명했죠. 이 관점에서 전방 프로세스는 SDE, 역방향 프로세스는 그 역 SDE에 해당합니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Diffusion은 VAE나 GAN과 완전히 다른 모델이다" — 사실 깊은 연결이 있습니다. VAE의 인코더-디코더 구조는 Diffusion의 전방-역방향과 대응되고, Score Matching은 GAN의 판별자와 유사한 역할을 합니다. [VAE](../11-generative-basics/02-vae.md)에서 배운 ELBO(Evidence Lower Bound) 역시 Diffusion의 손실 함수 유도에 핵심적으로 사용됩니다.

> 🔥 **실무 팁**: 노이즈 스케줄 선택이 생성 품질에 큰 영향을 미칩니다. Cosine 스케줄은 Linear보다 초반 단계에서 정보를 더 오래 보존하므로, 일반적으로 더 좋은 결과를 냅니다. Stable Diffusion은 Scaled Linear를 사용합니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 전방 프로세스 | 이미지에 점진적으로 노이즈를 추가하는 과정 |
| 역방향 프로세스 | 노이즈에서 이미지를 점진적으로 복원하는 과정 |
| $\beta_t$ | 시점 $t$에서의 노이즈 강도 (노이즈 스케줄) |
| $\bar{\alpha}_t$ | 누적 신호 보존 비율 ($\alpha_1 \times ... \times \alpha_t$) |
| 노이즈 예측 | 신경망이 추가된 노이즈 $\epsilon$을 예측하는 것이 핵심 |
| 마르코프 체인 | 각 단계가 직전 단계에만 의존하는 확률 과정 |

## 다음 섹션 미리보기

이론적 기초를 다졌으니, 다음 섹션 [DDPM](./02-ddpm.md)에서는 이 이론을 실제로 구현한 첫 번째 실용적 모델을 만나봅니다. 학습 알고리즘, 손실 함수, 그리고 실제 이미지를 생성하는 전체 파이프라인을 PyTorch로 직접 구현해볼 거예요.

## 참고 자료

- [Ho et al., "Denoising Diffusion Probabilistic Models" (2020)](https://arxiv.org/abs/2006.11239) - DDPM 원논문
- [Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015)](https://arxiv.org/abs/1503.03585) - Diffusion 모델의 시초
- [Lilian Weng, "What are Diffusion Models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - 최고의 Diffusion 모델 해설
- [Song et al., "Score-Based Generative Modeling through SDE" (2021)](https://arxiv.org/abs/2011.13456) - SDE 통합 프레임워크
