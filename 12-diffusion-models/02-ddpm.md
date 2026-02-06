# DDPM

> Denoising Diffusion Probabilistic Models

## 개요

DDPM(Denoising Diffusion Probabilistic Models)은 [Diffusion 이론](./01-diffusion-theory.md)을 처음으로 실용적인 수준으로 구현한 모델입니다. 2020년 Jonathan Ho가 발표한 이 논문은 "노이즈를 예측하라"는 단순한 아이디어로 GAN에 필적하는 이미지 생성 품질을 달성했죠. 이번 섹션에서는 DDPM의 학습 과정과 생성 과정을 완전히 이해하고, PyTorch로 구현합니다.

**선수 지식**: [Diffusion 이론](./01-diffusion-theory.md), [역전파 알고리즘](../03-deep-learning-basics/03-backpropagation.md)
**학습 목표**:
- DDPM의 학습 알고리즘(노이즈 예측)을 완전히 이해한다
- 단순화된 손실 함수가 왜 효과적인지 파악한다
- DDPM의 생성(샘플링) 과정을 단계별로 이해한다
- PyTorch로 DDPM 학습/생성 파이프라인을 구현할 수 있다

## 왜 알아야 할까?

DDPM은 현대 이미지 생성의 **근본 알고리즘**입니다. Stable Diffusion, DALL-E 2, Midjourney 모두 DDPM의 학습 방식을 기반으로 합니다. DDPM을 이해하면 이후 모든 Diffusion 모델 변형을 쉽게 따라갈 수 있어요. 그리고 놀라울 정도로 — 핵심 알고리즘이 간단합니다.

## 핵심 개념

### 개념 1: DDPM의 학습 — 놀랍도록 간단한 알고리즘

> 💡 **비유**: 미술 선생님이 학생에게 "이 그림에 내가 뿌린 물감 얼룩을 찾아내봐"라고 하는 것과 같습니다. 학생은 다양한 수준의 얼룩(약간 ~ 많이)을 보면서 "어떤 얼룩이 추가되었는지"를 맞추는 연습을 반복합니다. 충분히 연습하면 어떤 수준의 얼룩도 정확히 찾아낼 수 있게 되죠.

DDPM 학습 알고리즘은 놀라울 정도로 간단합니다:

**학습 루프 (한 스텝)**:
1. 학습 데이터에서 깨끗한 이미지 $x_0$를 하나 가져온다
2. 랜덤 시점 $t \sim \text{Uniform}(1, T)$를 뽑는다
3. 랜덤 노이즈 $\epsilon \sim \mathcal{N}(0, I)$를 생성한다
4. 노이즈 이미지를 만든다: $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$
5. 신경망에 $(x_t, t)$를 넣어 노이즈를 예측한다: $\epsilon_\theta(x_t, t)$
6. 실제 노이즈와 예측의 차이를 **MSE**로 최소화한다

**손실 함수**:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right]$$

> 그냥 "추가한 노이즈"와 "예측한 노이즈"의 MSE입니다. 이것이 전부예요!

> 💡 **알고 계셨나요?**: DDPM 논문의 원래 손실 함수는 변분 하한(Variational Lower Bound, VLB)에서 유도된 복잡한 형태였습니다. 하지만 Ho et al.은 이 단순화된 MSE 손실이 실제로는 **더 좋은 결과**를 낸다는 것을 실험적으로 발견했죠. 복잡한 수학이 단순함에 패한 드문 사례입니다.

### 개념 2: DDPM의 생성 — 한 걸음씩 노이즈 제거

학습된 모델로 이미지를 생성하는 과정:

**생성 알고리즘**:
1. 순수한 노이즈에서 시작: $x_T \sim \mathcal{N}(0, I)$
2. $t = T$부터 $t = 1$까지 반복:
   - 노이즈를 예측: $\epsilon_\theta(x_t, t)$
   - 한 단계 디노이징: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$
   - 여기서 $z \sim \mathcal{N}(0, I)$는 확률적 요소 (마지막 스텝에서는 $z = 0$)
3. 최종 $x_0$가 생성된 이미지

> ⚠️ **흔한 오해**: "DDPM은 1000 스텝을 반드시 거쳐야 한다" — DDPM의 원래 설정은 $T=1000$이지만, 이후 [DDIM](./03-ddim.md)이나 DPM-Solver 같은 가속 기법을 쓰면 20~50 스텝으로 줄일 수 있습니다.

### 개념 3: 시간 임베딩 — 모델에게 "지금 몇 단계인지" 알려주기

> 💡 **비유**: 안개 속에서 길을 찾는다고 생각해보세요. "안개가 10겹 남았는지, 2겹 남았는지"에 따라 전략이 달라지겠죠. 신경망에게도 현재 몇 번째 디노이징 단계인지 알려줘야 합니다.

DDPM은 시점 정보 $t$를 **사인/코사인 위치 인코딩**(Transformer에서 차용!)으로 임베딩합니다:

$$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

이것은 [Transformer](../09-vision-transformer/02-transformer-basics.md)의 위치 인코딩과 동일한 형태입니다! 다른 시점의 노이즈에 다르게 대응할 수 있도록 모델에 시간 정보를 주입하는 거죠.

### 개념 4: 예측 대상의 선택 — 노이즈 vs 원본 vs 속도

DDPM은 **노이즈 $\epsilon$**을 예측하지만, 다른 선택지도 있습니다:

| 예측 대상 | 수식 | 특징 |
|----------|------|------|
| **노이즈 $\epsilon$** | $\epsilon_\theta(x_t, t) \approx \epsilon$ | DDPM 기본, 높은 시점에서 안정적 |
| **원본 $x_0$** | $x_\theta(x_t, t) \approx x_0$ | 낮은 시점에서 안정적 |
| **속도 $v$** | $v_\theta(x_t, t) \approx \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$ | 두 장점의 균형, Stable Diffusion v2에서 사용 |

세 가지 모두 수학적으로 동등하지만(서로 변환 가능), 학습 안정성과 생성 품질에서 차이가 납니다.

## 실습: DDPM 학습 및 샘플링 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
    """시간 임베딩 — Transformer의 위치 인코딩과 동일한 원리"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SimpleNoisePredictor(nn.Module):
    """간소화된 노이즈 예측 네트워크 (실제로는 U-Net 사용)"""
    def __init__(self, img_channels=1, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )
        # 간단한 CNN 구조 (실제 DDPM은 U-Net을 사용)
        self.conv1 = nn.Conv2d(img_channels, 64, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, 128)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.norm3 = nn.GroupNorm(8, 64)
        self.conv_out = nn.Conv2d(64, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, 128)

    def forward(self, x, t):
        # 시간 임베딩 생성 및 프로젝션
        t_emb = self.time_mlp(t)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        # 노이즈 예측
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.conv2(h)
        h = h + t_proj                               # 시간 정보 주입!
        h = F.gelu(self.norm2(h))
        h = F.gelu(self.norm3(self.conv3(h)))
        return self.conv_out(h)                       # 예측된 노이즈


class DDPM:
    """DDPM 학습 및 샘플링"""
    def __init__(self, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.model = model
        self.T = num_timesteps
        self.device = device

        # 노이즈 스케줄 계산
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def train_step(self, x_0, optimizer):
        """DDPM 학습 한 스텝 — 놀라울 정도로 간단!"""
        batch_size = x_0.size(0)

        # 1. 랜덤 시점 선택
        t = torch.randint(0, self.T, (batch_size,), device=self.device)

        # 2. 랜덤 노이즈 생성
        noise = torch.randn_like(x_0)

        # 3. 노이즈 이미지 생성 (전방 프로세스)
        alpha_bar_t = self.alpha_cumprod[t][:, None, None, None]
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        # 4. 노이즈 예측
        noise_pred = self.model(x_t, t)

        # 5. MSE 손실 — 이것이 전부!
        loss = F.mse_loss(noise_pred, noise)

        # 6. 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, shape):
        """DDPM 샘플링 — T 단계에 걸쳐 디노이징"""
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t_batch)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_cumprod[t]
            beta_t = self.betas[t]

            # 디노이징 한 스텝
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
            )
            if t > 0:
                x = x + torch.sqrt(beta_t) * torch.randn_like(x)

        return x


# 테스트
device = 'cpu'
model = SimpleNoisePredictor(img_channels=1).to(device)
ddpm = DDPM(model, num_timesteps=1000, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# 학습 테스트
dummy = torch.randn(8, 1, 28, 28).to(device)
loss = ddpm.train_step(dummy, optimizer)
print(f"학습 손실: {loss:.4f}")
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
```

## 더 깊이 알아보기

### Jonathan Ho의 DDPM — 생성 모델의 패러다임 전환

2020년, Google Brain의 Jonathan Ho는 "Denoising Diffusion Probabilistic Models" 논문을 발표했습니다. 사실 Diffusion 기반 생성 모델의 아이디어는 2015년 Sohl-Dickstein에 의해 처음 제안되었지만, 당시에는 GAN에 비해 결과가 크게 뒤처졌죠.

Ho의 핵심 기여는 **단순화**였습니다. 복잡한 변분 추론 대신 "노이즈를 예측하라"는 간단한 목표 함수를 설정하고, 이것이 실제로 매우 잘 작동한다는 것을 보여줬어요. 이 논문은 불과 2년 만에 GAN을 제치고 이미지 생성의 주류 패러다임을 바꿔놓았습니다.

### 왜 MSE가 작동하는가 — 수학적 배경

단순한 MSE 손실이 효과적인 이유는 **변분 하한(ELBO)**과의 연결에 있습니다:

$$\mathcal{L}_{VLB} = \sum_{t=1}^{T} D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$$

이 KL 다이버전스를 전개하면, 각 항이 노이즈 예측의 MSE와 비례합니다. 즉, 노이즈 MSE를 최소화하는 것은 데이터의 로그 우도를 최대화하는 것과 수학적으로 동등해요. [VAE](../11-generative-basics/02-vae.md)에서 배운 ELBO가 여기서도 핵심적으로 등장하는 거죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "DDPM의 학습에 1000 스텝이 필요하다" — 학습 시에는 **1 스텝만** 필요합니다! 랜덤 시점 $t$를 하나 뽑아 그 시점의 노이즈만 예측하면 됩니다. 1000 스텝이 필요한 것은 **생성(추론)** 때입니다.

> 🔥 **실무 팁**: DDPM 학습 시 `lr=2e-4`, Adam 옵티마이저가 표준 설정입니다. EMA(Exponential Moving Average)를 적용한 모델이 더 안정적인 생성 결과를 냅니다. decay rate 0.9999가 일반적이에요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 학습 목표 | 추가된 노이즈 $\epsilon$을 예측하는 것 |
| 손실 함수 | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ — 단순 MSE |
| 시간 임베딩 | 사인/코사인 인코딩으로 시점 $t$를 벡터화 |
| 생성 과정 | $x_T \to x_{T-1} \to ... \to x_0$ — T 단계 디노이징 |
| 예측 대상 | 노이즈 $\epsilon$, 원본 $x_0$, 속도 $v$ 중 선택 가능 |

## 다음 섹션 미리보기

DDPM은 강력하지만 생성에 1000 스텝이 필요하다는 단점이 있습니다. 다음 섹션 [DDIM과 샘플링 가속](./03-ddim.md)에서는 이 속도 문제를 해결하는 혁신적인 방법들을 알아봅니다. DDPM 모델을 **재학습 없이** 20~50 스텝으로 줄이는 마법 같은 기법이 기다리고 있어요.

## 참고 자료

- [Ho et al., "Denoising Diffusion Probabilistic Models" (2020)](https://arxiv.org/abs/2006.11239) - DDPM 원논문
- [LearnOpenCV - DDPM Theory to Implementation](https://learnopencv.com/denoising-diffusion-probabilistic-models/) - 이론부터 구현까지 상세 가이드
- [Keras DDPM Tutorial](https://keras.io/examples/generative/ddpm/) - Keras 기반 DDPM 구현
- [Gabriel Mongaras - Diffusion Models Guide](https://betterprogramming.pub/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869) - DDPM/DDIM/CFG 종합 해설
