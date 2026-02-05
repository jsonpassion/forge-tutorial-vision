# Variational Autoencoder

> 잠재 공간과 재구성

## 개요

VAE(Variational Autoencoder)는 이미지를 **압축**했다가 다시 **복원**하는 과정에서, 그 "압축된 코드"의 세계를 탐험하여 새로운 이미지를 만들어내는 생성 모델입니다. 이번 섹션에서는 오토인코더에서 VAE로의 진화 과정과, 잠재 공간이라는 매력적인 개념을 깊이 있게 다룹니다.

**선수 지식**: [생성 모델 개론](./01-generative-intro.md), [신경망의 구조](../03-deep-learning-basics/01-neural-network.md), [손실 함수와 옵티마이저](../03-deep-learning-basics/04-loss-optimizer.md)
**학습 목표**:
- 오토인코더(AE)와 VAE의 핵심 차이를 이해한다
- 잠재 공간(Latent Space)의 의미와 활용법을 파악한다
- Reparameterization Trick의 원리를 이해한다
- PyTorch로 VAE를 직접 구현할 수 있다

## 왜 알아야 할까?

VAE는 단순히 "오래된 생성 모델"이 아닙니다. Stable Diffusion의 핵심 구성요소인 **VAE 인코더/디코더**가 바로 이 기술을 기반으로 합니다. 이미지를 잠재 공간(Latent Space)으로 압축하는 개념은 [CLIP](../10-vision-language/02-clip.md)의 임베딩 공간, [BLIP-2](../10-vision-language/03-blip.md)의 Q-Former 등 현대 멀티모달 모델에서도 핵심적으로 사용되죠. VAE를 이해하면 이후의 모든 생성 모델이 훨씬 쉽게 다가옵니다.

## 핵심 개념

### 개념 1: 오토인코더 — 압축의 달인

> 💡 **비유**: 오토인코더는 **메모 요약의 달인**입니다. 10페이지짜리 보고서를 핵심 키워드 3개로 요약한 뒤, 그 키워드만으로 보고서를 다시 복원하는 거죠. 요약이 잘 되었다면 복원도 잘 될 테고, 중요한 정보를 빠뜨렸다면 복원이 엉성해질 겁니다.

오토인코더(Autoencoder)는 두 부분으로 구성됩니다:

- **인코더(Encoder)**: 입력 이미지를 저차원 벡터(잠재 벡터)로 압축
- **디코더(Decoder)**: 잠재 벡터를 다시 원본 크기의 이미지로 복원

> 입력 이미지 → **인코더** → 잠재 벡터 $z$ → **디코더** → 복원된 이미지

학습 목표는 간단합니다 — 입력과 출력이 최대한 같도록, 즉 **재구성 오차(Reconstruction Loss)**를 최소화하는 것이죠.

하지만 일반 오토인코더에는 치명적인 문제가 있습니다. 잠재 공간이 **울퉁불퉁**하다는 겁니다. 학습 데이터에 대응하는 점들만 의미가 있고, 그 사이의 빈 공간에서 샘플링하면 이상한 이미지가 나올 수 있어요. 이것은 생성 모델로 쓰기에 부적합하죠.

### 개념 2: VAE의 핵심 아이디어 — 확률적 인코딩

> 💡 **비유**: 일반 오토인코더가 "이 사진의 요약은 정확히 이 좌표(점)입니다"라고 말한다면, VAE는 "이 사진의 요약은 대략 이 **영역(구름)** 어딘가입니다"라고 말합니다. 점 대신 구름을 사용하면 구름들이 서로 겹치면서 빈 공간이 사라지고, 잠재 공간의 어디를 찍어도 의미 있는 이미지를 만들 수 있게 되죠.

VAE의 혁신은 인코더의 출력을 **하나의 점**이 아니라 **확률 분포**로 만든 것입니다:

- 일반 AE 인코더: 이미지 → 잠재 벡터 $z$ (고정된 점)
- VAE 인코더: 이미지 → 평균 $\mu$와 분산 $\sigma^2$ (가우시안 분포의 파라미터)

그런 다음 이 분포에서 **샘플링**하여 잠재 벡터 $z$를 뽑습니다:

$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

> 여기서 $\mu$는 인코더가 예측한 평균, $\sigma$는 표준편차, $\epsilon$은 표준 정규분포에서 뽑은 랜덤 노이즈입니다. 이 수식이 바로 유명한 **Reparameterization Trick**이죠.

왜 이렇게 하는 걸까요? 두 가지 이유가 있습니다:

1. **잠재 공간이 매끄러워집니다**: 점 대신 "구름"을 사용하면 빈 공간이 채워져요
2. **역전파가 가능해집니다**: 샘플링은 미분이 안 되지만, $\mu + \sigma \cdot \epsilon$ 형태로 바꾸면 $\mu$와 $\sigma$에 대해 미분할 수 있죠

### 개념 3: VAE의 손실 함수 — 두 마리 토끼 잡기

VAE의 손실 함수는 두 부분으로 구성됩니다:

$$\mathcal{L}_{VAE} = \underbrace{\mathcal{L}_{recon}}_{\text{재구성 손실}} + \beta \cdot \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{KL 다이버전스}}$$

**1. 재구성 손실 (Reconstruction Loss)**
- 입력 이미지와 복원된 이미지가 얼마나 비슷한지 측정
- 보통 MSE 또는 Binary Cross-Entropy 사용
- "요약한 것에서 원본을 잘 복원해야 한다"

**2. KL 다이버전스 (KL Divergence)**
- 인코더가 만든 분포 $q(z|x)$가 표준 정규분포 $\mathcal{N}(0, I)$에 얼마나 가까운지 측정
- 잠재 공간을 **정규화**하여 매끄럽고 연속적으로 만드는 역할
- "요약이 너무 특이하지 않고 일반적이어야 한다"

> ⚠️ **흔한 오해**: "KL 다이버전스는 그냥 정규화 기법이다" — 아닙니다! KL 다이버전스는 잠재 공간의 **구조를 결정**하는 핵심 요소입니다. 이것 없이는 잠재 공간이 불연속적이 되어 새로운 이미지를 생성할 수 없게 됩니다.

### 개념 4: 잠재 공간의 마법

VAE가 학습한 잠재 공간에서는 재미있는 일이 벌어집니다:

**보간(Interpolation)**: 두 이미지의 잠재 벡터를 연결하면, 자연스러운 중간 이미지들이 생깁니다. 웃는 얼굴의 잠재 벡터와 무표정 얼굴의 잠재 벡터를 반반 섞으면 살짝 미소 짓는 얼굴이 나오죠.

**산술 연산**: 잠재 공간에서 벡터 연산이 가능합니다.

> **안경 쓴 남자** $-$ **남자** $+$ **여자** $=$ **안경 쓴 여자**

이런 연산이 가능한 이유는 VAE의 잠재 공간이 **의미적으로 구조화**되어 있기 때문입니다.

## 실습: PyTorch로 VAE 구현하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """MNIST용 Variational Autoencoder"""
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim

        # 인코더: 이미지 → 평균(μ)과 로그분산(log σ²)
        self.encoder = nn.Sequential(
            nn.Flatten(),                        # 28x28 → 784
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)      # 평균 μ
        self.fc_logvar = nn.Linear(256, latent_dim)   # 로그 분산 log(σ²)

        # 디코더: 잠재 벡터 → 이미지
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),                        # 픽셀값 0~1로 제한
        )

    def encode(self, x):
        """이미지를 μ와 log(σ²)로 인코딩"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization Trick: z = μ + σ * ε"""
        std = torch.exp(0.5 * logvar)           # log(σ²) → σ
        eps = torch.randn_like(std)             # ε ~ N(0, I)
        return mu + std * eps

    def decode(self, z):
        """잠재 벡터를 이미지로 디코딩"""
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """VAE 손실 = 재구성 손실 + KL 다이버전스"""
    # 재구성 손실: 원본과 복원의 차이
    recon_loss = F.binary_cross_entropy(
        recon_x.view(-1, 784), x.view(-1, 784), reduction='sum'
    )
    # KL 다이버전스: 잠재 분포를 정규분포로 유도
    # KL(q(z|x) || N(0,I)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


# 모델 생성 및 테스트
vae = VAE(latent_dim=20)
dummy_input = torch.randn(4, 1, 28, 28)       # 배치 4장, MNIST 크기

recon, mu, logvar = vae(dummy_input)
loss = vae_loss(recon, dummy_input, mu, logvar)

print(f"입력 크기: {dummy_input.shape}")        # [4, 1, 28, 28]
print(f"재구성 크기: {recon.shape}")             # [4, 1, 28, 28]
print(f"잠재 벡터 평균(μ) 크기: {mu.shape}")     # [4, 20]
print(f"잠재 벡터 로그분산 크기: {logvar.shape}") # [4, 20]
print(f"VAE 손실: {loss.item():.1f}")
```

### 새로운 이미지 생성하기

```python
# 학습이 완료된 후, 잠재 공간에서 샘플링하여 새 이미지 생성
with torch.no_grad():
    # 표준 정규분포에서 잠재 벡터 샘플링
    z = torch.randn(16, 20)           # 16개의 새로운 잠재 벡터
    generated = vae.decode(z)          # 디코더로 이미지 생성
    print(f"생성된 이미지: {generated.shape}")  # [16, 1, 28, 28]

    # 보간(Interpolation): 두 이미지 사이의 부드러운 전환
    z1 = torch.randn(1, 20)           # 시작점
    z2 = torch.randn(1, 20)           # 끝점
    alphas = torch.linspace(0, 1, 8)  # 8단계 보간
    interpolated = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        img = vae.decode(z_interp)
        interpolated.append(img)
    interpolated = torch.cat(interpolated, dim=0)
    print(f"보간 이미지: {interpolated.shape}")  # [8, 1, 28, 28]
```

## 더 깊이 알아보기

### VAE의 후예들

VAE의 기본 아이디어는 수많은 변형을 낳았습니다:

- **β-VAE (2017)**: KL 다이버전스에 가중치 $\beta$를 조절하여 잠재 공간의 분리(Disentanglement)를 강화
- **VQ-VAE (2017)**: 연속적인 잠재 공간 대신 **이산적인 코드북**을 사용 — 이미지 품질 대폭 향상
- **VQ-VAE-2 (2019)**: 다중 해상도 코드북으로 고해상도 이미지 생성 가능

특히 VQ-VAE는 이후 Stable Diffusion에서 이미지를 잠재 공간으로 압축하는 데 핵심적으로 활용됩니다. [Diffusion 모델](../12-diffusion-models/01-diffusion-theory.md)에서 더 자세히 다룰 예정이에요.

### Diederik Kingma의 박사 논문

VAE를 제안한 Diederik Kingma는 당시 박사 과정 학생이었습니다. 2013년에 발표된 "Auto-Encoding Variational Bayes" 논문은 현재 인용 수 3만 회를 넘겼죠. 재미있는 사실은, Kingma가 같은 시기에 **Adam 옵티마이저**도 발표했다는 겁니다! [옵티마이저 섹션](../03-deep-learning-basics/04-loss-optimizer.md)에서 배운 바로 그 Adam이요. 한 명의 박사생이 딥러닝의 두 가지 핵심 도구를 동시에 만든 셈입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "VAE의 이미지가 흐릿한 건 모델이 나빠서다" — VAE가 흐릿한 이미지를 생성하는 건 MSE 기반 재구성 손실의 본질적 한계입니다. MSE는 평균적으로 가장 안전한 답(흐릿한 중간값)을 선택하기 때문이죠. 이를 개선하기 위해 VQ-VAE나 Perceptual Loss 같은 기법이 등장했습니다.

> 💡 **알고 계셨나요?**: VAE의 "변분(Variational)"이라는 이름은 변분 추론(Variational Inference)에서 온 것입니다. 복잡한 사후 확률을 직접 계산하기 어려울 때, 간단한 분포로 근사하는 통계적 기법이죠. VAE는 이것을 신경망으로 구현한 겁니다.

> 🔥 **실무 팁**: `latent_dim`은 너무 크면 과적합, 너무 작으면 정보 손실이 발생합니다. MNIST는 20~50, 복잡한 이미지는 128~512가 일반적입니다. β-VAE처럼 KL 가중치를 조절하면 재구성 품질과 잠재 공간 구조 사이의 균형을 맞출 수 있어요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 오토인코더 | 입력을 압축(인코딩)하고 복원(디코딩)하는 신경망 |
| VAE | 잠재 벡터를 확률 분포로 모델링한 오토인코더 |
| 잠재 공간 | 데이터의 핵심 특성이 압축된 저차원 표현 공간 |
| Reparameterization Trick | $z = \mu + \sigma \cdot \epsilon$으로 샘플링을 미분 가능하게 만드는 기법 |
| KL 다이버전스 | 잠재 분포를 정규분포에 가깝게 유도하는 정규화 항 |
| VQ-VAE | 이산 코드북을 사용한 VAE 변형, Stable Diffusion의 기반 |

## 다음 섹션 미리보기

VAE가 "압축과 복원"이라는 우아한 전략을 쓴다면, 다음 섹션 [GAN 기초](./03-gan-basics.md)는 완전히 다른 접근 — **"위조범과 감정사의 대결"**이라는 게임 이론적 방법을 사용합니다. GAN이 어떻게 VAE보다 훨씬 선명한 이미지를 만들어내는지 알아볼까요?

## 참고 자료

- [Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)](https://arxiv.org/abs/1312.6114) - VAE 원논문
- [Variational Autoencoders (DataCamp Tutorial)](https://www.datacamp.com/tutorial/variational-autoencoders) - 실습 중심 VAE 가이드
- [van den Oord et al., "Neural Discrete Representation Learning" (2017)](https://arxiv.org/abs/1711.00937) - VQ-VAE 논문
- [Keras VAE Tutorial](https://keras.io/examples/generative/vae/) - Keras 기반 VAE 구현 예제
