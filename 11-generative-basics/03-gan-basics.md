# GAN 기초

> 생성자와 판별자의 적대적 학습

## 개요

GAN(Generative Adversarial Network)은 **두 신경망이 서로 경쟁**하며 실력을 키우는, 생성 모델 역사에서 가장 혁신적인 아이디어 중 하나입니다. 위조지폐범과 경찰의 관계처럼, 생성자(Generator)와 판별자(Discriminator)의 끊임없는 대결이 놀라운 결과를 만들어내죠.

**선수 지식**: [생성 모델 개론](./01-generative-intro.md), [VAE](./02-vae.md), [역전파 알고리즘](../03-deep-learning-basics/03-backpropagation.md)
**학습 목표**:
- GAN의 핵심 아이디어인 적대적 학습(Adversarial Training)을 이해한다
- 생성자와 판별자의 역할과 학습 과정을 파악한다
- GAN의 손실 함수와 학습 동역학을 이해한다
- PyTorch로 간단한 GAN을 직접 구현할 수 있다

## 왜 알아야 할까?

앞서 [VAE](./02-vae.md)에서 본 것처럼, VAE는 안정적이지만 생성된 이미지가 흐릿한 한계가 있었죠. GAN은 이 문제를 완전히 다른 각도에서 해결합니다. VAE가 "원본과 비슷하게 복원해"라는 직접적인 감독을 사용한다면, GAN은 "진짜와 구별할 수 없을 만큼 만들어"라는 **간접적인 피드백**으로 학습합니다. 그 결과, GAN은 놀랍도록 선명하고 사실적인 이미지를 만들어낼 수 있습니다.

## 핵심 개념

### 개념 1: 위조범과 감정사 — GAN의 직관

> 💡 **비유**: 화폐 위조범(생성자)과 은행 감정사(판별자)가 있다고 상상해보세요. 처음에 위조범의 실력은 형편없어서, 감정사는 가짜를 단번에 알아봅니다. 하지만 위조범은 매번 "왜 들켰는지" 피드백을 받아 점점 실력을 키우고, 감정사도 점점 더 미세한 차이를 잡아내려 노력합니다. 이 경쟁이 오래 계속되면? 감정사조차 진짜와 가짜를 구별할 수 없는 수준의 위조 화폐가 탄생하죠.

GAN은 정확히 이 원리로 작동합니다:

- **생성자(Generator, G)**: 랜덤 노이즈 $z$를 입력받아 가짜 이미지를 생성
- **판별자(Discriminator, D)**: 이미지를 입력받아 진짜인지 가짜인지 판별

이 두 네트워크는 **적대적(Adversarial)**으로 학습합니다. 생성자는 판별자를 속이려 하고, 판별자는 속지 않으려 합니다.

> 랜덤 노이즈 $z$ → **생성자 G** → 가짜 이미지 → **판별자 D** → 진짜? 가짜?

### 개념 2: GAN의 손실 함수 — 미니맥스 게임

GAN의 학습은 게임 이론의 **미니맥스 게임(Minimax Game)**으로 표현됩니다:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

복잡해 보이지만, 하나씩 뜯어보면 간단합니다:

- $D(x)$: 판별자가 진짜 이미지 $x$를 진짜라고 판별할 확률
- $D(G(z))$: 판별자가 가짜 이미지 $G(z)$를 진짜라고 판별할 확률
- **판별자 D의 목표** ($\max_D$): $D(x)$는 높이고, $D(G(z))$는 낮추고 → "진짜는 진짜로, 가짜는 가짜로"
- **생성자 G의 목표** ($\min_G$): $D(G(z))$를 높이고 → "가짜를 진짜로 착각하게"

> ⚠️ **흔한 오해**: "GAN의 목표는 판별자를 완전히 속이는 것이다" — 사실 GAN의 이론적 최적해에서 판별자는 모든 입력에 대해 $D(x) = 0.5$를 출력합니다. 즉, 진짜와 가짜를 구별할 수 없는 **균형 상태(Nash Equilibrium)**가 목표입니다.

### 개념 3: GAN의 학습 과정 — 번갈아 학습

GAN의 학습은 판별자와 생성자를 **번갈아가며** 업데이트합니다:

**Step 1: 판별자 업데이트**
1. 실제 데이터에서 진짜 이미지를 가져옴 → 레이블: 1 (진짜)
2. 생성자로 가짜 이미지를 만듦 → 레이블: 0 (가짜)
3. 판별자가 진짜/가짜를 올바르게 분류하도록 학습

**Step 2: 생성자 업데이트**
1. 랜덤 노이즈로 가짜 이미지를 생성
2. 판별자에 넣어서 "진짜라고 판단하게" 만드는 방향으로 학습
3. 이때 판별자의 가중치는 **고정** (생성자만 업데이트)

이 과정을 수만~수십만 번 반복하면, 생성자는 점점 더 사실적인 이미지를 만들게 됩니다.

### 개념 4: GAN의 도전 과제

GAN이 강력한 만큼, 학습이 까다롭기로도 유명합니다:

**모드 붕괴(Mode Collapse)**

> 💡 **비유**: 만약 위조범이 "100달러짜리만 완벽하게 만들자"라고 결정하고 다른 지폐는 포기한다면? 생성자가 데이터 분포의 일부만 학습하고 다양성을 잃는 현상입니다. MNIST에서 "3"만 잘 만들고 나머지 숫자는 못 만드는 것처럼요.

**학습 불안정**

판별자가 너무 강해지면 생성자에게 유의미한 그래디언트가 전달되지 않고, 반대로 생성자가 너무 강해지면 판별자가 학습할 동기를 잃습니다. 이 균형을 맞추는 것이 GAN 학습의 핵심 도전이죠.

**평가의 어려움**

[분류 모델](../06-image-classification/01-mnist.md)은 정확도로 쉽게 평가할 수 있지만, "생성된 이미지가 얼마나 좋은가?"를 수치화하기는 매우 어렵습니다. FID(Fréchet Inception Distance)와 IS(Inception Score) 같은 지표가 이를 위해 개발되었죠.

## 실습: PyTorch로 GAN 구현하기

```python
import torch
import torch.nn as nn

# ========================================
# 생성자: 랜덤 노이즈 → 이미지
# ========================================
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 784),        # 28x28 = 784
            nn.Tanh()                     # 출력 범위: -1 ~ 1
        )

    def forward(self, z):
        img = self.net(z)
        return img.view(-1, 1, 28, 28)   # 이미지 형태로 변환


# ========================================
# 판별자: 이미지 → 진짜/가짜 확률
# ========================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),             # 판별자 과적합 방지

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()                  # 0~1 확률 출력
        )

    def forward(self, img):
        return self.net(img)


# ========================================
# GAN 학습 루프 (한 에폭 기준)
# ========================================
def train_gan_step(G, D, real_images, opt_G, opt_D, criterion, latent_dim=100):
    batch_size = real_images.size(0)
    device = real_images.device

    # 레이블 생성
    real_labels = torch.ones(batch_size, 1, device=device)    # 진짜 = 1
    fake_labels = torch.zeros(batch_size, 1, device=device)   # 가짜 = 0

    # ---- 1단계: 판별자 학습 ----
    # 진짜 이미지로 학습
    real_pred = D(real_images)
    d_loss_real = criterion(real_pred, real_labels)

    # 가짜 이미지로 학습
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = G(z).detach()              # 생성자 그래프 끊기
    fake_pred = D(fake_images)
    d_loss_fake = criterion(fake_pred, fake_labels)

    d_loss = d_loss_real + d_loss_fake
    opt_D.zero_grad()
    d_loss.backward()
    opt_D.step()

    # ---- 2단계: 생성자 학습 ----
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = G(z)
    fake_pred = D(fake_images)
    # 핵심: 가짜를 "진짜"로 판별하게 만드는 방향으로 학습
    g_loss = criterion(fake_pred, real_labels)

    opt_G.zero_grad()
    g_loss.backward()
    opt_G.step()

    return d_loss.item(), g_loss.item()


# 모델 초기화 및 테스트
latent_dim = 100
G = Generator(latent_dim)
D = Discriminator()
criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 더미 데이터로 한 스텝 테스트
real = torch.randn(32, 1, 28, 28)    # 가상의 실제 이미지 배치
d_loss, g_loss = train_gan_step(G, D, real, opt_G, opt_D, criterion, latent_dim)
print(f"판별자 손실: {d_loss:.4f}")
print(f"생성자 손실: {g_loss:.4f}")

# 이미지 생성 테스트
with torch.no_grad():
    z = torch.randn(16, latent_dim)
    generated = G(z)
    print(f"생성된 이미지 크기: {generated.shape}")  # [16, 1, 28, 28]
```

> 🔥 **실무 팁**: GAN 학습에서 가장 중요한 하이퍼파라미터는 **학습률**입니다. Adam 옵티마이저의 `lr=0.0002`, `betas=(0.5, 0.999)`는 DCGAN 논문에서 제안된 값으로, GAN 학습의 사실상 표준입니다. 이 값에서 크게 벗어나면 학습이 불안정해질 수 있어요.

## 더 깊이 알아보기

### Ian Goodfellow와 술집의 밤

[생성 모델 개론](./01-generative-intro.md)에서 잠깐 언급했지만, GAN의 탄생 이야기를 더 자세히 해볼까요. 2014년 당시 몬트리올 대학 박사과정이던 Ian Goodfellow는 Yoshua Bengio 교수의 연구실 소속이었습니다.

당시 생성 모델 연구자들은 볼츠만 머신의 복잡한 근사 추론이나, VAE의 흐릿한 출력에 불만을 품고 있었죠. Goodfellow의 아이디어는 "생성 과정을 직접 모델링하지 말고, **판별자라는 비평가**를 세우자"는 것이었습니다.

놀라운 점은 이 아이디어가 **첫 구현에서 바로 작동했다**는 것입니다. 보통 새로운 알고리즘은 수없이 많은 디버깅과 수정을 거치는데, GAN은 예외였죠. Goodfellow는 NIPS 2014에 이 논문을 발표했고, 이는 딥러닝 역사에서 가장 영향력 있는 논문 중 하나가 되었습니다.

### GAN과 게임 이론

GAN의 수학적 기초는 게임 이론의 **내시 균형(Nash Equilibrium)**에 있습니다. 두 플레이어(생성자와 판별자)가 각각 자신의 전략을 최적화할 때, 어느 쪽도 일방적으로 전략을 바꿀 유인이 없는 상태가 내시 균형이죠.

GAN의 이론적 최적해에서:
- 생성자의 출력 분포 = 실제 데이터 분포: $p_G(x) = p_{data}(x)$
- 판별자의 출력 = 모든 입력에 대해 0.5: $D(x) = 0.5$

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "판별자 손실과 생성자 손실이 둘 다 내려가면 잘 학습되는 것이다" — 아닙니다! GAN에서는 두 손실이 **서로 경쟁**하기 때문에, 한쪽이 내려가면 다른 쪽이 올라가는 게 정상입니다. 두 손실이 어느 정도 **균형**을 이루면서 진동하는 것이 건강한 학습의 신호예요.

> 💡 **알고 계셨나요?**: GAN 논문의 제목에 있는 "Adversarial"은 "적대적"이라는 뜻이지만, 사실 두 네트워크는 **협력적 경쟁** 관계에 가깝습니다. 판별자가 좋은 피드백을 줘야 생성자가 잘 배울 수 있으니까요. 스포츠에서 좋은 라이벌이 서로를 성장시키는 것과 비슷하죠.

> 🔥 **실무 팁**: GAN 학습이 불안정할 때 시도해볼 것들 — (1) LeakyReLU 대신 ReLU를 쓰지 말 것, (2) BatchNorm은 생성자에만 적용, (3) 판별자에 Dropout 추가, (4) 학습률을 낮추기, (5) 판별자를 여러 번 업데이트한 후 생성자를 1번 업데이트하는 비율 조절.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 생성자 (G) | 랜덤 노이즈에서 가짜 이미지를 생성하는 네트워크 |
| 판별자 (D) | 이미지가 진짜인지 가짜인지 판별하는 네트워크 |
| 적대적 학습 | G와 D가 서로 경쟁하며 동시에 학습하는 방식 |
| 미니맥스 게임 | G는 최소화, D는 최대화하는 게임 이론적 프레임워크 |
| 모드 붕괴 | 생성자가 데이터의 일부 모드만 학습하는 실패 모드 |
| 내시 균형 | GAN의 이론적 최적해, $D(x) = 0.5$ |

## 다음 섹션 미리보기

기본 GAN의 원리를 이해했으니, 다음 섹션 [GAN 변형들](./04-gan-variants.md)에서는 이 아이디어를 발전시킨 혁신적인 모델들을 만나봅니다. DCGAN의 안정적 학습 비법, StyleGAN의 놀라운 얼굴 생성, CycleGAN의 마법 같은 스타일 변환까지 — GAN의 진화 과정을 따라가 봅시다.

## 참고 자료

- [Goodfellow et al., "Generative Adversarial Nets" (2014)](https://arxiv.org/abs/1406.2661) - GAN 원논문
- [Google ML - GAN Introduction](https://developers.google.com/machine-learning/gan) - 구글의 GAN 튜토리얼
- [DCGAN Tutorial (PyTorch)](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) - PyTorch 공식 DCGAN 튜토리얼
- [GAN Full Guide 2025](https://www.tutorialsfreak.com/ai-tutorial/generative-adversarial-networks) - 최신 GAN 종합 가이드
