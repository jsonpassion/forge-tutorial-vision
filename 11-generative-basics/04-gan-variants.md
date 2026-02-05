# GAN 변형들

> DCGAN, StyleGAN, CycleGAN

## 개요

기본 GAN이 "가능성"을 보여줬다면, 이후 등장한 변형들은 GAN의 **실용성**을 증명했습니다. CNN을 도입한 DCGAN, 포토리얼리스틱 얼굴을 만든 StyleGAN, 쌍이 없는 데이터로 스타일 변환을 해낸 CycleGAN까지 — 각 변형이 어떤 문제를 풀었고, 어떤 아이디어로 돌파했는지 알아봅니다.

**선수 지식**: [GAN 기초](./03-gan-basics.md), [합성곱 연산](../04-cnn-fundamentals/01-convolution.md), [배치 정규화](../04-cnn-fundamentals/03-batch-normalization.md)
**학습 목표**:
- DCGAN이 GAN 학습을 안정화시킨 핵심 기법들을 이해한다
- Conditional GAN으로 원하는 종류의 이미지를 생성하는 방법을 배운다
- StyleGAN의 스타일 기반 생성 구조를 이해한다
- CycleGAN의 비지도 이미지 변환 원리를 파악한다

## 왜 알아야 할까?

[기본 GAN](./03-gan-basics.md)은 MNIST 같은 작은 이미지에서는 잘 작동했지만, 고해상도 이미지에서는 학습이 불안정했습니다. GAN 변형들은 이 한계를 하나씩 극복해나갔고, 결국 진짜 사진과 구별할 수 없는 수준까지 도달했죠. 이 진화 과정을 이해하면, 왜 특정 아키텍처 설계가 중요한지 깊이 있게 알 수 있습니다.

## 핵심 개념

### 개념 1: DCGAN — CNN을 만난 GAN

> 💡 **비유**: 기본 GAN이 **스케치만으로 그림 그리기**였다면, DCGAN은 **전문 화구 세트**를 갖춘 것과 같습니다. [합성곱(Convolution)](../04-cnn-fundamentals/01-convolution.md)이라는 강력한 도구를 생성자와 판별자 모두에 도입한 거죠.

DCGAN(Deep Convolutional GAN, 2015)은 Alec Radford가 제안한 모델로, 안정적인 GAN 학습을 위한 **아키텍처 가이드라인**을 확립했습니다:

**DCGAN의 핵심 규칙들**:
1. **풀링 대신 스트라이드 합성곱** — [풀링](../04-cnn-fundamentals/02-pooling.md)을 쓰지 않고, 스트라이드 2인 합성곱으로 다운/업 샘플링
2. **배치 정규화(BatchNorm) 적용** — 생성자와 판별자 모두에 [BatchNorm](../04-cnn-fundamentals/03-batch-normalization.md) 사용 (생성자의 출력층과 판별자의 입력층 제외)
3. **완전 연결층 제거** — 글로벌 평균 풀링이나 합성곱으로 대체
4. **활성화 함수 선택** — 생성자는 ReLU(마지막은 Tanh), 판별자는 LeakyReLU

이 규칙들은 지금까지도 GAN 설계의 기본 원칙으로 사용됩니다.

```python
import torch
import torch.nn as nn

class DCGenerator(nn.Module):
    """DCGAN 생성자 — 전치 합성곱으로 이미지를 키워나감"""
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            # 입력: (latent_dim, 1, 1) → (256, 7, 7)
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # (256, 7, 7) → (128, 14, 14)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # (128, 14, 14) → (channels, 28, 28)
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()                         # 출력 범위: -1 ~ 1
        )

    def forward(self, z):
        z = z.view(-1, z.size(1), 1, 1)      # (B, latent_dim) → (B, latent_dim, 1, 1)
        return self.net(z)


class DCDiscriminator(nn.Module):
    """DCGAN 판별자 — 스트라이드 합성곱으로 이미지를 축소"""
    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            # (channels, 28, 28) → (64, 14, 14)
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 14, 14) → (128, 7, 7)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 7, 7) → (1, 1, 1)
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)


# 테스트
G = DCGenerator(latent_dim=100)
D = DCDiscriminator()
z = torch.randn(4, 100)
fake = G(z)
pred = D(fake)
print(f"잠재 벡터: {z.shape}")        # [4, 100]
print(f"생성 이미지: {fake.shape}")    # [4, 1, 28, 28]
print(f"판별 결과: {pred.shape}")      # [4, 1]
```

### 개념 2: Conditional GAN — 조건부 생성

> 💡 **비유**: 기본 GAN이 "아무 그림이나 그려"라는 주문이었다면, Conditional GAN(cGAN)은 "**고양이** 그림을 그려"처럼 **원하는 종류**를 지정할 수 있는 주문입니다.

Conditional GAN(2014)은 생성자와 판별자 모두에게 **조건 정보(condition)**를 추가합니다:

- 생성자: 노이즈 $z$ + 조건 $c$ → 이미지 생성
- 판별자: 이미지 + 조건 $c$ → 진짜/가짜 판별

조건 $c$는 클래스 레이블, 텍스트, 다른 이미지 등 다양한 형태가 될 수 있어요. MNIST에서 "숫자 7을 생성해줘"라고 할 수 있는 거죠.

이 아이디어는 이후 Pix2Pix, SPADE 등 다양한 응용으로 발전합니다. 특히 [CLIP](../10-vision-language/02-clip.md)의 텍스트 조건부 이미지 생성의 초기 형태라고 볼 수 있죠.

### 개념 3: StyleGAN — 포토리얼리스틱의 정점

> 💡 **비유**: 일반 화가가 빈 캔버스에서 한 붓으로 그림을 완성한다면, StyleGAN은 마치 **포토샵 레이어**처럼 작업합니다. 전체적인 포즈(거친 스타일) → 얼굴 형태(중간 스타일) → 피부 질감(미세 스타일)을 **각각 독립적으로** 조절할 수 있죠.

NVIDIA의 Tero Karras 팀이 개발한 StyleGAN(2018)은 GAN 이미지 생성의 판도를 바꿨습니다.

**StyleGAN의 핵심 혁신 — 스타일 기반 생성**:

기존 GAN은 잠재 벡터 $z$를 생성자에 직접 입력했지만, StyleGAN은 다른 구조를 사용합니다:

1. **매핑 네트워크**: 잠재 벡터 $z$를 스타일 벡터 $w$로 변환 (8층 MLP)
2. **AdaIN (Adaptive Instance Normalization)**: $w$를 각 합성곱 레이어에 스타일로 주입
3. **노이즈 주입**: 각 레이어에 랜덤 노이즈를 추가하여 세밀한 변화(주근깨, 머리카락 등) 생성

> 잠재 벡터 $z$ → **매핑 네트워크** → 스타일 $w$ → 각 레이어에 **AdaIN**으로 주입

**StyleGAN의 진화**:

| 버전 | 연도 | 핵심 개선 |
|------|------|----------|
| StyleGAN | 2018 | 스타일 기반 생성 구조 제안 |
| StyleGAN2 | 2020 | 물방울 아티팩트 제거, 경로 길이 정규화 |
| StyleGAN3 | 2021 | Alias-Free 생성, 텍스처 고정 문제 해결 |
| StyleGAN-XL | 2022 | ImageNet 규모 학습 성공, 1024² 해상도 |

> 💡 **알고 계셨나요?**: StyleGAN이 생성한 "존재하지 않는 사람의 얼굴"은 너무 사실적이어서, 가짜 SNS 프로필 생성에 악용되기도 했습니다. 이 때문에 "This Person Does Not Exist" 웹사이트가 AI 생성 얼굴의 위험성을 알리기 위해 만들어졌죠. AI 윤리와 딥페이크 탐지 기술의 중요성을 일깨워준 사례입니다.

### 개념 4: CycleGAN — 짝이 없어도 변환 가능

> 💡 **비유**: 한국어-영어 번역을 배울 때, 보통은 **쌍을 이룬 예문**(한국어 문장 ↔ 영어 번역)이 필요하죠. 그런데 CycleGAN은 한국어 책과 영어 책을 **따로** 주고 "이 두 언어의 관계를 스스로 파악해봐"라고 하는 셈입니다.

CycleGAN(2017)의 핵심 아이디어는 **순환 일관성 손실(Cycle Consistency Loss)**입니다:

1. 말(Horse)의 이미지를 얼룩말(Zebra)로 변환: $G_{H→Z}(horse)$ = zebra'
2. 변환된 얼룩말을 다시 말로 복원: $G_{Z→H}(zebra')$ = horse'
3. 원래 말과 복원된 말이 같아야 함: $horse \approx horse'$

> 말 → **G: 말→얼룩말** → 얼룩말' → **F: 얼룩말→말** → 말' ≈ 원래 말

이렇게 **왕복 변환이 원본을 보존**해야 한다는 제약 덕분에, 쌍을 이룬 데이터 없이도 의미 있는 변환을 학습할 수 있습니다.

**CycleGAN의 대표적 응용**:
- 말 ↔ 얼룩말 변환
- 여름 풍경 ↔ 겨울 풍경
- 사진 ↔ 모네/고흐 스타일 그림
- 낮 사진 ↔ 밤 사진

### 개념 5: Pix2Pix — 쌍이 있는 이미지 변환

CycleGAN이 비지도 변환이라면, **Pix2Pix(2016)**는 **쌍을 이룬 데이터**가 있을 때 사용하는 조건부 이미지 변환 모델입니다:

- 스케치 → 실사 사진
- 위성 사진 → 지도
- 흑백 사진 → 컬러 사진
- 에지 맵 → 실물 이미지

Pix2Pix는 [U-Net](../08-segmentation/01-semantic-segmentation.md) 구조를 생성자로, PatchGAN을 판별자로 사용합니다. 특히 판별자가 이미지 전체가 아닌 **패치 단위**로 진짜/가짜를 판별하는 것이 핵심이죠.

## 더 깊이 알아보기

### WGAN — 손실 함수의 혁명

Wasserstein GAN(WGAN, 2017)은 기본 GAN의 학습 불안정 문제를 **수학적으로 분석**하고 해결한 모델입니다. 핵심은 BCE 손실 대신 **Wasserstein 거리(Earth Mover's Distance)**를 사용하는 것:

- 기존 GAN: KL/JS 다이버전스 → 두 분포가 겹치지 않으면 그래디언트가 0
- WGAN: Wasserstein 거리 → 항상 의미 있는 그래디언트 제공

이후 WGAN-GP(Gradient Penalty)가 더 안정적인 학습을 가능하게 했고, 이 손실 함수 아이디어는 많은 후속 GAN에 영향을 미쳤습니다.

### GAN의 황금기: 2017~2020

2017년부터 2020년까지는 GAN의 황금기였습니다. 매년 새로운 GAN 변형이 쏟아져 나왔고, 그 목록만으로 유명한 "GAN Zoo"라는 리포지토리가 만들어질 정도였죠. 한때 GAN 변형이 **500개**를 넘었다고 합니다! 이 중 살아남은 것은 소수지만, 각각이 해결한 문제는 이후 Diffusion 모델 등에서도 재활용되고 있습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "StyleGAN은 얼굴만 만들 수 있다" — StyleGAN의 구조는 범용적입니다. 얼굴 외에도 자동차, 교회, 고양이, 예술 작품 등 다양한 도메인에서 훈련할 수 있어요. StyleGAN-XL은 ImageNet의 1000가지 클래스에 대해 성공적으로 학습했습니다.

> 🔥 **실무 팁**: CycleGAN을 사용할 때, 두 도메인의 **구조적 유사성**이 중요합니다. 말↔얼룩말은 체형이 비슷해서 잘 작동하지만, 사과↔오렌지처럼 형태가 매우 다른 경우에는 성능이 떨어질 수 있어요. 도메인 간 구조적 차이가 클수록 학습이 어렵습니다.

## 핵심 정리

| 모델 | 연도 | 핵심 기여 | 특징 |
|------|------|----------|------|
| DCGAN | 2015 | CNN 기반 안정적 GAN 학습 규칙 | BatchNorm, 스트라이드 합성곱 |
| cGAN | 2014 | 조건부 생성 가능 | 클래스/텍스트 조건 추가 |
| WGAN | 2017 | Wasserstein 거리 기반 안정적 학습 | 그래디언트 소실 문제 해결 |
| Pix2Pix | 2016 | 쌍이 있는 이미지 변환 | U-Net 생성자 + PatchGAN |
| CycleGAN | 2017 | 쌍 없는 이미지 변환 | 순환 일관성 손실 |
| StyleGAN | 2018 | 스타일 기반 포토리얼리스틱 생성 | 매핑 네트워크, AdaIN |

## 다음 섹션 미리보기

GAN의 다양한 변형들을 살펴봤으니, 다음 섹션 [GAN 응용](./05-gan-applications.md)에서는 이 모델들이 실제로 어떻게 활용되는지 알아봅니다. 이미지 초해상도, 얼굴 편집, 데이터 증강, 그리고 의료 영상까지 — GAN이 만들어낸 실질적인 가치를 확인해볼까요?

## 참고 자료

- [Radford et al., "Unsupervised Representation Learning with DCGANs" (2015)](https://arxiv.org/abs/1511.06434) - DCGAN 원논문
- [Karras et al., "A Style-Based Generator Architecture for GANs" (2018)](https://arxiv.org/abs/1812.04948) - StyleGAN 원논문
- [Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (2017)](https://arxiv.org/abs/1703.10593) - CycleGAN 원논문
- [Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (2016)](https://arxiv.org/abs/1611.07004) - Pix2Pix 원논문
- [NVlabs StyleGAN3 (GitHub)](https://github.com/NVlabs/stylegan3) - StyleGAN3 공식 구현체
