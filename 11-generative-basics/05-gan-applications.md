# GAN 응용

> 이미지 편집, 스타일 변환

## 개요

GAN의 다양한 변형들을 배웠으니, 이제 실전에서 어떻게 활용되는지 살펴볼 차례입니다. 이미지 초해상도(Super Resolution), 인페인팅(Inpainting), 얼굴 편집, 데이터 증강까지 — GAN은 "그럴듯한 이미지를 만든다"는 능력을 놀라울 정도로 다양한 곳에 적용하고 있습니다.

**선수 지식**: [GAN 기초](./03-gan-basics.md), [GAN 변형들](./04-gan-variants.md)
**학습 목표**:
- GAN 기반 이미지 초해상도(Super Resolution)의 원리를 이해한다
- 인페인팅과 아웃페인팅의 작동 방식을 파악한다
- GAN을 활용한 데이터 증강의 가능성과 한계를 안다
- GAN 기반 응용의 현재 위치와 Diffusion 모델로의 전환을 이해한다

## 왜 알아야 할까?

GAN의 응용 사례를 이해하면 두 가지 이점이 있습니다. 첫째, 생성 모델이 **실제로 어떤 문제를 풀 수 있는지** 감을 잡을 수 있죠. 둘째, 이 응용들의 대부분이 이후 [Diffusion 모델](../12-diffusion-models/01-diffusion-theory.md)과 [Stable Diffusion](../13-stable-diffusion/01-sd-architecture.md)에서도 동일하게 다뤄지기 때문에, GAN에서 먼저 이해하면 이후 학습이 훨씬 수월합니다.

## 핵심 개념

### 개념 1: 이미지 초해상도 — 흐릿한 사진을 선명하게

> 💡 **비유**: 오래된 VHS 테이프의 영상을 4K로 업스케일링한다고 상상해보세요. 단순히 픽셀을 늘리면 뭉개진 이미지가 되지만, GAN은 "고화질 사진은 이런 디테일을 가지고 있다"는 지식을 바탕으로 **없던 디테일을 그럴듯하게 채워넣습니다**.

**SRGAN(Super-Resolution GAN, 2017)**은 이 분야의 선구자입니다:

- **생성자**: 저해상도 이미지를 입력받아 고해상도 이미지를 출력
- **판별자**: 생성된 고해상도와 실제 고해상도를 구별
- **Perceptual Loss**: 픽셀 단위 비교 대신, VGG 네트워크의 특징 맵에서 비교 → 시각적으로 더 자연스러운 결과

이후 **ESRGAN(2018)**이 더 선명한 결과를 달성했고, **Real-ESRGAN(2021)**은 실제 사진의 다양한 열화(압축 아티팩트, 노이즈 등)에도 강건하게 작동합니다.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """SRGAN 생성자의 잔차 블록"""
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)         # 잔차 연결 (ResNet과 동일!)


class SRGenerator(nn.Module):
    """간소화된 SRGAN 생성자 (2배 업스케일)"""
    def __init__(self, scale_factor=2):
        super().__init__()
        # 초기 특징 추출
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),    # 큰 커널로 넓은 맥락 포착
            nn.PReLU()
        )
        # 잔차 블록 — ResNet의 아이디어 활용
        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(5)]
        )
        # 업스케일링 — 서브픽셀 합성곱 사용
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(scale_factor),    # 채널 → 공간 해상도 변환
            nn.PReLU()
        )
        # 최종 출력
        self.output = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        initial = self.initial(x)
        residual = self.residuals(initial)
        upscaled = self.upsample(residual + initial)  # 전역 잔차 연결
        return self.output(upscaled)


# 테스트: 32x32 → 64x64 업스케일
sr_model = SRGenerator(scale_factor=2)
low_res = torch.randn(1, 3, 32, 32)      # 저해상도 입력
high_res = sr_model(low_res)
print(f"저해상도 입력: {low_res.shape}")    # [1, 3, 32, 32]
print(f"고해상도 출력: {high_res.shape}")   # [1, 3, 64, 64]
```

> 🔥 **실무 팁**: `PixelShuffle`은 채널 차원의 정보를 공간 차원으로 재배치하는 연산으로, 전치 합성곱(Transposed Convolution)보다 체커보드 아티팩트가 적어 초해상도에서 선호됩니다. `nn.PixelShuffle(2)`는 채널을 4배 줄이고 높이/너비를 각각 2배로 키웁니다.

### 개념 2: 인페인팅 — 지워진 부분을 복원

> 💡 **비유**: 오래된 벽화의 떨어져 나간 부분을 **복원 전문가**가 주변 맥락을 보고 채워넣는 것과 같습니다. GAN은 주변 픽셀의 패턴을 분석하여 빈 영역에 가장 자연스러운 내용을 생성합니다.

인페인팅(Inpainting)은 이미지의 일부 영역을 자연스럽게 채우는 기술입니다:

- **마스크 영역 제거**: 불필요한 객체(전선, 사람 등)를 지우고 배경으로 채움
- **손상 복원**: 오래된 사진의 긁힘, 얼룩 등을 복원
- **창작 도구**: 이미지의 일부를 새로운 내용으로 교체

대표적인 GAN 기반 인페인팅 모델로는 **DeepFill v2(2019)**와 **LaMa(2022)**가 있습니다. LaMa는 Fast Fourier Convolution을 사용하여 넓은 영역도 일관성 있게 채울 수 있어요.

이 기술은 이후 [인페인팅과 아웃페인팅](../14-generative-practice/06-inpainting-outpainting.md)에서 Diffusion 기반으로 더 발전된 형태를 만나게 됩니다.

### 개념 3: 얼굴 편집과 속성 변환

GAN의 잠재 공간을 탐색하면 이미지의 **특정 속성만 변경**할 수 있습니다:

**InterFaceGAN**: StyleGAN의 잠재 공간에서 "나이", "성별", "안경 착용" 등의 방향을 찾아 이동

> 잠재 벡터 $w$ + $\alpha \cdot \text{나이 방향}$ → 나이를 든/젊은 얼굴로 변환

**StarGAN**: 하나의 모델로 여러 속성(머리색, 표정, 나이 등)을 자유롭게 변환

이런 얼굴 편집 기술은 엔터테인먼트와 AR 필터에서 널리 사용되지만, 동시에 **딥페이크(Deepfake)** 문제를 야기하기도 했습니다. [객체 탐지](../07-object-detection/01-detection-basics.md)와 [분류](../06-image-classification/03-transfer-learning.md) 기술이 딥페이크 탐지에 활용되는 것은 AI의 양면성을 보여주는 흥미로운 사례죠.

### 개념 4: 데이터 증강 — 학습 데이터를 늘리기

> 💡 **비유**: 요리 경연 대회에 참가하려는데 연습용 재료가 부족합니다. GAN은 기존 재료를 바탕으로 **가상의 새로운 재료**를 만들어 연습량을 늘려주는 셈이죠.

GAN을 사용한 데이터 증강은 특히 **데이터가 부족한 분야**에서 강력합니다:

- **의료 영상**: 희귀 질환의 X-ray, MRI, CT 이미지 생성
- **자율주행**: 다양한 기상 조건(안개, 비, 야간)의 도로 이미지 생성
- **위성 영상**: 다양한 토지 유형의 위성 사진 증강
- **제조업 품질 검사**: 결함 이미지가 부족할 때 합성 데이터 생성

```python
import torch
import torch.nn as nn

class SimpleDataAugGAN(nn.Module):
    """데이터 증강을 위한 간단한 조건부 생성자"""
    def __init__(self, latent_dim=100, num_classes=10, img_channels=1):
        super().__init__()
        # 클래스 임베딩: 원하는 클래스의 벡터 표현
        self.label_embed = nn.Embedding(num_classes, 50)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 50, 256),     # 노이즈 + 클래스 정보
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 잠재 벡터와 클래스 임베딩을 결합
        label_vec = self.label_embed(labels)
        combined = torch.cat([z, label_vec], dim=1)
        return self.net(combined).view(-1, 1, 28, 28)


# 원하는 클래스의 이미지를 생성하여 데이터 증강
gen = SimpleDataAugGAN(latent_dim=100, num_classes=10)
z = torch.randn(8, 100)

# "숫자 7" 클래스의 이미지 8장 생성
labels = torch.full((8,), 7, dtype=torch.long)
augmented_7s = gen(z, labels)
print(f"증강된 '7' 이미지: {augmented_7s.shape}")  # [8, 1, 28, 28]

# "숫자 3" 클래스의 이미지 8장 생성
labels = torch.full((8,), 3, dtype=torch.long)
augmented_3s = gen(z, labels)
print(f"증강된 '3' 이미지: {augmented_3s.shape}")  # [8, 1, 28, 28]
```

> ⚠️ **흔한 오해**: "GAN으로 만든 데이터는 실제 데이터와 같다" — GAN 생성 데이터는 학습 데이터의 분포를 모방할 뿐, 완벽한 대체제가 아닙니다. 특히 의료 영상처럼 정밀도가 중요한 분야에서는, GAN 생성 데이터만으로 모델을 학습하면 위험할 수 있어요. 실제 데이터를 **보충**하는 용도로 사용해야 합니다.

### 개념 5: GAN에서 Diffusion으로의 전환

2022년 이후, 이미지 생성의 주류는 GAN에서 **Diffusion 모델**로 빠르게 이동하고 있습니다. 그 이유는 무엇일까요?

| 비교 항목 | GAN | Diffusion |
|----------|-----|-----------|
| **이미지 품질** | 매우 높음 | 매우 높음 (약간 우세) |
| **다양성** | 모드 붕괴 위험 | 높은 다양성 |
| **학습 안정성** | 불안정 | 안정적 |
| **생성 속도** | 매우 빠름 (한 번의 forward) | 느림 (수십~수백 스텝) |
| **조건부 생성** | 추가 설계 필요 | 자연스러운 통합 |
| **텍스트 제어** | 제한적 | 강력 (Classifier-Free Guidance) |

GAN은 **속도**에서 여전히 압도적인 장점을 가지고 있어요. Diffusion이 수십 스텝의 반복이 필요한 반면, GAN은 단 한 번의 forward pass로 이미지를 생성하죠. 그래서 **실시간 처리**가 필요한 응용에서는 GAN이 여전히 선호됩니다.

하지만 텍스트 기반 이미지 생성, 편집, 인페인팅 등 **품질과 제어 가능성**이 중요한 영역에서는 Diffusion 모델이 대세가 되었습니다. [다음 챕터](../12-diffusion-models/01-diffusion-theory.md)에서 이 혁명적인 모델을 자세히 알아볼 예정입니다.

## 더 깊이 알아보기

### GAN이 바꾼 산업들

GAN 기술은 학술 연구를 넘어 다양한 산업에 실질적인 영향을 미쳤습니다:

**게임 & 엔터테인먼트**: NVIDIA의 GauGAN은 간단한 스케치를 사실적인 풍경으로 변환합니다. 게임 에셋 제작 시간을 획기적으로 단축시킨 사례죠.

**패션**: 가상 피팅(Virtual Try-On) 기술에 GAN이 활용됩니다. 옷 이미지와 사람 이미지를 합성하여 실제로 입어보지 않고도 착용 모습을 확인할 수 있죠.

**건축 & 인테리어**: 스케치나 도면을 사실적인 렌더링으로 변환하는 데 Pix2Pix 기반 기술이 사용됩니다.

### Perceptual Loss의 중요성

SRGAN에서 도입된 **Perceptual Loss**는 이후 생성 모델 전반에 큰 영향을 미쳤습니다. 기존의 픽셀 단위 MSE는 "평균적으로 안전한 답"을 선택하여 흐릿한 결과를 만들지만, Perceptual Loss는 VGG 같은 사전학습된 네트워크의 고수준 특징에서 비교하기 때문에 시각적으로 더 자연스러운 결과를 만듭니다.

이 아이디어는 [VAE](./02-vae.md)의 흐릿한 출력 문제를 개선하는 데도 활용되었고, Diffusion 모델의 학습에서도 유사한 개념이 적용됩니다.

## 흔한 오해와 팁

> 💡 **알고 계셨나요?**: Real-ESRGAN은 오픈소스로 공개되어 누구나 사용할 수 있으며, 오래된 사진이나 저화질 영상을 업스케일링하는 데 폭넓게 쓰이고 있습니다. 명령줄 한 줄로 이미지 해상도를 4배 높일 수 있어요. 실제로 많은 영상 복원 유튜브 채널이 이 기술을 활용하고 있죠.

> 🔥 **실무 팁**: GAN 기반 초해상도를 사용할 때, 생성된 디테일은 **그럴듯하지만 실제와 다를 수 있습니다**. 의료 영상이나 위성 사진처럼 정확한 디테일이 중요한 분야에서는 GAN 초해상도의 결과를 그대로 신뢰하면 안 됩니다. 존재하지 않는 패턴이 추가될 수 있기 때문이죠.

## 핵심 정리

| 응용 분야 | 대표 모델 | 핵심 아이디어 |
|----------|----------|-------------|
| 초해상도 | SRGAN, ESRGAN, Real-ESRGAN | 저해상도 → 고해상도, Perceptual Loss |
| 인페인팅 | DeepFill, LaMa | 마스크 영역을 주변 맥락으로 복원 |
| 얼굴 편집 | InterFaceGAN, StarGAN | 잠재 공간 탐색으로 속성 변환 |
| 스타일 변환 | CycleGAN, Pix2Pix | 도메인 간 이미지 변환 |
| 데이터 증강 | cGAN 기반 | 부족한 학습 데이터를 합성으로 보충 |

## 다음 섹션 미리보기

Chapter 11에서 생성 모델의 기초를 다졌습니다 — VAE의 잠재 공간, GAN의 적대적 학습, 그리고 다양한 실전 응용까지. 다음 챕터 [Diffusion 모델](../12-diffusion-models/01-diffusion-theory.md)에서는 현재 이미지 생성의 왕좌를 차지한 Diffusion 모델의 수학과 구현을 깊이 있게 다룹니다. "노이즈에서 예술로"라는 마법 같은 과정이 실제로 어떻게 작동하는지, 함께 알아볼까요?

## 참고 자료

- [Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a GAN" (2017)](https://arxiv.org/abs/1609.04802) - SRGAN 원논문
- [Wang et al., "ESRGAN: Enhanced Super-Resolution GAN" (2018)](https://arxiv.org/abs/1809.00219) - ESRGAN 원논문
- [Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution" (2021)](https://arxiv.org/abs/2107.10833) - Real-ESRGAN 논문
- [Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (2022)](https://arxiv.org/abs/2109.07161) - LaMa 인페인팅 논문
- [GAN Applications for Image Synthesis (XenonStack)](https://www.xenonstack.com/blog/gans-for-image-synthesis) - GAN 응용 종합 가이드
