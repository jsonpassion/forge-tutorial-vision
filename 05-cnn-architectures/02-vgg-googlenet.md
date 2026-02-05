# VGG와 GoogLeNet

> 깊이의 힘과 Inception 모듈

## 개요

[AlexNet](./01-lenet-alexnet.md)이 "딥러닝이 동작한다"를 증명한 이후, 2014년 ImageNet 대회에서 두 가지 혁신적 접근법이 동시에 등장합니다. **VGG**는 "단순하게, 하지만 깊게" 전략으로, **GoogLeNet**은 "똑똑하게, 그리고 효율적으로" 전략으로 CNN의 한계를 밀어붙였습니다.

**선수 지식**: [LeNet과 AlexNet](./01-lenet-alexnet.md), [합성곱 연산](../04-cnn-fundamentals/01-convolution.md)
**학습 목표**:
- VGG의 "3×3만 쓰자" 설계 철학을 이해한다
- GoogLeNet의 Inception 모듈과 1×1 합성곱의 역할을 설명할 수 있다
- 두 아키텍처의 트레이드오프를 비교 분석할 수 있다

## 왜 알아야 할까?

VGG는 **전이 학습의 기본 백본**으로 오랫동안 사용되었고, 그 단순한 구조 덕분에 CNN 아키텍처 설계의 교과서 역할을 합니다. GoogLeNet의 Inception 모듈은 "효율적인 설계"라는 화두를 처음 던졌고, 1×1 합성곱은 이후 거의 모든 아키텍처에서 사용됩니다. 둘 다 이해해야 이후 ResNet, EfficientNet 등의 설계 동기를 파악할 수 있습니다.

## 핵심 개념

### 1. VGGNet (2014) — 단순함의 힘

> 💡 **비유**: AlexNet이 이것저것 다양한 크기의 도구를 꺼내 쓴 **만능 공구함**이었다면, VGG는 **3mm 드라이버 하나로 모든 것을 해결한** 장인과 같습니다.

옥스포드 대학 **Visual Geometry Group**의 **카렌 시모니안(Karen Simonyan)**과 **앤드류 지서만(Andrew Zisserman)**이 개발한 VGGNet의 핵심 아이디어는 놀랍도록 단순합니다:

> **"3×3 합성곱만 사용하고, 깊이를 늘리자"**

왜 3×3만 사용할까요? 3×3 두 층을 쌓으면 5×5 하나와 같은 수용 영역(Receptive Field)을 가지면서도:
- **파라미터가 더 적습니다**: 5×5=25 vs 3×3×2=18
- **비선형성이 더 많습니다**: ReLU가 2번 적용 → 더 강한 표현력
- **3개 쌓으면 7×7과 동일**: 파라미터는 27 vs 49, 거의 절반

**VGG16 아키텍처:**

| 블록 | 레이어 | 출력 크기 | 채널 |
|------|--------|----------|------|
| 입력 | - | 224×224 | 3 |
| Block 1 | Conv3×3 × 2 + MaxPool | 112×112 | 64 |
| Block 2 | Conv3×3 × 2 + MaxPool | 56×56 | 128 |
| Block 3 | Conv3×3 × 3 + MaxPool | 28×28 | 256 |
| Block 4 | Conv3×3 × 3 + MaxPool | 14×14 | 512 |
| Block 5 | Conv3×3 × 3 + MaxPool | 7×7 | 512 |
| 분류기 | FC 4096 → 4096 → 1000 | - | - |

채널 수가 64 → 128 → 256 → 512로 **2배씩 증가**하고, 공간 크기는 MaxPool로 **절반씩 감소**하는 규칙적인 패턴이 보이시죠? 이 단순하고 규칙적인 구조가 VGG의 매력입니다.

**VGG의 한계:**
- 파라미터가 **1억 3,800만 개** — FC 레이어에 대부분이 집중
- 메모리 사용량이 매우 큼
- 학습과 추론이 느림

### 2. GoogLeNet / Inception (2014) — 효율적인 깊이

> 💡 **비유**: VGG가 "긴 일직선 도로"라면, GoogLeNet은 **여러 차선이 동시에 달리는 고속도로**와 같습니다. 1차선(1×1), 3차선(3×3), 5차선(5×5)이 동시에 진행한 뒤, 나중에 합류합니다.

구글의 **크리스찬 세게디(Christian Szegedy)** 팀이 만든 GoogLeNet은 VGG와 정반대 접근법을 취했습니다. 깊이(22층)를 늘리면서도 파라미터는 VGG의 **1/20** 수준으로 유지했죠. 그 비결이 **Inception 모듈**입니다.

**Inception 모듈의 구조:**

> 하나의 입력에 **4가지 연산을 동시에 적용**하고 결과를 합침(concat)

| 경로 | 연산 | 역할 |
|------|------|------|
| 경로 1 | 1×1 Conv | 점별(pixelwise) 특성 |
| 경로 2 | 1×1 Conv → 3×3 Conv | 작은 공간 특성 |
| 경로 3 | 1×1 Conv → 5×5 Conv | 큰 공간 특성 |
| 경로 4 | 3×3 MaxPool → 1×1 Conv | 풀링 특성 |

> 4개 경로의 출력을 **채널 축으로 합침(concat)** → 다음 레이어로 전달

핵심은 경로 2, 3, 4 앞에 있는 **1×1 합성곱**입니다. [합성곱 연산](../04-cnn-fundamentals/01-convolution.md)에서 배운 1×1 합성곱은 채널 수를 줄여 연산량을 크게 절감합니다. 이 "병목(bottleneck)" 구조가 GoogLeNet을 효율적으로 만든 핵심입니다.

### 3. 1×1 합성곱의 마법

1×1 합성곱이 왜 중요한지 숫자로 보겠습니다:

**1×1 없이 5×5 합성곱을 적용하는 경우:**
- 입력: 256채널 × 28×28
- 5×5 Conv → 128채널: 연산량 = 256 × 128 × 5 × 5 × 28 × 28 = **약 6.4억**

**1×1로 채널을 먼저 줄이는 경우:**
- 입력: 256채널 × 28×28
- 1×1 Conv → 32채널: 연산량 = 256 × 32 × 28 × 28 = **약 0.06억**
- 5×5 Conv → 128채널: 연산량 = 32 × 128 × 5 × 5 × 28 × 28 = **약 0.8억**
- 합계: **약 0.86억** (원래의 ~13%!)

> 💡 **비유**: 256명이 5가지 시험을 봐야 한다면, 먼저 32명을 선발(1×1)한 뒤 시험(5×5)을 치르는 것이 훨씬 효율적인 것과 같습니다.

### 4. GoogLeNet의 보조 분류기(Auxiliary Classifiers)

GoogLeNet은 22층이나 되기 때문에, 역전파 시 그래디언트가 앞쪽 레이어까지 제대로 전달되지 않는 **기울기 소실** 문제가 걱정되었습니다. 이를 해결하기 위해 네트워크 중간에 **보조 분류기** 2개를 추가했습니다.

보조 분류기는 중간 레이어의 특성 맵으로 직접 분류를 수행하여:
- 그래디언트가 중간 레이어에도 직접 전달되도록 함
- 중간 레이어가 의미 있는 특성을 학습하도록 유도
- 추론 시에는 제거 (학습 시에만 사용)

> 이후 ResNet의 Skip Connection이 등장하면서 보조 분류기는 거의 사라졌습니다.

### 5. VGG vs GoogLeNet — 두 철학의 비교

| 비교 항목 | VGG16 | GoogLeNet |
|-----------|-------|-----------|
| 깊이 | 16층 | 22층 |
| 파라미터 | **1억 3,800만** | **약 700만** (20배 적음) |
| 커널 크기 | 3×3만 | 1×1, 3×3, 5×5 혼합 |
| 설계 철학 | 단순함, 규칙성 | 효율성, 다중 스케일 |
| Top-5 에러 | 7.3% | **6.7%** |
| 전이 학습 | 매우 좋음 (단순 구조) | 복잡해서 변형 어려움 |
| FC 레이어 | 4096 × 2 | **GAP** (파라미터 절약) |

VGG는 구조가 단순해서 이해하고 변형하기 쉬워 **전이 학습 백본**으로 오래 사랑받았고, GoogLeNet은 효율성 면에서 앞섰지만 복잡한 구조 때문에 변형이 어려웠습니다.

## 실습: PyTorch로 구현하기

### VGG 블록 패턴

```python
import torch
import torch.nn as nn

def make_vgg_block(in_channels, out_channels, num_convs):
    """VGG 스타일 블록: Conv3x3 × n + MaxPool"""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))  # 현대적 개선
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2, stride=2))
    return nn.Sequential(*layers)

# VGG16 특성 추출기
vgg16_features = nn.Sequential(
    make_vgg_block(3, 64, 2),    # Block 1: 224→112
    make_vgg_block(64, 128, 2),  # Block 2: 112→56
    make_vgg_block(128, 256, 3), # Block 3: 56→28
    make_vgg_block(256, 512, 3), # Block 4: 28→14
    make_vgg_block(512, 512, 3), # Block 5: 14→7
)

x = torch.randn(1, 3, 224, 224)
out = vgg16_features(x)
print(f"VGG16 특성 맵: {out.shape}")  # [1, 512, 7, 7]
```

### 간단한 Inception 모듈

```python
import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    """GoogLeNet의 Inception 모듈 (간략화)"""
    def __init__(self, in_ch, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        # 경로 1: 1×1 합성곱
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1x1, 1), nn.ReLU(inplace=True)
        )
        # 경로 2: 1×1 → 3×3 합성곱
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, ch3x3_reduce, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, 3, padding=1), nn.ReLU(inplace=True)
        )
        # 경로 3: 1×1 → 5×5 합성곱
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, ch5x5_reduce, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, 5, padding=2), nn.ReLU(inplace=True)
        )
        # 경로 4: MaxPool → 1×1 합성곱
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, 1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 4개 경로를 병렬 실행 후 채널 축으로 합침
        return torch.cat([
            self.branch1(x), self.branch2(x),
            self.branch3(x), self.branch4(x)
        ], dim=1)

# 테스트: Inception 3a 모듈 (GoogLeNet 원본 설정)
inception_3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
x = torch.randn(1, 192, 28, 28)
out = inception_3a(x)
print(f"Inception 출력: {out.shape}")  # [1, 256, 28, 28] (64+128+32+32)
print(f"파라미터: {sum(p.numel() for p in inception_3a.parameters()):,}")
```

### torchvision 사전 학습 모델

```python
import torchvision.models as models

# 사전 학습된 VGG16 불러오기
vgg16 = models.vgg16(weights='IMAGENET1K_V1')

# 특성 추출기만 사용 (전이 학습의 기본)
vgg_backbone = vgg16.features  # Conv 레이어만
print(f"VGG16 전체 파라미터: {sum(p.numel() for p in vgg16.parameters()):,}")

# GoogLeNet (Inception v1)
googlenet = models.googlenet(weights='IMAGENET1K_V1')
print(f"GoogLeNet 전체 파라미터: {sum(p.numel() for p in googlenet.parameters()):,}")
# VGG16: ~138M vs GoogLeNet: ~7M — 약 20배 차이!
```

## 더 깊이 알아보기

### 이름에 담긴 이야기

**VGG**의 이름은 단순합니다 — 옥스포드 대학의 연구 그룹 이름 **V**isual **G**eometry **G**roup의 약자입니다. 카렌 시모니안은 러시아 출신의 연구자로, VGG 논문을 발표한 뒤 딥마인드(DeepMind)에 합류하여 이후 AI 연구를 이어갔습니다.

**GoogLeNet**의 이름에는 재미있는 경의가 담겨 있습니다. Google + LeNet, 즉 **구글이 만든 LeNet에 대한 오마주**입니다. 얀 르쿤의 LeNet이 CNN의 출발점이었음을 인정하면서, 구글이 이를 극적으로 발전시켰다는 의미를 담았죠. 대소문자를 자세히 보면 "Google**N**et"이 아니라 "Google**N**et"임에 주의하세요.

또한 Inception이라는 이름은 크리스토퍼 놀란 감독의 영화 **인셉션(Inception, 2010)**에서 따왔습니다. 영화의 "꿈 속의 꿈" 구조처럼, 네트워크 안에 **작은 네트워크(모듈)**를 넣는다는 의미입니다.

> 💡 **알고 계셨나요?**: VGG의 FC 레이어는 전체 1.38억 파라미터 중 **1.24억 개**(약 90%)를 차지합니다. 이것이 GoogLeNet이 GAP(Global Average Pooling)을 채택한 핵심 이유였고, 이후 사실상 모든 CNN이 GAP을 사용하게 된 계기입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "깊을수록 무조건 좋다" — VGG는 16/19층에서 이미 한계를 보였습니다. 20층 이상으로 깊이를 늘리면 오히려 성능이 **떨어졌습니다**. 이 "성능 저하(degradation)" 문제를 해결한 것이 다음에 배울 [ResNet](./03-resnet.md)입니다.

> 🔥 **실무 팁**: VGG16은 여전히 **스타일 트랜스퍼(Style Transfer)**와 **퍼셉추얼 손실(Perceptual Loss)** 계산에 널리 사용됩니다. 중간 레이어의 특성 맵이 이미지의 "스타일"과 "콘텐츠"를 잘 분리하기 때문입니다. 생성 모델(GAN, Diffusion)에서도 VGG 특성을 손실 함수에 자주 활용합니다.

> 🔥 **실무 팁**: 새로운 아키텍처를 설계할 때, 채널 수를 2배씩 늘리고 공간 크기를 절반으로 줄이는 VGG의 패턴은 여전히 유효한 경험 법칙입니다. ResNet, EfficientNet 등도 이 패턴을 기본적으로 따릅니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| VGG | 3×3 합성곱만으로 깊은 네트워크 구성 (16~19층) |
| 3×3의 장점 | 파라미터 적고, 비선형성 많고, 수용 영역은 동일 |
| GoogLeNet | Inception 모듈로 효율적인 22층 네트워크 |
| Inception 모듈 | 1×1, 3×3, 5×5, Pool 4개 경로 병렬 처리 |
| 1×1 합성곱 | 채널 수 축소(병목)로 연산량 대폭 절감 |
| 보조 분류기 | 깊은 네트워크의 그래디언트 소실 완화 (학습 시만) |
| GAP | FC 레이어 대체로 파라미터 극적 감소 |

## 다음 섹션 미리보기

VGG에서 보았듯이 20층 이상으로 깊이를 늘리면 오히려 성능이 떨어지는 문제가 있었습니다. [ResNet과 Skip Connection](./03-resnet.md)에서는 이 문제를 우아하게 해결한 **잔차 학습(Residual Learning)**을 배웁니다. 이 아이디어 하나가 100층, 1000층 네트워크를 가능하게 만들었습니다.

## 참고 자료

- [Very Deep Convolutional Networks (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556) - VGG 원조 논문
- [Going Deeper with Convolutions (Szegedy et al., 2014)](https://arxiv.org/abs/1409.4842) - GoogLeNet/Inception 원조 논문
- [Understanding VGG - viso.ai](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/) - VGG 아키텍처 상세 해설
- [GoogLeNet Explained - viso.ai](https://viso.ai/deep-learning/googlenet-explained-the-inception-model-that-won-imagenet/) - Inception 모듈의 직관적 설명
