# ConvNeXt

> Transformer 시대의 순수 CNN

## 개요

2020년 Vision Transformer(ViT)가 등장하면서 "CNN의 시대는 끝났다"는 이야기가 나왔습니다. 그런데 2022년, 메타(Meta) AI의 연구진이 "잠깐, Transformer가 잘 되는 이유가 어텐션 때문이 아니라 **설계 원리** 때문이라면?"이라는 질문을 던집니다. 그리고 ResNet을 Transformer의 설계 원리로 하나씩 현대화하여 **순수 CNN만으로 Transformer와 대등한 성능**을 보여준 **ConvNeXt**를 발표합니다.

**선수 지식**: [ResNet](./03-resnet.md), [EfficientNet](./05-efficientnet.md)
**학습 목표**:
- ConvNeXt가 ResNet을 현대화한 핵심 변경 사항들을 이해한다
- Transformer의 어떤 설계 원리가 CNN에 적용되었는지 설명할 수 있다
- "CNN vs Transformer" 논쟁의 현재 위치를 파악한다

## 왜 알아야 할까?

ConvNeXt는 CNN 아키텍처 진화의 **최신 지점**이자, "좋은 아키텍처 설계란 무엇인가?"에 대한 깊은 통찰을 제공합니다. ViT, Swin Transformer 같은 Transformer 기반 모델은 [Vision Transformer](../09-vision-transformer/03-vit.md) 챕터에서 다루게 되는데, ConvNeXt를 먼저 이해하면 "무엇이 Transformer를 강하게 만드는지"를 CNN 관점에서 파악할 수 있습니다.

## 핵심 개념

### 1. "Transformer가 잘 되는 진짜 이유는?"

2020~2021년, ViT와 Swin Transformer가 CNN을 이기는 결과가 쏟아지면서 비전 커뮤니티에 큰 충격을 주었습니다. 그런데 ConvNeXt 저자들의 핵심 의문은 이것이었습니다:

> "Transformer의 성능 향상이 **Self-Attention** 메커니즘 덕분인가, 아니면 **학습 기법과 설계 원리**의 차이 때문인가?"

이 질문에 답하기 위해, ResNet-50에서 시작하여 Swin Transformer의 설계 원리를 **하나씩 적용**하며 성능 변화를 추적했습니다. 결과는 놀라웠습니다 — Self-Attention 없이 순수 합성곱만으로도 Transformer에 필적하는 성능을 달성할 수 있었습니다.

### 2. ResNet → ConvNeXt: 단계별 현대화

> 💡 **비유**: 오래된 집(ResNet)을 리모델링하는 과정과 같습니다. 기둥(합성곱)은 그대로 두고, 벽지(활성화), 조명(정규화), 창문 크기(커널), 방 배치(매크로 설계)를 현대적으로 바꾸는 것이죠.

**Step 1: 학습 기법 현대화** (76.1% → 78.8%)

| 기존 (ResNet) | 현대적 (ConvNeXt) |
|--------------|-------------------|
| 에포크 90 | **에포크 300** |
| SGD | **AdamW** |
| 단순 증강 | **Mixup, CutMix, RandAugment** |
| 없음 | **Stochastic Depth, Label Smoothing** |

학습 기법만 바꿔도 **2.7% 향상**! 아키텍처 변경 전에 이미 큰 차이가 났습니다.

**Step 2: 매크로 설계 변경** (78.8% → 79.4%)

Swin Transformer의 스테이지별 블록 수 비율(1:1:3:1)을 따라, ResNet의 (3,4,6,3)을 **(3,3,9,3)**으로 변경했습니다.

**Step 3: Depthwise Convolution 도입** (79.4% → 80.5%)

일반 합성곱을 **Depthwise Conv + Pointwise Conv**로 분리했습니다. 이는 Transformer의 "공간 정보 처리(Attention)와 채널 정보 처리(FFN)를 분리"하는 구조와 유사합니다.

**Step 4: 커널 크기 확대 → 7×7** (80.5% → 80.6%)

3×3 대신 **7×7 Depthwise Conv**를 사용했습니다. Transformer의 Self-Attention이 글로벌 수용 영역을 갖는 것에 대응하여 수용 영역을 넓힌 것입니다.

> VGG 이후 3×3이 절대 원칙이었는데, Depthwise Conv에서는 커널이 커도 연산량이 적기 때문에 7×7이 가능합니다.

**Step 5: 마이크로 설계 변경** (80.6% → 82.0%)

| 변경 사항 | 기존 | ConvNeXt | Transformer 대응 |
|-----------|------|----------|-----------------|
| 활성화 함수 | ReLU | **GELU** | Transformer가 GELU 사용 |
| 활성화 개수 | 블록당 여러 개 | **1개만** | FFN에 하나만 사용 |
| 정규화 | BatchNorm | **LayerNorm** | Transformer가 LN 사용 |
| 정규화 개수 | 블록당 여러 개 | **1개만** | 최소한의 정규화 |
| 다운샘플링 | 스트라이드 Conv | **별도 레이어** | 패치 병합 |

### 3. ConvNeXt Block — 최종 형태

ConvNeXt의 하나의 블록은 아래와 같이 매우 깔끔합니다:

| 순서 | 레이어 | 역할 |
|------|--------|------|
| 1 | Depthwise Conv 7×7 | 공간 특성 추출 (채널별 독립) |
| 2 | LayerNorm | 정규화 (블록당 1회만) |
| 3 | Linear (1×1 Conv) | 채널 확장 (4배) |
| 4 | GELU | 활성화 (1회만) |
| 5 | Linear (1×1 Conv) | 채널 축소 (원래 크기) |
| + | Skip Connection | 잔차 연결 |

이 구조는 Transformer의 **FFN(Feed-Forward Network)** 블록과 거의 동일합니다. 차이는 Self-Attention 대신 **Depthwise Conv 7×7**을 사용한다는 것뿐입니다.

### 4. ConvNeXt 성능 — Swin Transformer와 대등

| 모델 | 파라미터 | FLOPs | ImageNet Top-1 |
|------|---------|-------|---------------|
| Swin-T | 28M | 4.5G | 81.3% |
| **ConvNeXt-T** | 29M | 4.5G | **82.1%** |
| Swin-S | 50M | 8.7G | 83.0% |
| **ConvNeXt-S** | 50M | 8.7G | **83.1%** |
| Swin-B | 88M | 15.4G | 83.5% |
| **ConvNeXt-B** | 89M | 15.4G | **83.8%** |

비슷한 모델 크기에서 ConvNeXt가 Swin Transformer를 약간 상회합니다! Self-Attention 없이도 이것이 가능했습니다.

### 5. ConvNeXt V2 (2023) — 자기지도 학습과의 결합

ConvNeXt V2는 두 가지를 추가했습니다:

**FCMAE (Fully Convolutional Masked Autoencoder)**: 이미지의 60%를 마스킹하고 복원하는 **자기지도 사전 학습**. MAE(Masked Autoencoder)의 CNN 버전입니다.

**GRN (Global Response Normalization)**: 채널 간 경쟁을 유도하는 새로운 정규화 레이어. SE 블록과 유사한 채널 재조정 효과를 냅니다.

ConvNeXt V2-Huge는 FCMAE 사전 학습으로 ImageNet에서 **88.9%** 정확도를 달성했습니다.

## 실습: PyTorch로 ConvNeXt 블록 구현하기

### ConvNeXt Block

```python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """ConvNeXt의 기본 블록"""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        # 1. Depthwise Conv 7×7
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7,
                                padding=3, groups=dim)  # groups=dim → depthwise
        # 2. LayerNorm (채널 축)
        self.norm = nn.LayerNorm(dim)
        # 3. Pointwise 확장 (4배)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # 4. GELU 활성화
        self.act = nn.GELU()
        # 5. Pointwise 축소
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        # Depthwise Conv (채널별 공간 처리)
        x = self.dwconv(x)
        # (B, C, H, W) → (B, H, W, C) for LayerNorm과 Linear
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = residual + x  # Skip Connection
        return x

# 테스트
block = ConvNeXtBlock(dim=96)
x = torch.randn(1, 96, 56, 56)
out = block(x)
print(f"ConvNeXt Block: {x.shape} → {out.shape}")  # 크기 불변
print(f"파라미터: {sum(p.numel() for p in block.parameters()):,}")
```

### torchvision으로 ConvNeXt 사용

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ConvNeXt 패밀리
for name, model_fn in [
    ('ConvNeXt-Tiny', models.convnext_tiny),
    ('ConvNeXt-Small', models.convnext_small),
    ('ConvNeXt-Base', models.convnext_base),
]:
    model = model_fn(weights='IMAGENET1K_V1')
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:18s} | 파라미터: {params/1e6:.1f}M")

# 출력:
# ConvNeXt-Tiny      | 파라미터: 28.6M
# ConvNeXt-Small     | 파라미터: 50.2M
# ConvNeXt-Base      | 파라미터: 88.6M

# 파인튜닝: 마지막 분류기만 교체
convnext = models.convnext_tiny(weights='IMAGENET1K_V1')
convnext.classifier[2] = nn.Linear(768, 10)  # 10 클래스

# 추론 테스트
convnext.eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = convnext(x)
print(f"출력: {output.shape}")  # [1, 10]
```

### CNN 아키텍처 총정리 비교

```python
import torch
import torchvision.models as models

# 이 챕터에서 배운 모든 아키텍처 비교
architectures = {
    'AlexNet':         models.alexnet,
    'VGG16':           models.vgg16,
    'GoogLeNet':       models.googlenet,
    'ResNet-50':       models.resnet50,
    'DenseNet-121':    models.densenet121,
    'EfficientNet-B0': models.efficientnet_b0,
    'ConvNeXt-Tiny':   models.convnext_tiny,
}

print(f"{'모델':20s} | {'파라미터':>10s}")
print("-" * 35)
for name, model_fn in architectures.items():
    model = model_fn()
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:20s} | {params/1e6:8.1f}M")

# AlexNet              |     61.1M
# VGG16                |    138.4M  ← FC 레이어 때문에 거대
# GoogLeNet            |      6.6M  ← 당시 최고 효율
# ResNet-50            |     25.6M  ← 실무 표준
# DenseNet-121         |      8.0M  ← 특성 재사용으로 효율적
# EfficientNet-B0      |      5.3M  ← NAS 최적화
# ConvNeXt-Tiny        |     28.6M  ← Transformer 경쟁
```

## 더 깊이 알아보기

### "CNN은 죽지 않았다" — 논문의 메시지

ConvNeXt 논문 "A ConvNet for the 2020s"의 제1저자 **류 좡(Zhuang Liu)**은 메타 AI와 UC 버클리의 연구자입니다. 이 논문의 핵심 메시지는 학계에 큰 반향을 일으켰습니다:

"Vision Transformer의 성능 우위는 Self-Attention 자체보다 **학습 기법**(더 긴 학습, 강한 증강, AdamW)과 **설계 원리**(큰 커널, 적은 활성화, LayerNorm)에서 비롯된 부분이 크다."

이 결론은 두 가지 중요한 시사점을 줍니다:
1. **CNN은 아직 발전 여지가 많다** — 10년 전 설계(ResNet)를 그대로 쓰면서 Transformer와 비교하는 것은 공정하지 않다
2. **아키텍처 간 아이디어 교류가 중요하다** — CNN과 Transformer는 경쟁이 아니라 서로 배워야 할 관계

> 💡 **알고 계셨나요?**: ConvNeXt 논문의 가장 큰 기여는 어쩌면 모델 자체보다, **체계적인 제거 연구(Ablation Study)** 방법론입니다. "하나의 변수만 바꾸고 효과를 측정한다"는 접근법은 아키텍처 연구의 모범 사례가 되었습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "ConvNeXt가 Transformer보다 항상 좋다" — 그렇지 않습니다. ConvNeXt는 ImageNet 분류에서 대등하지만, **장거리 의존성**이 중요한 태스크(예: 문서 이해, 의료 영상의 글로벌 패턴)에서는 Transformer가 여전히 유리할 수 있습니다. 태스크에 맞는 선택이 중요합니다.

> 🔥 **실무 팁**: "CNN과 Transformer 중 뭘 써야 할지 모르겠다면?" — 데이터가 적고 빠른 학습이 필요하면 **ConvNeXt**, 대규모 데이터와 사전 학습을 활용하면 **Swin Transformer**, 가장 안전한 선택은 **ResNet-50 + 최신 학습 기법**입니다. ConvNeXt의 교훈처럼, 아키텍처보다 학습 기법이 더 중요한 경우가 많습니다.

> 🔥 **실무 팁**: ConvNeXt에서 BatchNorm을 LayerNorm으로 바꾼 것이 성능에 기여했지만, LayerNorm은 CNN에서 BatchNorm보다 **추론 속도가 느릴 수** 있습니다. 엣지 배포 시에는 이 점을 고려하세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| ConvNeXt | Transformer 설계 원리를 CNN에 적용한 현대화된 순수 CNN |
| 핵심 변경 | 7×7 DWConv, GELU, LayerNorm, 역 병목, 최소 활성화 |
| 학습 기법의 중요성 | ResNet + 현대 학습 기법만으로도 +2.7% 향상 |
| Depthwise Conv 7×7 | 넓은 수용 영역 + 적은 연산량의 조합 |
| ConvNeXt V2 | FCMAE 사전 학습 + GRN. 88.9% 달성 |
| CNN vs Transformer | 경쟁이 아닌 아이디어 교류의 관계 |

## 다음 섹션 미리보기

Chapter 05에서 LeNet부터 ConvNeXt까지 CNN 아키텍처의 진화를 모두 살펴보았습니다. 다음 [이미지 분류 실전](../06-image-classification/01-mnist.md)에서는 이 아키텍처들을 실제로 활용하여 MNIST, CIFAR-10 같은 데이터셋으로 분류 모델을 학습시켜봅니다. 이론에서 실전으로 넘어가는 시간입니다!

## 참고 자료

- [A ConvNet for the 2020s (Liu et al., 2022)](https://arxiv.org/abs/2201.03545) - ConvNeXt 원조 논문 (CVPR 2022)
- [ConvNeXt V2 (Woo et al., 2023)](https://arxiv.org/abs/2301.00808) - FCMAE와 GRN을 추가한 후속 논문
- [ConvNeXt - Hugging Face Course](https://huggingface.co/learn/computer-vision-course/en/unit2/cnns/convnext) - ConvNeXt의 단계별 현대화 과정을 시각적으로 설명
- [ConvNeXt Backbone: Modernizing CNNs](https://www.emergentmind.com/topics/convnext-backbone) - ConvNeXt의 설계 선택과 영향을 분석
