# EfficientNet

> 복합 스케일링으로 효율성 극대화

## 개요

지금까지 CNN을 개선하는 방법은 주로 "더 깊게"(ResNet), "더 넓게"(WideResNet), "더 큰 이미지로"(Higher Resolution) 중 하나를 선택하는 것이었습니다. **EfficientNet**은 이 세 가지를 **최적의 비율로 동시에 확장**하는 **복합 스케일링(Compound Scaling)** 전략을 제안하여, 기존 모델 대비 훨씬 적은 파라미터로 더 높은 정확도를 달성했습니다.

**선수 지식**: [DenseNet과 SENet](./04-densenet-senet.md), [합성곱 연산](../04-cnn-fundamentals/01-convolution.md)
**학습 목표**:
- 복합 스케일링의 원리와 세 가지 축(깊이, 너비, 해상도)을 이해한다
- MBConv 블록과 SE 블록의 결합을 설명할 수 있다
- EfficientNet 패밀리(B0~B7)의 특성과 선택 기준을 파악한다

## 왜 알아야 할까?

EfficientNet은 "같은 정확도를 **얼마나 적은 자원으로** 달성할 수 있는가"라는 **효율성**의 관점을 CNN 설계의 중심에 놓았습니다. 모바일, 엣지 기기, 비용에 민감한 실무 환경에서 특히 중요한 모델이며, 그 설계 원칙은 이후의 모든 효율적 아키텍처에 영향을 미쳤습니다.

## 핵심 개념

### 1. 스케일링의 세 축 — 무엇을 키울까?

CNN의 성능을 높이려면 모델을 "키워야" 합니다. 키우는 방법은 세 가지입니다:

| 축 | 방법 | 예시 | 효과 |
|----|------|------|------|
| **깊이 (Depth)** | 레이어 수 증가 | ResNet-18 → ResNet-152 | 더 복잡한 특성 학습 |
| **너비 (Width)** | 채널 수 증가 | 64채널 → 256채널 | 더 풍부한 특성 |
| **해상도 (Resolution)** | 입력 이미지 크기 증가 | 224×224 → 380×380 | 더 세밀한 패턴 포착 |

> 💡 **비유**: 카메라의 성능을 높이는 세 가지 방법과 같습니다. **렌즈 수를 늘리거나**(깊이), **각 렌즈를 더 크게 만들거나**(너비), **더 고해상도 센서를 쓰거나**(해상도). 하나만 극단적으로 키우면 비효율적이고, 셋의 **균형이 중요**합니다.

기존 연구들은 이 세 축 중 **하나만** 키우는 경향이 있었습니다. 하지만 실험 결과:
- 깊이만 키우면 → 일정 수준 이후 성능 포화
- 너비만 키우면 → 파라미터가 급격히 증가
- 해상도만 키우면 → 연산량이 기하급수적 증가

### 2. 복합 스케일링(Compound Scaling)

EfficientNet의 핵심 관찰:

> **세 축을 균형 있게 동시에 키우면, 어느 하나만 키우는 것보다 훨씬 효율적이다.**

수식으로 표현하면:

- 깊이: $d = \alpha^\phi$
- 너비: $w = \beta^\phi$
- 해상도: $r = \gamma^\phi$
- 제약 조건: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (연산량이 약 2배 증가)

여기서 $\phi$는 **복합 계수**로, 사용 가능한 자원에 따라 조절합니다. $\phi$를 1 올리면 연산량이 약 2배 증가하면서 세 축이 균형 있게 확장됩니다.

EfficientNet-B0에서 찾은 최적 계수: $\alpha=1.2$, $\beta=1.1$, $\gamma=1.15$

| 모델 | $\phi$ | 해상도 | 파라미터 | Top-1 정확도 |
|------|--------|--------|---------|------------|
| B0 | 0 | 224 | 5.3M | 77.1% |
| B1 | 1 | 240 | 7.8M | 79.1% |
| B2 | 2 | 260 | 9.2M | 80.1% |
| B3 | 3 | 300 | 12M | 81.6% |
| B4 | 4 | 380 | 19M | 82.9% |
| B5 | 5 | 456 | 30M | 83.6% |
| B6 | 6 | 528 | 43M | 84.0% |
| B7 | 7 | 600 | 66M | 84.4% |

B0부터 B7까지, 복합 계수만 바꾸면 자동으로 세 축이 확장됩니다!

### 3. MBConv — EfficientNet의 빌딩 블록

EfficientNet의 기본 블록은 **MBConv(Mobile Inverted Bottleneck Convolution)**입니다. MobileNetV2에서 유래한 이 블록은:

| 순서 | 레이어 | 역할 |
|------|--------|------|
| 1 | Conv 1×1 + BN + Swish | 채널 **확장** (expand ratio, 보통 6배) |
| 2 | Depthwise Conv 3×3/5×5 + BN + Swish | 공간 특성 추출 (채널별 독립) |
| 3 | SE 블록 | 채널 어텐션 |
| 4 | Conv 1×1 + BN | 채널 **축소** (원래 크기로) |
| + | Skip Connection | 잔차 연결 (입출력 크기 같을 때) |

**핵심 특징:**
- **역 병목(Inverted Bottleneck)**: 일반 ResNet 병목과 반대로, 중간이 넓고 양끝이 좁음
- **Depthwise Separable Convolution**: 일반 합성곱 대비 연산량 ~1/9
- **SE 블록**: [DenseNet과 SENet](./04-densenet-senet.md)에서 배운 채널 어텐션
- **Swish 활성화**: $x \cdot \sigma(x)$, ReLU보다 부드러워 깊은 네트워크에 유리

### 4. NAS — 사람이 아닌 AI가 설계한 아키텍처

EfficientNet-B0의 구조는 사람이 설계한 것이 아니라 **Neural Architecture Search(NAS)**로 자동 탐색되었습니다.

> 💡 **비유**: 건축가가 직접 도면을 그리는 것(수동 설계)이 아니라, AI에게 "이 예산으로 가장 좋은 집을 설계해줘"라고 맡기는 것(NAS)과 같습니다.

구글의 **AutoML MNAS 프레임워크**를 사용하여 정확도와 효율성(FLOPS)을 동시에 최적화한 결과, 7개 스테이지로 구성된 B0 베이스라인이 탄생했습니다. 이 B0에 복합 스케일링을 적용한 것이 B1~B7입니다.

### 5. EfficientNetV2 (2021) — 더 빠른 학습

EfficientNet의 후속작인 **EfficientNetV2**는 두 가지 핵심 개선을 도입했습니다:

**Fused-MBConv**: 초기 레이어에서 Depthwise Conv + 1×1 Conv를 하나의 **일반 3×3 Conv로 통합**. GPU에서의 실행 속도가 크게 향상됩니다.

**점진적 학습(Progressive Learning)**: 학습 초반에는 **작은 이미지 + 약한 정규화**로, 후반에는 **큰 이미지 + 강한 정규화**로 학습. 학습 속도가 5~11배 빨라집니다.

| 모델 | 파라미터 | ImageNet Top-1 | 학습 속도 |
|------|---------|---------------|----------|
| EfficientNetV2-S | 22M | 83.9% | V1 대비 5× 빠름 |
| EfficientNetV2-M | 54M | 85.1% | V1 대비 빠름 |
| EfficientNetV2-L | 120M | 85.7% | V1 대비 빠름 |

## 실습: PyTorch로 EfficientNet 사용하기

### MBConv 블록 구현

```python
import torch
import torch.nn as nn

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (간략화)"""
    def __init__(self, in_ch, out_ch, expand_ratio=6, stride=1, se_ratio=0.25):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        # 1. 확장 (expand)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(inplace=True),  # Swish 활성화
            ]
        # 2. Depthwise Conv
        layers += [
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                      padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        ]
        # 3. SE 블록 (간략화)
        se_ch = max(1, int(in_ch * se_ratio))
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, se_ch, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_ch, mid_ch, 1),
            nn.Sigmoid(),
        ]
        self.se = nn.Sequential(*layers[-(5):])
        self.main = nn.Sequential(*layers[:-(5)])
        # 4. 축소 (project)
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.main(x)
        # SE attention
        se_weight = self.se(out)
        out = out * se_weight
        out = self.project(out)
        if self.use_residual:
            out = out + x  # Skip Connection
        return out
```

### torchvision으로 EfficientNet 사용

```python
import torch
import torchvision.models as models

# EfficientNet-B0 (가장 가벼움)
effnet_b0 = models.efficientnet_b0(weights='IMAGENET1K_V1')
print(f"B0 파라미터: {sum(p.numel() for p in effnet_b0.parameters()):,}")
# ~5,288,548

# EfficientNet-B4 (성능/효율 균형)
effnet_b4 = models.efficientnet_b4(weights='IMAGENET1K_V1')
print(f"B4 파라미터: {sum(p.numel() for p in effnet_b4.parameters()):,}")
# ~19,341,616

# EfficientNetV2-S (빠른 학습)
effnet_v2s = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
print(f"V2-S 파라미터: {sum(p.numel() for p in effnet_v2s.parameters()):,}")

# 10 클래스 파인튜닝
effnet_b0.classifier[1] = nn.Linear(1280, 10)

# 추론 테스트
effnet_b0.eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = effnet_b0(x)
print(f"출력: {output.shape}")  # [1, 10]
```

### EfficientNet vs ResNet 비교 실험

```python
import torch
import torchvision.models as models

# 파라미터 수와 정확도 비교
models_info = {
    'ResNet-50': (models.resnet50, 'IMAGENET1K_V2'),
    'EfficientNet-B0': (models.efficientnet_b0, 'IMAGENET1K_V1'),
    'EfficientNet-B4': (models.efficientnet_b4, 'IMAGENET1K_V1'),
    'EfficientNetV2-S': (models.efficientnet_v2_s, 'IMAGENET1K_V1'),
}

for name, (model_fn, weights) in models_info.items():
    model = model_fn(weights=weights)
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:20s} | 파라미터: {params/1e6:6.1f}M")

# 출력 예시:
# ResNet-50            | 파라미터:  25.6M
# EfficientNet-B0      | 파라미터:   5.3M  ← 5배 적지만 비슷한 정확도!
# EfficientNet-B4      | 파라미터:  19.3M  ← ResNet-50보다 적지만 더 높은 정확도
# EfficientNetV2-S     | 파라미터:  21.5M
```

## 더 깊이 알아보기

### 구글 브레인의 "자동 설계" 철학

EfficientNet은 구글 브레인의 **밍싱 탄(Mingxing Tan)**과 **쿠옥 르(Quoc V. Le)**가 개발했습니다. 쿠옥 르는 구글 브레인의 리더이자 **AutoML** 프로젝트의 핵심 인물로, "AI가 AI를 설계하게 하자"는 철학을 추구해왔습니다.

EfficientNet의 흥미로운 점은, 기본 아키텍처(B0)를 NAS로 자동 탐색한 뒤, 스케일링 법칙도 실험적으로 발견했다는 것입니다. "어떤 비율로 키워야 최적인가?"를 수학적으로 유도한 것이 아니라, **수천 번의 실험을 통해 경험적으로** 찾아냈습니다.

이 접근법은 큰 논쟁을 불러일으키기도 했습니다. "인간이 직접 설계한 아키텍처가 AI에 의한 것보다 본질적으로 나은가?"라는 질문인데, 적어도 EfficientNet의 결과는 NAS의 손을 들어주었습니다.

> 💡 **알고 계셨나요?**: EfficientNet-B7은 당시 최고 성능 모델(GPipe)보다 **8.4배 작고 6.1배 빠르면서도** 비슷한 정확도를 달성했습니다. "크고 비싼 모델 = 좋은 모델"이라는 통념에 정면으로 도전한 결과입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "EfficientNet-B7이 항상 B0보다 좋다" — 데이터셋이 작거나 학습 시간이 제한적이면, 큰 모델은 오히려 과적합되거나 학습이 충분히 이루어지지 않습니다. CIFAR-10 같은 작은 데이터셋에서는 B0~B2가 더 적합할 수 있습니다.

> 🔥 **실무 팁**: 실무에서 EfficientNet을 사용할 때의 **선택 가이드**: 빠른 프로토타이핑엔 **B0**, 일반적 분류 태스크엔 **B3~B4**, 최고 성능이 필요하면 **B5~B7**, 학습 속도가 중요하면 **EfficientNetV2-S**를 추천합니다.

> 🔥 **실무 팁**: EfficientNet은 입력 해상도가 모델마다 다릅니다(B0: 224, B4: 380, B7: 600). 파인튜닝 시 해당 모델의 **원래 해상도**에 맞춰 이미지를 리사이즈하는 것이 중요합니다. 해상도를 잘못 설정하면 성능이 크게 떨어질 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 복합 스케일링 | 깊이, 너비, 해상도를 균형 있게 동시 확장 |
| 복합 계수 φ | 자원에 따라 세 축의 확장 정도를 제어하는 단일 파라미터 |
| MBConv | 역 병목 + Depthwise Conv + SE 블록 조합의 빌딩 블록 |
| NAS | Neural Architecture Search로 B0 베이스라인 자동 탐색 |
| Swish/SiLU | $x \cdot \sigma(x)$, ReLU보다 부드러운 활성화 함수 |
| EfficientNetV2 | Fused-MBConv + 점진적 학습으로 학습 속도 대폭 향상 |
| B0~B7 | 복합 계수에 따라 자동 확장되는 모델 패밀리 |

## 다음 섹션 미리보기

2020년 Vision Transformer(ViT)가 등장하면서 "CNN의 시대는 끝났나?"라는 질문이 제기되었습니다. [ConvNeXt](./06-convnext.md)에서는 Transformer의 설계 원리를 CNN에 적용하여 "순수 CNN도 Transformer와 경쟁할 수 있다"는 것을 증명한 최신 아키텍처를 만납니다.

## 참고 자료

- [EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) - 복합 스케일링을 제안한 원조 논문
- [EfficientNetV2: Smaller Models and Faster Training (Tan & Le, 2021)](https://arxiv.org/abs/2104.00298) - V2의 점진적 학습과 Fused-MBConv
- [EfficientNet Explained - Aman Arora](https://amaarora.github.io/posts/2020-08-13-efficientnet.html) - 복합 스케일링을 시각적으로 설명하는 블로그
- [Google AI Blog: EfficientNet](https://research.google/blog/efficientnet-improving-accuracy-and-efficiency-through-automl-and-model-scaling/) - 구글 공식 블로그의 EfficientNet 소개
