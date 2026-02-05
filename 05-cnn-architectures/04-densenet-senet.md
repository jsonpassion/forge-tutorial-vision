# DenseNet과 SENet

> 밀집 연결과 채널 어텐션

## 개요

[ResNet](./03-resnet.md)의 Skip Connection이 "입력을 출력에 **더하는**" 방식이었다면, **DenseNet**은 "모든 이전 레이어를 **연결(concatenate)**하는" 더 극단적인 접근법을 취합니다. 한편 **SENet**은 전혀 다른 방향에서, 채널마다 **중요도를 다르게 부여**하는 어텐션(Attention) 메커니즘을 제안합니다.

**선수 지식**: [ResNet과 Skip Connection](./03-resnet.md)
**학습 목표**:
- DenseNet의 밀집 연결 구조와 성장률(Growth Rate)을 이해한다
- SENet의 Squeeze-and-Excitation 블록의 동작을 설명할 수 있다
- 두 아이디어를 기존 아키텍처에 적용하는 방법을 익힌다

## 왜 알아야 할까?

DenseNet의 **특성 재사용(Feature Reuse)** 아이디어는 파라미터 효율성을 극적으로 높였고, SENet의 **채널 어텐션**은 이후 거의 모든 CNN과 Transformer에 영향을 미쳤습니다. 특히 SENet의 어텐션 개념은 [Vision Transformer](../09-vision-transformer/01-attention-mechanism.md)를 이해하기 위한 첫 단추이기도 합니다.

## 핵심 개념

### 1. DenseNet (2017) — 모든 레이어가 모든 레이어와 대화한다

> 💡 **비유**: ResNet이 릴레이 경주(바통을 다음 주자에게 전달 + 지름길)라면, DenseNet은 **단체 카톡방**과 같습니다. 모든 참가자(레이어)가 이전 참가자들의 메시지를 **전부 볼 수 있고**, 자기 메시지도 이후 모든 참가자에게 공유됩니다.

**핵심 차이: 덧셈 vs 연결**

| 방식 | 연산 | 정보 보존 |
|------|------|----------|
| ResNet | $y = F(x) + x$ (덧셈) | 입력 정보가 변환에 **섞임** |
| DenseNet | $y = [x_0, x_1, ..., x_l]$ (연결) | 모든 이전 특성이 **그대로 보존** |

DenseNet에서 $l$번째 레이어는 이전 **모든 레이어의 출력을 채널 축으로 합쳐서** 입력으로 받습니다:

$$x_l = H_l([x_0, x_1, x_2, ..., x_{l-1}])$$

$L$개 레이어가 있으면 연결의 수는 $L(L+1)/2$개 — ResNet의 $L$개와 비교하면 훨씬 조밀합니다.

### 2. 성장률(Growth Rate)과 Dense Block

DenseNet에서 각 레이어는 **k개의 특성 맵**만 새로 만듭니다. 이 k를 **성장률(Growth Rate)**이라 합니다.

> 💡 **비유**: 팀 프로젝트에서 각 팀원이 매번 보고서 **전체를 다시 쓰는 것**(일반 CNN)이 아니라, 기존 보고서에 **새 페이지 k장만 추가**하는 것(DenseNet)입니다. 이전 내용은 그대로 유지되니 중복 작업이 없죠.

예를 들어 성장률 k=32, 입력 채널 64인 Dense Block에서:
- 레이어 1 출력: 64 + 32 = 96 채널
- 레이어 2 출력: 96 + 32 = 128 채널
- 레이어 3 출력: 128 + 32 = 160 채널
- 레이어 4 출력: 160 + 32 = 192 채널

채널이 계속 늘어나므로, Dense Block 사이에 **Transition Layer**(1×1 Conv + AvgPool)를 넣어 채널 수를 절반으로 줄입니다.

**DenseNet 전체 구조:**

| 구성 요소 | 역할 |
|-----------|------|
| Dense Block | BN-ReLU-Conv 반복, 모든 이전 출력과 연결 |
| Transition Layer | 1×1 Conv(채널 축소) + 2×2 AvgPool(공간 축소) |
| 반복 | Dense Block → Transition → Dense Block → ... |
| 마지막 | GAP → FC |

**DenseNet의 장점:**
- **특성 재사용**: 이전 레이어의 특성을 버리지 않고 재활용 → 파라미터 효율 극대화
- **기울기 흐름 개선**: 모든 레이어가 손실 함수와 직접 연결
- **적은 파라미터**: DenseNet-121(8M)이 ResNet-50(25.6M)보다 파라미터가 적으면서 비슷한 성능

### 3. SENet (2018) — "어떤 채널이 중요할까?"

> 💡 **비유**: TV 리모컨의 **볼륨 조절**과 같습니다. 모든 채널(TV 채널이 아니라 특성 맵의 채널)의 소리를 똑같이 내는 대신, 중요한 채널은 볼륨을 높이고 덜 중요한 채널은 줄이는 거죠.

SENet의 **Squeeze-and-Excitation(SE) 블록**은 단 두 단계로 동작합니다:

**Step 1: Squeeze (압축)** — Global Average Pooling으로 각 채널의 전체 정보를 하나의 숫자로 요약

> 입력: C×H×W → Squeeze → C×1×1

**Step 2: Excitation (자극)** — FC 레이어 2개로 각 채널의 중요도(0~1)를 학습

> C×1×1 → FC(C/r) → ReLU → FC(C) → Sigmoid → **채널별 가중치**

- $r$: 축소 비율 (보통 16). 연산량 절약을 위해 중간에 채널을 줄였다 복원
- 출력: 각 채널에 곱할 **스칼라 가중치** (0~1 사이)

**Step 3: Scale** — 원래 특성 맵에 채널별 가중치를 곱함

> 최종 출력 = 원래 특성 맵 × 채널 가중치

이렇게 하면 네트워크가 "이 이미지에서 **에지 채널**은 중요하고 **색상 채널**은 덜 중요하다"와 같은 판단을 **자동으로** 학습합니다.

### 4. SE 블록의 위력 — 어디에나 붙일 수 있다

SE 블록의 가장 큰 장점은 **기존 아키텍처에 플러그인처럼 추가**할 수 있다는 것입니다:

| 모델 | + SE 블록 | Top-1 에러 개선 | 추가 파라미터 |
|------|----------|----------------|-------------|
| ResNet-50 | SE-ResNet-50 | 7.48% → 6.62% | +2.5M (~10%) |
| ResNet-101 | SE-ResNet-101 | 6.52% → 5.99% | 미미 |
| Inception | SE-Inception | 향상 | 미미 |

파라미터 증가는 미미한데, 성능 향상은 상당합니다. 이것이 SE 블록이 "효율적인 플러그인"으로 사랑받는 이유입니다.

> ⚠️ **흔한 오해**: "어텐션은 Transformer에서 시작되었다" — 채널 어텐션은 SENet(2017)이 먼저 제안했고, Vision 분야에서 "어텐션"이라는 개념을 대중화한 것도 SENet입니다. Transformer의 Self-Attention과는 메커니즘이 다르지만, "중요한 것에 집중한다"는 철학은 같습니다.

## 실습: PyTorch로 구현하기

### DenseNet의 Dense Block

```python
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    """DenseNet의 단일 레이어: BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)"""
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        # Bottleneck: 4*k 채널로 줄인 뒤 3×3 적용
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)  # 이전 + 새 특성 연결!


class DenseBlock(nn.Module):
    """Dense Block: n개의 DenseLayer를 밀집 연결"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# 테스트: 64채널 입력, 성장률 32, 4개 레이어
dense_block = DenseBlock(64, growth_rate=32, num_layers=4)
x = torch.randn(1, 64, 32, 32)
out = dense_block(x)
print(f"입력: {x.shape} → 출력: {out.shape}")
# [1, 64, 32, 32] → [1, 192, 32, 32]  (64 + 32×4 = 192)
```

### SE 블록

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 블록"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # GAP: C×H×W → C×1×1
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),  # 0~1 사이 가중치
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # Squeeze: 각 채널의 글로벌 정보 압축
        y = self.squeeze(x).view(b, c)
        # Excitation: 채널별 중요도 학습
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: 원래 특성에 채널 가중치 곱하기
        return x * y

# 테스트
se = SEBlock(channels=256, reduction=16)
x = torch.randn(2, 256, 14, 14)
out = se(x)
print(f"SE 블록: {x.shape} → {out.shape}")  # 크기 불변!
print(f"SE 파라미터: {sum(p.numel() for p in se.parameters()):,}")
# 256*(256/16) + (256/16)*256 = 8,192개 (매우 적음!)
```

### SE-ResNet: ResNet에 SE 블록 추가

```python
import torch
import torch.nn as nn

class SEBasicBlock(nn.Module):
    """SE 블록이 추가된 ResNet Basic Block"""
    def __init__(self, in_ch, out_ch, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch, reduction)  # SE 블록 추가!

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)      # 채널 어텐션 적용!
        out += identity
        return self.relu(out)

# 테스트
se_block = SEBasicBlock(64, 128, stride=2)
x = torch.randn(1, 64, 32, 32)
print(f"SE-ResNet Block: {x.shape} → {se_block(x).shape}")
```

## 더 깊이 알아보기

### DenseNet — CVPR 2017 최우수 논문

DenseNet은 코넬 대학의 **가오 황(Gao Huang)**과 페이스북 AI의 **로런스 반 데르 마텐(Laurens van der Maaten)** 등이 개발하여 CVPR 2017에서 **최우수 논문상(Best Paper Award)**을 수상했습니다. 로런스 반 데르 마텐은 차원 축소 기법인 **t-SNE**의 저자로도 유명합니다.

DenseNet의 탄생 과정도 흥미롭습니다. ResNet의 Skip Connection이 성공한 이유를 분석하던 중, "그렇다면 **모든 레이어를 모든 레이어에** 연결하면 어떨까?"라는 자연스러운 질문에서 출발했다고 합니다. 실험 결과 놀랍게도, 이 과감한 연결이 파라미터를 **줄이면서도** 성능을 높이는 결과를 보여주었죠.

### SENet — ILSVRC 2017 마지막 우승자

SENet은 중국의 모멘타(Momenta) 회사의 **후 제(Jie Hu)** 등이 개발하여 **ILSVRC 2017 ImageNet 대회의 마지막 우승자**가 되었습니다 (2017년 이후 대회가 종료). Top-5 에러율 **2.251%**를 달성하며, 이전 우승 대비 25%의 상대적 개선을 이뤄냈습니다.

> 💡 **알고 계셨나요?**: SE 블록의 "채널에 가중치를 부여한다"는 아이디어는 이후 **CBAM**(채널 + 공간 어텐션), **ECA-Net**(효율적 채널 어텐션) 등 다양한 변형을 낳았고, 이 계보는 결국 Transformer의 Self-Attention까지 이어집니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "DenseNet은 메모리를 많이 쓴다" — 구현에 따라 다릅니다. 연결(concat) 연산이 많아 중간 특성 맵을 모두 저장해야 하므로 메모리가 많이 필요한 것은 사실입니다. 하지만 **체크포인팅(checkpointing)** 기법을 사용하면 메모리 사용량을 크게 줄일 수 있습니다.

> 🔥 **실무 팁**: SE 블록을 추가할 때 `reduction=16`이 기본값이지만, 채널 수가 적은(64 이하) 초기 레이어에서는 `reduction=4`나 `reduction=8`이 더 나을 수 있습니다. 축소가 너무 극단적이면 정보 손실이 생기기 때문입니다.

> 🔥 **실무 팁**: DenseNet-121은 **의료 영상 분류**에서 특히 인기가 많습니다. 적은 파라미터로 좋은 성능을 내고, 특성 재사용 덕분에 작은 데이터셋에서도 잘 작동하기 때문입니다. 의료 AI 논문에서 DenseNet을 자주 만나게 될 겁니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| DenseNet | 모든 이전 레이어의 출력을 채널 연결(concat)하는 구조 |
| 성장률 (Growth Rate) | 각 레이어가 추가하는 새 채널 수 (보통 k=32) |
| Transition Layer | Dense Block 사이에서 채널과 공간 크기를 줄이는 레이어 |
| 특성 재사용 | 이전 특성을 버리지 않고 재활용 → 파라미터 효율화 |
| SE 블록 | Squeeze(GAP) + Excitation(FC×2) → 채널별 가중치 |
| 채널 어텐션 | 중요한 채널의 가중치를 높이고, 덜 중요한 채널은 낮춤 |
| 플러그인 설계 | SE 블록은 기존 아키텍처에 쉽게 추가 가능 |

## 다음 섹션 미리보기

지금까지 CNN의 깊이, 연결 방식, 어텐션을 개선하는 방법을 배웠습니다. 그런데 "깊이, 너비, 해상도 중 어떤 것을 늘려야 가장 효율적일까?"라는 근본적 질문에 답한 모델이 있습니다. [EfficientNet](./05-efficientnet.md)에서는 이 세 축을 **최적의 비율로 동시에 확장**하는 **복합 스케일링(Compound Scaling)** 전략을 배웁니다.

## 참고 자료

- [Densely Connected Convolutional Networks (Huang et al., 2017)](https://arxiv.org/abs/1608.06993) - DenseNet 원조 논문 (CVPR 2017 Best Paper)
- [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507) - SENet 원조 논문, ILSVRC 2017 우승
- [Channel Attention and SENet - DigitalOcean](https://www.digitalocean.com/community/tutorials/channel-attention-squeeze-and-excitation-networks) - SE 블록의 직관적 설명
- [DenseNet Explained - GeeksforGeeks](https://www.geeksforgeeks.org/computer-vision/densenet-explained/) - DenseNet 구조를 시각적으로 설명
