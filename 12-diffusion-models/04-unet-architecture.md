# U-Net 아키텍처

> 노이즈 예측 네트워크 구조

## 개요

[DDPM](./02-ddpm.md)에서 "노이즈를 예측하는 신경망 $\epsilon_\theta$"가 핵심이라고 했죠. 이 신경망의 정체가 바로 **U-Net**입니다. [세그멘테이션](../08-segmentation/01-semantic-segmentation.md)에서 이미 만났던 U-Net이 Diffusion에서는 어떻게 변형되어 사용되는지, 그리고 Transformer로의 진화까지 다룹니다.

**선수 지식**: [DDPM](./02-ddpm.md), [합성곱 연산](../04-cnn-fundamentals/01-convolution.md), [시맨틱 세그멘테이션의 U-Net](../08-segmentation/01-semantic-segmentation.md)
**학습 목표**:
- 세그멘테이션 U-Net과 Diffusion U-Net의 차이를 이해한다
- 시간 임베딩, 셀프 어텐션, 크로스 어텐션의 역할을 파악한다
- ResNet 블록과 어텐션 블록의 구조를 이해한다
- DiT(Diffusion Transformer)로의 진화 방향을 안다

## 왜 알아야 할까?

U-Net은 Diffusion 모델의 **두뇌**입니다. Stable Diffusion의 추론 시간 중 대부분이 이 U-Net의 연산에 사용되죠. U-Net의 구조를 이해하면 [ControlNet](../14-generative-practice/03-controlnet.md), [LoRA](../14-generative-practice/01-lora.md) 같은 실전 기법이 "어디에, 어떻게" 작용하는지 명확하게 알 수 있습니다.

## 핵심 개념

### 개념 1: 왜 U-Net인가

> 💡 **비유**: 노이즈를 제거하는 것은 마치 **더러운 유리창을 닦는 것**과 같습니다. 먼 거리에서 보면 전체적인 구도(큰 얼룩)를 파악하고, 가까이에서는 세밀한 먼지를 닦아야 하죠. U-Net은 이미지를 **축소(먼 거리)** 했다가 **확대(가까이)**하면서, 큰 구조와 세밀한 디테일을 동시에 처리합니다.

[세그멘테이션](../08-segmentation/01-semantic-segmentation.md)의 U-Net이 "픽셀별 분류"에 사용되었다면, Diffusion의 U-Net은 "픽셀별 노이즈 예측"에 사용됩니다. 둘 다 입력과 출력의 크기가 같아야 하기 때문에 U-Net이 적합하죠.

**세그멘테이션 U-Net vs Diffusion U-Net**:

| 비교 항목 | 세그멘테이션 U-Net | Diffusion U-Net |
|----------|-------------------|----------------|
| **입력** | 이미지 | 노이즈 이미지 + 시간 $t$ |
| **출력** | 클래스 맵 | 예측된 노이즈 $\epsilon$ |
| **추가 입력** | 없음 | 시간 임베딩, 텍스트 조건 |
| **어텐션** | 보통 없음 | 셀프/크로스 어텐션 포함 |
| **정규화** | BatchNorm | GroupNorm |

### 개념 2: Diffusion U-Net의 구조

Diffusion U-Net은 세 부분으로 구성됩니다:

**인코더 (다운샘플링)**
- 해상도를 줄이면서 특징을 추출
- 각 레벨: ResNet 블록 → 어텐션 블록 → 다운샘플

**미들 블록 (병목)**
- 가장 낮은 해상도에서 깊은 처리
- ResNet → 어텐션 → ResNet

**디코더 (업샘플링)**
- 해상도를 키우면서 디테일 복원
- 각 레벨: 업샘플 → ResNet 블록 → 어텐션 블록
- **스킵 연결**: 인코더의 같은 해상도 특징맵을 concat

### 개념 3: 시간 임베딩의 주입 — "지금 몇 단계인지"

[DDPM](./02-ddpm.md)에서 배운 시간 임베딩이 U-Net의 **모든 ResNet 블록**에 주입됩니다:

> 시간 $t$ → 사인/코사인 인코딩 → MLP → 각 ResNet 블록에 더하기

이렇게 하면 같은 네트워크가 $t=999$(거의 노이즈)와 $t=1$(거의 깨끗)에서 서로 다른 행동을 할 수 있습니다.

### 개념 4: 어텐션 블록 — 전역적 관계 포착

[어텐션 메커니즘](../09-vision-transformer/01-attention-mechanism.md)이 U-Net에 핵심적으로 사용됩니다:

**셀프 어텐션(Self-Attention)**:
- 이미지 내부의 장거리 의존성 포착
- "머리카락 색과 눈썹 색을 일치시키기" 같은 전역적 일관성

**크로스 어텐션(Cross-Attention)**:
- 텍스트 조건을 이미지 생성에 반영하는 핵심 메커니즘
- Query: 이미지 특징 → Key, Value: 텍스트 임베딩
- "a red car" → 빨간색이 자동차 영역에 적용되도록 유도

[DETR](../07-object-detection/05-detr.md)과 [Mask2Former](../08-segmentation/03-panoptic-segmentation.md)에서 크로스 어텐션이 쿼리와 이미지를 연결했던 것처럼, 여기서는 텍스트와 이미지를 연결합니다.

> ⚠️ **흔한 오해**: "어텐션은 모든 해상도에서 사용된다" — 아닙니다! 어텐션의 연산량은 $O(n^2)$이므로, 높은 해상도(64×64 이상)에서는 너무 비쌉니다. 보통 16×16, 32×32 같은 **낮은 해상도**에서만 어텐션을 적용합니다.

### 개념 5: DiT — Diffusion Transformer

최근에는 U-Net을 **Transformer**로 대체하는 추세입니다:

**DiT(Diffusion Transformer, 2023)**:
- U-Net 대신 [ViT](../09-vision-transformer/03-vit.md) 구조를 사용
- 이미지를 패치로 나눈 뒤 Transformer로 처리
- 스케일링 법칙(Scaling Law)이 잘 적용됨 — 모델을 키울수록 성능 향상

FLUX와 Stable Diffusion 3는 이 DiT 아키텍처를 채택했습니다. [Swin Transformer](../09-vision-transformer/04-swin-transformer.md)의 효율적인 어텐션 아이디어도 활용되고 있죠.

## 실습: 간소화된 Diffusion U-Net 블록

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """Diffusion U-Net의 ResNet 블록 — 시간 임베딩 주입 포함"""
    def __init__(self, in_ch, out_ch, time_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        # 시간 임베딩을 채널 수에 맞게 프로젝션
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        # 채널 수가 다르면 잔차 연결에 1x1 합성곱 사용
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        # 시간 임베딩 주입 (채널 방향으로 더하기)
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)                    # 잔차 연결


class SelfAttentionBlock(nn.Module):
    """이미지 내 전역적 관계를 포착하는 셀프 어텐션"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)                  # 셀프 어텐션
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h                                # 잔차 연결


class CrossAttentionBlock(nn.Module):
    """텍스트 조건을 이미지에 주입하는 크로스 어텐션"""
    def __init__(self, channels, context_dim=512, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True,
                                          kdim=context_dim, vdim=context_dim)

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        # Query: 이미지, Key/Value: 텍스트
        h, _ = self.attn(h, context, context)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


# 테스트
resblock = ResNetBlock(64, 128, time_dim=128)
self_attn = SelfAttentionBlock(128)
cross_attn = CrossAttentionBlock(128, context_dim=512)

x = torch.randn(2, 64, 32, 32)          # 이미지 특징
t_emb = torch.randn(2, 128)             # 시간 임베딩
context = torch.randn(2, 77, 512)       # 텍스트 임베딩 (77 토큰)

h = resblock(x, t_emb)                  # ResNet + 시간
print(f"ResNet 출력: {h.shape}")         # [2, 128, 32, 32]

h = self_attn(h)                         # 셀프 어텐션
print(f"Self-Attn 출력: {h.shape}")      # [2, 128, 32, 32]

h = cross_attn(h, context)               # 크로스 어텐션 (텍스트 주입)
print(f"Cross-Attn 출력: {h.shape}")     # [2, 128, 32, 32]
```

## 더 깊이 알아보기

### U-Net에서 DiT로의 전환

2023년 이후 Diffusion 모델의 백본이 U-Net에서 DiT로 전환되고 있습니다. 핵심 이유는 **스케일링 법칙(Scaling Law)**입니다.

U-Net은 해상도별로 설계가 고정되어 있어 모델을 키우기 어렵지만, DiT는 Transformer의 레이어 수나 hidden 차원을 늘리는 것만으로 쉽게 스케일업할 수 있죠. GPT가 모델 크기를 키울수록 좋아지듯, DiT도 같은 경향을 보입니다.

### ControlNet이 작동하는 이유

[ControlNet](../14-generative-practice/03-controlnet.md)은 U-Net의 인코더 부분을 **복제**하여, 에지 맵이나 포즈 정보를 주입합니다. U-Net의 스킵 연결 구조 덕분에, 인코더의 출력이 디코더에 직접 전달되어 제어 신호가 효과적으로 반영되는 거죠.

## 흔한 오해와 팁

> 💡 **알고 계셨나요?**: Stable Diffusion 1.5의 U-Net은 약 **8.6억 개**의 파라미터를 가지고 있습니다. SDXL은 26억 개, FLUX는 120억 개로 점점 커지고 있죠. 이 파라미터 대부분이 어텐션 레이어에 집중되어 있습니다.

> 🔥 **실무 팁**: [LoRA](../14-generative-practice/01-lora.md)가 효과적인 이유는 U-Net의 어텐션 레이어(특히 크로스 어텐션의 Q, K, V 프로젝션)만 미세 조정하기 때문입니다. 전체 모델의 일부만 바꿔도 스타일을 크게 변화시킬 수 있어요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| Diffusion U-Net | 노이즈 이미지 + 시간 + 텍스트 조건 → 예측 노이즈 |
| 시간 임베딩 | 사인/코사인 인코딩으로 시점 $t$를 ResNet 블록에 주입 |
| 셀프 어텐션 | 이미지 내 장거리 의존성 포착 |
| 크로스 어텐션 | 텍스트 조건을 이미지 특징에 주입 |
| 스킵 연결 | 인코더→디코더로 세밀한 정보 전달 |
| DiT | U-Net을 Transformer로 대체, 스케일링에 유리 |

## 다음 섹션 미리보기

U-Net이 노이즈를 예측하는 "두뇌"라면, 다음 섹션 [Classifier-Free Guidance](./05-cfg.md)는 이 두뇌에게 "무엇을 생성할지" 방향을 잡아주는 **나침반**입니다. "a beautiful sunset"이라는 텍스트가 어떻게 이미지 생성을 조종하는지 알아봅시다.

## 참고 자료

- [Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)](https://arxiv.org/abs/1505.04597) - 원조 U-Net
- [Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)" (2023)](https://arxiv.org/abs/2212.09748) - DiT 논문
- [Stable Diffusion Series - UNET and CLIP](https://mkdyasserh.github.io/blog/2025/stable-diffusion/) - U-Net 구조의 시각적 해설
- [Encord - Diffusion Transformer](https://encord.com/blog/diffusion-models-with-transformers/) - DiT 아키텍처 해설
