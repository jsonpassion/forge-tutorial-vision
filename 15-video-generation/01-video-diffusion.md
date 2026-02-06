# 비디오 Diffusion 기초

> 시간 축으로의 확장

## 개요

[인페인팅과 아웃페인팅](../14-generative-practice/06-inpainting-outpainting.md)에서 이미지를 수정하고 확장하는 기술을 배웠습니다. 이제 드디어 **정지 이미지를 넘어 움직이는 영상**의 세계로 들어갑니다. 비디오 Diffusion 모델은 이미지 생성에서 배운 노이즈 제거 원리를 **시간 축**으로 확장하여, 텍스트나 이미지로부터 자연스러운 동영상을 생성합니다.

**선수 지식**: [Diffusion 이론](../12-diffusion-models/01-diffusion-theory.md), [U-Net 아키텍처](../12-diffusion-models/04-unet-architecture.md), [Latent Diffusion](../12-diffusion-models/06-latent-diffusion.md)
**학습 목표**:
- 이미지 Diffusion과 비디오 Diffusion의 차이를 이해한다
- 3D U-Net과 Temporal Attention의 역할을 파악한다
- 비디오 생성의 핵심 과제와 해결 방법을 안다
- 간단한 비디오 생성 코드를 작성할 수 있다

## 왜 알아야 할까?

영상은 단순히 이미지의 연속이 아닙니다. 각 프레임이 **시간적으로 일관성** 있게 연결되어야 자연스러운 동영상이 됩니다. 2024년 이후 Sora, Runway Gen-3, Kling, Veo 등 비디오 생성 AI가 폭발적으로 성장하면서, 영화 프리비즈, 광고 제작, 콘텐츠 크리에이션 분야에서 혁명이 일어나고 있습니다. 이 챕터에서는 그 핵심 원리를 이해합니다.

## 핵심 개념

### 개념 1: 이미지 vs 비디오 — 차원의 확장

> 💡 **비유**: 이미지 생성이 **한 장의 사진을 그리는 것**이라면, 비디오 생성은 **플립북(Flip Book)을 만드는 것**입니다. 각 페이지가 조금씩 다르지만, 넘기면 매끄럽게 움직여야 하죠. 페이지마다 따로 그리면 캐릭터가 순간이동하거나 크기가 변하는 문제가 생깁니다.

**차원 비교:**

| 구분 | 이미지 | 비디오 |
|------|--------|--------|
| **데이터 형태** | (H, W, C) | (T, H, W, C) |
| **연산 차원** | 2D (공간) | 3D (공간 + 시간) |
| **일관성** | 단일 프레임 내부 | 프레임 간 시간적 일관성 |
| **컨볼루션** | 3×3 커널 | 1×3×3 또는 3×3×3 커널 |

여기서 T는 **시간 축(Temporal dimension)**으로, 프레임 수를 의미합니다. 16프레임, 25프레임 등 생성할 영상 길이에 따라 달라지죠.

### 개념 2: 3D U-Net — 공간과 시간을 함께 처리

> 💡 **비유**: 기존 U-Net이 **한 장의 사진을 정밀 스캔하는 MRI 기계**라면, 3D U-Net은 **환자의 움직임까지 촬영하는 동영상 MRI**입니다. 공간적 특징뿐 아니라 시간에 따른 변화도 함께 분석합니다.

[U-Net 아키텍처](../12-diffusion-models/04-unet-architecture.md)에서 배운 인코더-디코더 구조를 떠올려 보세요. 비디오 Diffusion에서는 이를 **시간 축으로 확장**합니다.

**아키텍처 구성:**

> **3D U-Net 구조**
> - 기존 2D Conv (3×3) → **1×3×3 Conv** (공간만 처리)
> - **새로 추가**: Temporal Conv (3×1×1) 또는 Temporal Attention
> - 인코더/디코더 블록마다 **시공간 처리 레이어** 삽입

**Factorized Space-Time 접근:**

Google의 Video Diffusion Models(2022) 논문에서 제안한 방식은 공간과 시간을 **분리해서 처리**합니다:

1. **공간 처리**: 각 프레임을 독립적으로 2D Conv 적용
2. **시간 처리**: 같은 위치의 픽셀들을 시간 축으로 연결하여 Temporal Attention 적용

이렇게 분리하면 계산량을 크게 줄이면서도 시간적 일관성을 학습할 수 있습니다.

### 개념 3: Temporal Attention — 프레임 간 대화

> 💡 **비유**: Temporal Attention은 **편집자가 영화 필름을 검토하는 과정**과 같습니다. "이 장면에서 주인공이 왼쪽으로 걸어갔으니, 다음 장면에서는 더 왼쪽에 있어야 해"라고 프레임들이 서로 정보를 교환하죠.

[어텐션 메커니즘](../09-vision-transformer/01-attention-mechanism.md)에서 Self-Attention이 이미지 내 위치들 간의 관계를 학습한다고 배웠습니다. **Temporal Attention**은 같은 원리를 **시간 축**에 적용합니다.

**작동 방식:**

1. 각 프레임의 같은 공간 위치에서 특징 벡터 추출
2. 시간 축을 따라 Query, Key, Value 계산
3. 어텐션 가중치로 프레임 간 정보 교환
4. **상대적 위치 임베딩**으로 프레임 순서 인식

**공간 어텐션 vs 시간 어텐션:**

| 구분 | Spatial Attention | Temporal Attention |
|------|-------------------|-------------------|
| **대상** | 한 프레임 내 픽셀들 | 여러 프레임의 같은 위치 |
| **배치 축** | 시간 (T) | 공간 (H×W) |
| **학습 내용** | 객체 내부 관계 | 움직임, 변화 패턴 |
| **적용 순서** | 먼저 | 공간 어텐션 후 |

### 개념 4: 비디오 생성의 핵심 과제

비디오 생성이 이미지 생성보다 훨씬 어려운 이유가 있습니다:

**1. 시간적 일관성 (Temporal Consistency)**

> ⚠️ **흔한 문제**: 프레임마다 독립적으로 생성하면 캐릭터의 얼굴이 프레임마다 바뀌거나, 옷의 패턴이 깜빡거리는 **플리커링(Flickering)** 현상이 발생합니다.

**해결 방법:**
- Temporal Attention으로 프레임 간 정보 공유
- Temporal Convolution으로 시간 축 특징 혼합
- 조건부 생성 (첫 프레임 또는 중간 프레임 고정)

**2. 메모리와 계산량**

16프레임 512×512 비디오는 단일 이미지 대비 **16배의 데이터**를 처리해야 합니다. 실제로는 어텐션 연산 때문에 훨씬 더 많은 메모리가 필요하죠.

**해결 방법:**
- Latent Space에서 생성 (Latent Video Diffusion)
- Factorized Attention (공간/시간 분리)
- 프레임 서브샘플링 및 보간

**3. 동작의 자연스러움**

정적인 장면은 쉽지만, 복잡한 동작(춤, 격투, 빠른 카메라 이동)은 학습 데이터도 부족하고 모델링도 어렵습니다.

### 개념 5: Latent Video Diffusion

> 💡 **비유**: 영화 전체를 4K로 편집하면 컴퓨터가 터지니까, **낮은 해상도 프록시 파일로 작업하고 마지막에 고화질로 변환**하는 것처럼, Latent Video Diffusion은 압축된 공간에서 비디오를 생성합니다.

[Latent Diffusion](../12-diffusion-models/06-latent-diffusion.md)에서 배운 것처럼, 픽셀 공간 대신 **잠재 공간(Latent Space)**에서 작업하면 효율성이 크게 향상됩니다.

**비디오용 VAE의 특징:**

- **3D VAE**: 시공간 압축 (T, H, W) → (T', H', W')
- **시간 압축**: 16프레임 → 4 latent 프레임 (4배 압축)
- **공간 압축**: 512×512 → 64×64 (8배 압축)
- 최종적으로 **256배 데이터 감소**

## 실습: 비디오 Diffusion 개념 구현

간단한 Pseudo-3D 블록을 구현해 봅시다:

```python
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """시간 축을 따라 어텐션을 수행하는 레이어"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V 프로젝션
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)

        # 상대적 위치 임베딩 (프레임 순서 인식용)
        self.rel_pos_emb = nn.Parameter(torch.randn(1, num_heads, 32, 32))

    def forward(self, x):
        """
        x: (B, T, H, W, C) - 시공간 텐서
        """
        B, T, H, W, C = x.shape

        # 공간을 배치로 합침: (B*H*W, T, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)

        # Q, K, V 계산
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(-1, T, self.num_heads, self.head_dim)
                      .transpose(1, 2), qkv)

        # 어텐션 계산 (시간 축)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 상대적 위치 바이어스 추가
        rel_pos = self.rel_pos_emb[:, :, :T, :T]
        attn = attn + rel_pos

        attn = attn.softmax(dim=-1)

        # Value 적용
        out = (attn @ v).transpose(1, 2).reshape(B * H * W, T, C)
        out = self.proj_out(out)

        # 원래 형태로 복원: (B, T, H, W, C)
        out = out.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4)

        return out


class Pseudo3DBlock(nn.Module):
    """공간 Conv + 시간 Attention 결합 블록"""
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()

        # 공간 처리 (2D Conv)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        # 시간 처리 (Temporal Attention)
        self.temporal_attn = TemporalAttention(out_channels, num_heads)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
        x: (B, T, C, H, W) - 비디오 텐서
        """
        B, T, C, H, W = x.shape

        # 1. 공간 컨볼루션 (각 프레임 독립)
        x = x.view(B * T, C, H, W)
        x = self.spatial_conv(x)
        _, C_out, H, W = x.shape
        x = x.view(B, T, C_out, H, W)

        # 2. 시간 어텐션 (프레임 간)
        x = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        x = x + self.temporal_attn(self.norm(x))  # 잔차 연결
        x = x.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

        return x


# 테스트
if __name__ == "__main__":
    # 16프레임, 64x64 크기의 비디오 배치
    video = torch.randn(2, 16, 64, 64, 64)  # (B, T, C, H, W)

    block = Pseudo3DBlock(in_channels=64, out_channels=128)
    output = block(video)

    print(f"입력: {video.shape}")   # [2, 16, 64, 64, 64]
    print(f"출력: {output.shape}")  # [2, 16, 128, 64, 64]
    print("✅ Pseudo-3D 블록 동작 확인!")
```

### Diffusers로 비디오 생성 (Text-to-Video)

실제 비디오 생성 모델을 사용해 봅시다:

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# ModelScope의 Text-to-Video 모델 로드
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")

# 메모리 최적화
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# 비디오 생성
prompt = "A cat playing with a ball, high quality, detailed"

video_frames = pipe(
    prompt,
    num_inference_steps=25,
    num_frames=16,          # 생성할 프레임 수
    height=256,
    width=256,
    guidance_scale=9.0
).frames[0]

# 비디오 저장
video_path = export_to_video(video_frames, output_video_path="cat_playing.mp4")
print(f"비디오 저장됨: {video_path}")
```

## 더 깊이 알아보기: Video Diffusion의 역사

**2022년 — Video Diffusion Models의 탄생**

Google Brain의 Jonathan Ho와 Tim Salimans가 2022년 논문 "Video Diffusion Models"에서 첫 비디오 Diffusion 모델을 제안했습니다. 이들은 이미 DDPM(2020)과 Classifier-Free Guidance(2022)를 만든 Diffusion 분야의 선구자들이죠.

핵심 아이디어는 놀랍게도 간단했습니다: **기존 2D U-Net의 컨볼루션을 3D로 확장하고, 시간 축 어텐션을 추가하면 된다**는 것이었죠. 하지만 당시에는 계산 비용이 너무 높아 64×64 해상도의 짧은 영상만 생성할 수 있었습니다.

**2023년 — 실용화의 시작**

Stability AI의 Stable Video Diffusion, Runway의 Gen-2, Pika Labs 등이 등장하면서 비디오 생성이 실용화되기 시작했습니다. 핵심은 **Latent Space에서 작업**하여 계산 효율을 높인 것이었죠.

**2024년 — DiT 혁명**

OpenAI의 Sora가 등장하면서 패러다임이 바뀌었습니다. U-Net 대신 **Diffusion Transformer(DiT)**를 사용하고, 비디오를 **시공간 패치(Spacetime Patches)**로 처리하는 방식이 새로운 표준이 되었습니다. [FLUX와 SD3](../13-stable-diffusion/05-flux.md)에서 배운 DiT가 비디오로 확장된 것이죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "비디오 생성은 이미지를 여러 장 연속으로 만드는 것이다"
>
> 그렇지 않습니다. 프레임마다 독립적으로 생성하면 **플리커링**이 발생합니다. 비디오 모델은 **모든 프레임을 동시에** 생성하면서 시간적 일관성을 유지합니다.

> 💡 **알고 계셨나요?**: 초기 비디오 Diffusion 연구는 64×64 해상도의 5초 영상을 생성하는 데 **NVIDIA A100 8대로 수 시간**이 걸렸습니다. Latent Diffusion과 최적화 기술로 이제는 소비자용 GPU에서도 실시간에 가까운 생성이 가능해졌습니다.

> 🔥 **실무 팁**: 비디오 생성 시 **첫 프레임을 고정(Image-to-Video)**하면 훨씬 안정적인 결과를 얻습니다. 텍스트만으로 생성하면 원하는 스타일이나 구도를 맞추기 어렵거든요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **비디오 텐서** | (T, H, W, C) 형태로 시간 축 추가 |
| **3D U-Net** | 2D Conv + Temporal Conv/Attention으로 확장 |
| **Temporal Attention** | 같은 공간 위치의 프레임 간 정보 교환 |
| **Factorized Attention** | 공간/시간 분리 처리로 계산량 감소 |
| **Latent Video Diffusion** | 3D VAE로 시공간 압축, 효율적 생성 |
| **시간적 일관성** | 프레임 간 연속성 유지가 핵심 과제 |

## 다음 섹션 미리보기

이제 비디오 Diffusion의 기본 원리를 이해했습니다. 다음 섹션 [AnimateDiff](./02-animatediff.md)에서는 **기존 이미지 생성 모델(SD 1.5, SDXL)을 비디오로 확장하는 방법**을 배웁니다. LoRA처럼 작은 모듈만 추가해서 이미지 모델을 비디오 모델로 변환하는 놀라운 기술이죠!

## 참고 자료

- [Video Diffusion Models (NeurIPS 2022)](https://arxiv.org/abs/2204.03458) - 비디오 Diffusion의 시초 논문
- [Lilian Weng - Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/) - 비디오 생성 모델 종합 정리
- [Video Generation Models Explosion 2024](https://yenchenlin.me/blog/2025/01/08/video-generation-models-explosion-2024/) - 2024년 비디오 생성 모델 총정리
- [Diffusion Models for Video Generation - Marvik](https://www.marvik.ai/blog/diffusion-models-for-video-generation) - 비디오 Diffusion 아키텍처 해설
