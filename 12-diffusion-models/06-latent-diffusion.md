# Latent Diffusion

> 잠재 공간에서의 확산

## 개요

기존 DDPM은 **픽셀 공간**에서 직접 Diffusion을 수행했기 때문에, 512×512 이미지를 생성하려면 엄청난 연산이 필요했습니다. Latent Diffusion Model(LDM)은 이 문제를 해결한 혁신적인 아이디어로, [VAE](../11-generative-basics/02-vae.md)로 이미지를 먼저 **압축**한 뒤, 그 압축된 잠재 공간에서 Diffusion을 수행합니다. 이것이 바로 **Stable Diffusion의 핵심 원리**입니다.

**선수 지식**: [VAE](../11-generative-basics/02-vae.md), [DDPM](./02-ddpm.md), [U-Net 아키텍처](./04-unet-architecture.md), [CFG](./05-cfg.md)
**학습 목표**:
- 픽셀 공간 vs 잠재 공간 Diffusion의 차이를 이해한다
- LDM의 3요소(VAE, U-Net, 텍스트 인코더)를 파악한다
- VAE 인코더/디코더의 역할을 이해한다
- LDM이 Stable Diffusion으로 이어지는 과정을 안다

## 왜 알아야 할까?

Stable Diffusion을 한 줄로 설명하면 "**Latent Diffusion Model + CLIP 텍스트 인코더**"입니다. LDM을 이해하면 Stable Diffusion의 전체 아키텍처가 명확해지고, [다음 챕터](../13-stable-diffusion/01-sd-architecture.md)의 내용을 훨씬 쉽게 따라갈 수 있어요.

## 핵심 개념

### 개념 1: 픽셀 공간의 한계

> 💡 **비유**: 512×512 컬러 이미지는 786,432개의 숫자(512 × 512 × 3)로 이루어져 있습니다. DDPM은 이 78만 개 숫자 각각에 대해 노이즈를 예측해야 해요. 마치 **서울시 전체의 먼지를 한 알 한 알 치우는 것**처럼 비효율적이죠.

픽셀 공간에서의 문제:
- **연산량 폭발**: 해상도가 2배 → 연산량 4배 이상
- **메모리 부족**: 고해상도 이미지의 U-Net은 GPU 메모리를 초과
- **비효율성**: 인접 픽셀은 대부분 비슷한 값 — 중복 정보가 많음

### 개념 2: Latent Diffusion의 핵심 아이디어

> 💡 **비유**: 서울시 전체의 먼지를 치우는 대신, 먼저 서울의 **축소 모형(1/8 스케일)**을 만들고, 그 모형에서 먼지를 치운 뒤, 다시 실물 크기로 확대하는 겁니다. 훨씬 효율적이죠!

LDM은 3단계로 작동합니다:

**1단계: 인코딩 (VAE Encoder)**
- 512×512×3 이미지 → 64×64×4 잠재 표현
- 공간 크기가 **64배 축소**! (512/8 = 64)

**2단계: 잠재 공간에서 Diffusion**
- 64×64×4 크기에서 노이즈 추가/제거
- 연산량이 픽셀 공간 대비 **약 50배 감소**

**3단계: 디코딩 (VAE Decoder)**
- 64×64×4 잠재 표현 → 512×512×3 이미지
- 고해상도 디테일 복원

> 원본 이미지 → **VAE 인코더** → 잠재 표현(64×64×4) → **Diffusion(U-Net)** → 디노이즈된 잠재 표현 → **VAE 디코더** → 생성된 이미지

### 개념 3: LDM의 세 기둥

LDM은 세 개의 독립적인 모듈로 구성됩니다:

**1. VAE (Variational Autoencoder)**
- 역할: 이미지 ↔ 잠재 공간 변환
- [Ch11에서 배운 VAE](../11-generative-basics/02-vae.md)를 기반으로 하되, VQ 정규화를 사용
- 인코더: 이미지 → 잠재 벡터 (8배 다운샘플)
- 디코더: 잠재 벡터 → 이미지 (8배 업샘플)

**2. U-Net (노이즈 예측기)**
- 역할: 잠재 공간에서 노이즈를 예측
- [이전 섹션](./04-unet-architecture.md)에서 배운 Diffusion U-Net
- 시간 임베딩 + 크로스 어텐션으로 조건 주입

**3. 텍스트 인코더 (CLIP / T5)**
- 역할: 텍스트 프롬프트를 임베딩 벡터로 변환
- [CLIP](../10-vision-language/02-clip.md)의 텍스트 인코더를 사용
- 임베딩은 U-Net의 크로스 어텐션에 주입

| 모듈 | 학습 | 추론 시 역할 |
|------|------|------------|
| VAE | 별도 학습 (고정) | 인코딩/디코딩만 수행 |
| U-Net | Diffusion 학습 | 핵심 — 반복적 노이즈 예측 |
| 텍스트 인코더 | 사전학습 (고정) | 텍스트를 임베딩으로 변환 |

> ⚠️ **흔한 오해**: "Stable Diffusion은 하나의 거대한 모델이다" — 아닙니다! 세 개의 독립적인 모델이 파이프라인으로 연결된 것입니다. VAE와 텍스트 인코더는 Diffusion 학습 시 **고정(frozen)**되어 있고, U-Net만 학습됩니다.

### 개념 4: 잠재 공간의 품질이 핵심

LDM의 성능은 VAE의 품질에 크게 의존합니다. VAE가 이미지를 잘 압축하고 복원하지 못하면, 아무리 U-Net이 잘 학습되어도 최종 이미지가 엉망이 되죠.

Stable Diffusion의 VAE는 다음과 같은 특징을 가집니다:
- **압축률**: 8× 다운샘플 (512→64)
- **잠재 채널**: 4채널 (RGB 3채널이 아닌!)
- **KL 정규화**: 잠재 공간이 너무 불규칙해지지 않도록 약한 KL penalty 적용

### 개념 5: LDM에서 Stable Diffusion으로

2022년 CompVis 팀(하이델베르크 대학)이 LDM 논문을 발표하고, 이를 대규모 데이터셋(LAION-5B)으로 학습한 것이 바로 **Stable Diffusion**입니다.

- **LDM 논문 (2022.01)**: 아키텍처 제안 및 검증
- **Stable Diffusion v1.1~1.5 (2022.08~10)**: LAION 데이터로 대규모 학습
- **Stability AI**: CompVis 팀과 협력하여 오픈소스 공개

이후 SDXL, SD3, FLUX 등으로 발전했지만, 핵심 아이디어 — **"잠재 공간에서 Diffusion"** — 은 동일합니다.

## 실습: LDM 파이프라인 이해하기

```python
import torch
import torch.nn as nn

class SimpleVAEEncoder(nn.Module):
    """간소화된 VAE 인코더 — 이미지를 잠재 공간으로 압축"""
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            # 512 → 256
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            # 256 → 128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            # 128 → 64
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.SiLU(),
            # 최종 잠재 채널
            nn.Conv2d(256, latent_channels * 2, 3, padding=1),  # μ와 σ
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)  # μ와 log(σ²) 분리
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z  # 잠재 표현


class SimpleVAEDecoder(nn.Module):
    """간소화된 VAE 디코더 — 잠재 공간에서 이미지로 복원"""
    def __init__(self, latent_channels=4, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, padding=1),
            nn.SiLU(),
            # 64 → 128
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            # 128 → 256
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            # 256 → 512
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.decoder(z)


class LatentDiffusionPipeline:
    """LDM 생성 파이프라인 — 세 모듈의 협력"""
    def __init__(self, vae_encoder, vae_decoder, unet, text_encoder):
        self.vae_enc = vae_encoder
        self.vae_dec = vae_decoder
        self.unet = unet
        self.text_enc = text_encoder

    def encode_image(self, image):
        """이미지 → 잠재 표현"""
        return self.vae_enc(image)

    def decode_latent(self, latent):
        """잠재 표현 → 이미지"""
        return self.vae_dec(latent)

    def generate(self, prompt_embedding, num_steps=50):
        """텍스트 → 이미지 생성 (개념적 구현)"""
        # 1. 잠재 공간에서 랜덤 노이즈 시작 (64×64×4)
        latent = torch.randn(1, 4, 64, 64)

        # 2. 잠재 공간에서 디노이징 (U-Net 사용)
        for t in reversed(range(num_steps)):
            # U-Net이 잠재 공간의 노이즈를 예측
            # noise_pred = self.unet(latent, t, prompt_embedding)
            # latent = denoise_step(latent, noise_pred, t)
            pass  # 실제로는 DDIM/DPM-Solver 사용

        # 3. 잠재 표현을 이미지로 디코딩
        image = self.decode_latent(latent)
        return image


# 크기 비교 테스트
vae_enc = SimpleVAEEncoder()
vae_dec = SimpleVAEDecoder()

# 픽셀 공간 vs 잠재 공간 크기 비교
image = torch.randn(1, 3, 512, 512)     # 원본 이미지
latent = vae_enc(image)                   # 잠재 표현으로 압축
reconstructed = vae_dec(latent)           # 이미지로 복원

print(f"원본 이미지:    {image.shape}")          # [1, 3, 512, 512]
print(f"잠재 표현:      {latent.shape}")          # [1, 4, 64, 64]
print(f"복원 이미지:    {reconstructed.shape}")   # [1, 3, 512, 512]

pixel_size = 3 * 512 * 512
latent_size = 4 * 64 * 64
print(f"\n픽셀 공간 크기: {pixel_size:,} (786,432)")
print(f"잠재 공간 크기: {latent_size:,} (16,384)")
print(f"압축률: {pixel_size / latent_size:.1f}배 축소!")
```

## 더 깊이 알아보기

### Robin Rombach와 LDM의 탄생

LDM을 제안한 Robin Rombach는 하이델베르크 대학의 박사과정 학생이었습니다. 그의 핵심 통찰은 "고해상도 이미지에서 대부분의 정보는 **지각적으로 무의미한 세부사항**이다"라는 것이었죠.

512×512 이미지의 대부분의 픽셀은 부드러운 그라데이션이나 반복 패턴으로, 핵심적인 의미 정보는 아주 적습니다. VAE로 압축하면 이런 중복을 제거하고 **의미 있는 정보만** 남길 수 있어요. Diffusion은 이 "의미 있는 정보"에만 집중하면 되니까 훨씬 효율적이죠.

이 아이디어 덕분에 수십만 달러의 GPU 클러스터가 아니라, **일반 소비자 GPU**로도 이미지를 생성할 수 있게 되었습니다. 이것이 Stable Diffusion이 "오픈소스 이미지 생성 혁명"을 이끌 수 있었던 이유입니다.

### 잠재 공간의 스케일링 팩터

Stable Diffusion의 VAE는 잠재 표현에 **스케일링 팩터**(약 0.18215)를 곱합니다. 이것은 잠재 벡터의 값 범위를 U-Net이 처리하기 좋은 범위로 조정하기 위함이에요. 이 작은 상수를 무시하면 생성 품질이 크게 떨어지는 — 의외로 중요한 디테일입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "VAE가 이미지 품질의 병목이 아니다" — 실은 그렇습니다! 특히 텍스트나 사람 얼굴의 세밀한 디테일에서, VAE의 압축/복원 과정에서 정보 손실이 발생합니다. SDXL이 더 나은 VAE를 사용하는 것도 이 때문이에요.

> 💡 **알고 계셨나요?**: LDM 논문은 2022년 CVPR에서 **Outstanding Paper Award**를 수상했습니다. Stable Diffusion의 오픈소스 공개는 AI 역사에서 가장 영향력 있는 사건 중 하나로, 수백만 명이 이미지 생성 AI에 접근할 수 있게 되었죠.

> 🔥 **실무 팁**: HuggingFace Diffusers 라이브러리를 사용하면 LDM 파이프라인을 5줄의 코드로 실행할 수 있습니다. `StableDiffusionPipeline.from_pretrained()` 한 줄이면 모든 모듈(VAE, U-Net, 텍스트 인코더)이 자동으로 로드됩니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| Latent Diffusion | 잠재 공간에서 Diffusion을 수행하여 연산 효율 향상 |
| VAE 인코더 | 이미지(512²) → 잠재 표현(64²) 압축 (8× 다운샘플) |
| VAE 디코더 | 잠재 표현 → 이미지 복원 |
| 압축률 | 약 48배 데이터 축소 (786,432 → 16,384) |
| LDM의 3요소 | VAE + U-Net + 텍스트 인코더 |
| Stable Diffusion | LDM을 대규모 데이터로 학습한 모델 |

## 다음 섹션 미리보기

Chapter 12에서 Diffusion 모델의 수학과 핵심 구성요소를 모두 다뤘습니다. 다음 챕터 [Stable Diffusion 심화](../13-stable-diffusion/01-sd-architecture.md)에서는 이 모든 요소가 결합된 실제 Stable Diffusion의 아키텍처를 자세히 살펴보고, SD 1.5부터 SDXL, FLUX까지의 발전 과정을 따라갑니다. 지금까지 배운 이론이 실전에서 어떻게 적용되는지 확인해볼까요?

## 참고 자료

- [Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)](https://arxiv.org/abs/2112.10752) - LDM 원논문 (CVPR 2022 Outstanding Paper)
- [CompVis/stable-diffusion (GitHub)](https://github.com/CompVis/stable-diffusion) - Stable Diffusion 공식 리포지토리
- [How Stable Diffusion Works](https://stable-diffusion-art.com/how-stable-diffusion-work/) - LDM 파이프라인의 시각적 해설
- [Stable Diffusion with Diffusers (HuggingFace)](https://huggingface.co/blog/stable_diffusion) - HuggingFace 기반 실습 가이드
