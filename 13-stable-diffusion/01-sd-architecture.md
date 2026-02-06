# Stable Diffusion 아키텍처

> VAE, U-Net, CLIP 텍스트 인코더

## 개요

[이전 챕터](../12-diffusion-models/06-latent-diffusion.md)에서 Latent Diffusion Model(LDM)의 이론을 배웠습니다. 이제 그 이론이 실제 제품으로 어떻게 구현되었는지 살펴볼 차례입니다. Stable Diffusion은 **VAE + U-Net + CLIP 텍스트 인코더**라는 세 기둥 위에 세워진 모델로, 이 세 모듈의 협력이 텍스트에서 이미지를 만들어내는 마법의 비결입니다.

**선수 지식**: [Latent Diffusion](../12-diffusion-models/06-latent-diffusion.md), [CFG](../12-diffusion-models/05-cfg.md), [CLIP](../10-vision-language/02-clip.md)
**학습 목표**:
- Stable Diffusion의 전체 파이프라인을 이해한다
- VAE, U-Net, CLIP 각각의 역할과 상호작용을 파악한다
- 텍스트에서 이미지가 생성되는 전체 흐름을 추적할 수 있다
- HuggingFace Diffusers로 Stable Diffusion을 실행할 수 있다

## 왜 알아야 할까?

Stable Diffusion은 2022년 8월 오픈소스로 공개되면서 이미지 생성 AI의 대중화를 이끌었습니다. 아키텍처를 이해하면 [LoRA](../14-generative-practice/01-lora.md), [ControlNet](../14-generative-practice/03-controlnet.md) 등의 확장 기법이 "어디에, 어떻게" 작용하는지 명확히 알 수 있어요.

## 핵심 개념

### 개념 1: 전체 파이프라인 — 텍스트에서 이미지까지

> 💡 **비유**: Stable Diffusion은 **영화 제작 팀**과 같습니다. 각본가(CLIP 텍스트 인코더)가 대본을 쓰고, 감독(U-Net)이 씬을 구성하며, 촬영감독(VAE 디코더)이 최종 영상을 완성하죠. 이 세 파트가 분업하면서도 긴밀히 협력합니다.

텍스트 → 이미지 생성의 전체 흐름:

1. **텍스트 인코딩**: "a beautiful sunset over the ocean" → CLIP 텍스트 인코더 → 텍스트 임베딩 (77×768)
2. **잠재 노이즈 생성**: 랜덤 노이즈 (4×64×64) 생성
3. **반복 디노이징**: U-Net이 텍스트 조건을 받아 잠재 공간에서 노이즈 제거 (20~50 스텝)
4. **이미지 디코딩**: VAE 디코더가 잠재 표현 (4×64×64) → 이미지 (3×512×512) 변환

### 개념 2: 세 기둥의 상세 역할

**1. CLIP 텍스트 인코더**

[CLIP](../10-vision-language/02-clip.md)에서 배운 텍스트 인코더가 여기서 핵심 역할을 합니다:
- 텍스트를 최대 77개의 토큰으로 변환
- 각 토큰을 768차원(SD 1.5) 또는 1280차원(SDXL) 벡터로 인코딩
- 이 임베딩이 U-Net의 **크로스 어텐션**에 주입되어 생성 방향을 조종

**2. U-Net (노이즈 예측기)**

[이전에 배운](../12-diffusion-models/04-unet-architecture.md) Diffusion U-Net이 핵심 연산을 담당:
- 입력: 노이즈가 섞인 잠재 표현 (4×64×64) + 시간 $t$ + 텍스트 임베딩
- 출력: 예측된 노이즈 (4×64×64)
- 이 과정을 20~50번 반복하여 점진적으로 디노이징

**3. VAE (이미지 ↔ 잠재 공간 변환)**

[VAE](../11-generative-basics/02-vae.md)가 이미지와 잠재 공간 사이의 **번역기** 역할:
- 인코더: 이미지 (3×512×512) → 잠재 표현 (4×64×64) — img2img에 사용
- 디코더: 잠재 표현 (4×64×64) → 이미지 (3×512×512) — 최종 출력

| 모듈 | 파라미터 수 (SD 1.5) | 입력 | 출력 |
|------|---------------------|------|------|
| CLIP 텍스트 인코더 | 1.23억 | 텍스트 (77 토큰) | 임베딩 (77×768) |
| U-Net | 8.6억 | 잠재 노이즈 + 시간 + 텍스트 | 예측 노이즈 |
| VAE | 0.84억 | 잠재 표현 / 이미지 | 이미지 / 잠재 표현 |

### 개념 3: 크로스 어텐션 — 텍스트가 이미지를 조종하는 핵심

> 💡 **비유**: 크로스 어텐션은 **무전기**와 같습니다. 각본가(텍스트)가 무전기로 "지금 석양 장면이야, 바다를 넣어!"라고 지시하면, 감독(U-Net)이 그에 맞게 장면을 구성하죠.

U-Net의 크로스 어텐션에서:
- **Query**: 이미지 특징맵에서 생성 (64×64 위치별 "여기에 뭘 그릴까?")
- **Key/Value**: 텍스트 임베딩에서 생성 ("sunset", "ocean" 등의 의미)
- 어텐션 가중치로 각 이미지 위치가 어떤 텍스트 토큰에 주목할지 결정

이 메커니즘 덕분에 "red car on a blue road"라고 하면 빨간색이 자동차에, 파란색이 도로에 정확히 적용됩니다.

> ⚠️ **흔한 오해**: "Stable Diffusion이 텍스트를 직접 이해한다" — 아닙니다! SD 자체는 텍스트를 이해하지 못합니다. CLIP 텍스트 인코더가 텍스트를 숫자 벡터로 변환하고, U-Net은 이 벡터에 반응하는 겁니다. 텍스트 이해의 한계는 CLIP의 한계와 직결되죠.

### 개념 4: img2img — 이미지를 조건으로 사용

Stable Diffusion은 텍스트뿐 아니라 **이미지도 조건**으로 사용할 수 있습니다:

1. 입력 이미지를 VAE 인코더로 잠재 표현으로 변환
2. 잠재 표현에 **일부 노이즈를 추가** (strength 파라미터로 조절)
3. U-Net이 텍스트 조건과 함께 디노이징
4. VAE 디코더로 최종 이미지 출력

**strength** 파라미터 (0.0~1.0):
- 0.0: 원본 이미지 그대로 (변화 없음)
- 0.5: 원본의 구조 유지, 세부 변경
- 1.0: 원본 무시, 완전히 새로 생성

## 실습: HuggingFace Diffusers로 이미지 생성

```python
# Stable Diffusion 파이프라인 — Diffusers 라이브러리 사용
# pip install diffusers transformers accelerate torch

from diffusers import StableDiffusionPipeline
import torch

# 모델 로드 (SD 1.5 기준)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,            # 메모리 절약
)
pipe = pipe.to("cuda")                    # GPU 사용

# 텍스트 → 이미지 생성
image = pipe(
    prompt="a beautiful sunset over the ocean, photorealistic",
    negative_prompt="blurry, low quality",  # 네거티브 프롬프트
    num_inference_steps=30,                 # 디노이징 스텝 수
    guidance_scale=7.5,                     # CFG 스케일
    width=512,
    height=512,
).images[0]

image.save("sunset.png")
print("이미지 생성 완료!")
```

### 파이프라인 내부 동작 이해하기

```python
import torch

# 파이프라인의 각 단계를 수동으로 실행하여 내부 동작 이해
# (개념적 코드 — 실제 실행은 위의 pipe() 사용 권장)

# 1단계: 텍스트 인코딩 (CLIP)
prompt = "a beautiful sunset"
# text_embeddings = pipe.text_encoder(pipe.tokenizer(prompt))
# 결과: (1, 77, 768) 텐서

# 2단계: 초기 잠재 노이즈 생성
latent = torch.randn(1, 4, 64, 64)   # (배치, 채널, 높이, 너비)
print(f"잠재 노이즈 크기: {latent.shape}")  # [1, 4, 64, 64]

# 3단계: 반복 디노이징 (U-Net × N 스텝)
# for t in scheduler.timesteps:
#     noise_pred = unet(latent, t, text_embeddings)
#     latent = scheduler.step(noise_pred, t, latent)

# 4단계: VAE 디코딩
# image = vae.decode(latent / 0.18215)  # 스케일링 팩터 적용
# 결과: (1, 3, 512, 512) 텐서 → PIL Image

print(f"최종 이미지 크기: 3×512×512")
print(f"픽셀 수: {3 * 512 * 512:,} = 786,432")
print(f"잠재 공간 크기: {4 * 64 * 64:,} = 16,384 (약 48배 압축)")
```

## 더 깊이 알아보기

### Stable Diffusion의 탄생 — 오픈소스 혁명

2022년, CompVis 팀(하이델베르크 대학의 Robin Rombach, Patrick Esser 등)이 LDM 논문을 발표하고, Stability AI의 자금 지원으로 LAION-5B 데이터셋(50억 개 이미지-텍스트 쌍)으로 대규모 학습을 진행했습니다.

가장 혁명적이었던 것은 이 모델을 **완전히 오픈소스로 공개**한 것입니다. DALL-E 2나 Midjourney가 API로만 접근 가능했던 반면, Stable Diffusion은 누구나 자신의 GPU에서 실행하고 수정할 수 있었죠. 이 결정이 [LoRA](../14-generative-practice/01-lora.md), [ControlNet](../14-generative-practice/03-controlnet.md), [ComfyUI](../14-generative-practice/05-comfyui.md) 등 거대한 오픈소스 생태계를 탄생시켰습니다.

## 흔한 오해와 팁

> 💡 **알고 계셨나요?**: Stable Diffusion 1.5의 전체 모델 크기는 약 4GB입니다. float16으로 로드하면 4GB GPU에서도 실행할 수 있어요. 이전까지 이미지 생성 AI는 수십 GB의 GPU 메모리가 필요했던 것을 생각하면 놀라운 효율성이죠.

> 🔥 **실무 팁**: `torch.float16`(half precision)을 사용하면 메모리를 절반으로 줄이면서 생성 속도도 빨라집니다. 품질 차이는 거의 없어요. 메모리가 부족하면 `pipe.enable_attention_slicing()`으로 어텐션 연산을 분할할 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| CLIP 텍스트 인코더 | 텍스트를 77×768 임베딩으로 변환 |
| U-Net | 잠재 공간에서 노이즈를 예측하는 핵심 네트워크 (8.6억 파라미터) |
| VAE | 이미지 ↔ 잠재 공간 변환 (8× 압축) |
| 크로스 어텐션 | 텍스트 임베딩을 U-Net에 주입하는 메커니즘 |
| img2img | 입력 이미지에 노이즈를 추가한 후 텍스트 조건으로 디노이징 |

## 다음 섹션 미리보기

Stable Diffusion의 기본 아키텍처를 이해했으니, 다음 섹션 [SD 1.5 vs SDXL](./02-sd15-vs-sdxl.md)에서는 모델 버전별 차이를 상세히 비교합니다. 왜 SDXL이 더 좋은 결과를 내는지, 그 대가로 무엇이 필요한지 알아볼까요?

## 참고 자료

- [Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)](https://arxiv.org/abs/2112.10752) - LDM/Stable Diffusion 원논문
- [CompVis/stable-diffusion (GitHub)](https://github.com/CompVis/stable-diffusion) - 공식 리포지토리
- [Stable Diffusion with Diffusers (HuggingFace)](https://huggingface.co/blog/stable_diffusion) - HuggingFace 공식 가이드
- [How Stable Diffusion Works](https://stable-diffusion-art.com/how-stable-diffusion-work/) - 파이프라인 시각적 해설
