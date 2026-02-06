# LoRA 학습

> 효율적인 모델 미세조정

## 개요

[SD3와 미래 방향](../13-stable-diffusion/06-sd3-future.md)에서 이미지 생성 모델의 현재와 미래를 살펴봤습니다. 이제부터는 이 모델들을 **내 스타일로 커스터마이징**하는 실전 기법을 배웁니다. 첫 번째 주제는 **LoRA(Low-Rank Adaptation)** — 수십억 파라미터의 거대한 모델을 **몇 MB짜리 작은 파일 하나**로 내 취향대로 바꿀 수 있는 마법 같은 기술입니다.

**선수 지식**: [SD 아키텍처](../13-stable-diffusion/01-sd-architecture.md), [Latent Diffusion](../12-diffusion-models/06-latent-diffusion.md)
**학습 목표**:
- LoRA의 저랭크 분해(Low-Rank Decomposition) 원리를 이해한다
- Rank와 Alpha 파라미터의 역할을 안다
- LoRA 학습에 필요한 데이터셋 준비 방법을 익힌다
- Diffusers와 Kohya로 LoRA를 학습하고 적용할 수 있다

## 왜 알아야 할까?

[SD 아키텍처](../13-stable-diffusion/01-sd-architecture.md)에서 배운 Stable Diffusion 모델은 수십억 개의 파라미터를 가지고 있습니다. 이걸 전체 파인튜닝하려면 수십 GB의 VRAM과 며칠의 학습 시간이 필요하죠. 하지만 LoRA를 사용하면 **10~50장의 이미지**와 **일반 소비자 GPU**로도 나만의 스타일이나 캐릭터를 학습할 수 있습니다. 결과물은 1~100MB 정도의 작은 파일이라 공유도 쉽고요. Civitai 같은 커뮤니티에 수만 개의 LoRA가 공유되는 이유입니다.

## 핵심 개념

### 개념 1: LoRA의 핵심 아이디어 — 저랭크 분해

> 💡 **비유**: 전체 모델을 파인튜닝하는 것이 **거대한 오케스트라 전체를 재훈련**시키는 거라면, LoRA는 **몇 명의 솔리스트만 추가 연습**시키는 것과 같습니다. 오케스트라 전체(원본 모델)는 그대로 두고, 핵심 주자(저랭크 행렬)만 새로 학습시켜서 원하는 곡(스타일)을 연주하게 하는 거죠.

LoRA의 핵심은 **행렬 분해**입니다. 원래 가중치 행렬 $W$를 직접 수정하는 대신, 훨씬 작은 두 행렬 $A$와 $B$를 학습하여 변화량 $\Delta W$를 근사합니다:

$$W' = W + \Delta W = W + BA$$

여기서:
- $W$: 원본 모델의 가중치 행렬 (예: 1000×2000, 200만 개 파라미터)
- $B$: 새로 학습할 행렬 (1000×r)
- $A$: 새로 학습할 행렬 (r×2000)
- $r$: **랭크(Rank)** — 핵심 하이퍼파라미터 (보통 4~128)

**파라미터 절감 효과:**

| 원본 행렬 | 크기 | LoRA (r=8) | 절감률 |
|-----------|------|------------|--------|
| 1000×2000 | 200만 | (1000×8) + (8×2000) = 2.4만 | **98.8%** |
| 4096×4096 | 1,677만 | (4096×8) + (8×4096) = 6.5만 | **99.6%** |

> ⚠️ **흔한 오해**: "LoRA는 모델을 압축한다" — LoRA는 모델을 압축하는 게 아닙니다. **원본 모델은 그대로** 유지하고, 그 위에 작은 "변화량"을 더하는 겁니다. 그래서 여러 LoRA를 하나의 기본 모델에 **조합해서 사용**할 수 있어요.

### 개념 2: Rank와 Alpha — 두 가지 핵심 파라미터

LoRA 학습에서 가장 중요한 두 파라미터는 **Rank**와 **Alpha**입니다.

**Rank (Network Dimension)**

Rank는 LoRA의 "용량"을 결정합니다. 높을수록 더 복잡한 스타일을 학습할 수 있지만, 파일 크기도 커집니다.

| Rank | 파일 크기 (SD 1.5) | 적합한 용도 |
|------|-------------------|-------------|
| 4~8 | ~2MB | 단순한 색감, 톤 변화 |
| 16~32 | ~4-8MB | 일반적인 스타일 학습 |
| 64~128 | ~15-50MB | 복잡한 캐릭터, 세밀한 스타일 |
| 256+ | ~100MB+ | 매우 복잡한 컨셉 (잘 사용 안 함) |

**Alpha (Network Alpha)**

Alpha는 학습률과 함께 LoRA의 **강도**를 조절합니다. 일반적인 규칙:

- `alpha = rank`: LoRA가 원래 강도로 적용됨
- `alpha = rank/2`: LoRA 강도가 절반으로 줄어듦 (가장 흔한 설정)
- `alpha < rank`: 학습이 더 안정적, 과적합 방지

> 🔥 **실무 팁**: 처음 LoRA를 학습할 때는 **Rank=32, Alpha=16** (alpha = rank/2)으로 시작하세요. 결과를 보고 Rank를 높이거나 낮추면 됩니다.

### 개념 3: 어디에 LoRA를 적용할까?

[SD 아키텍처](../13-stable-diffusion/01-sd-architecture.md)에서 배운 것처럼, Stable Diffusion은 여러 모듈로 구성됩니다. LoRA는 보통 **U-Net의 어텐션 레이어**에 적용합니다.

**적용 가능한 위치:**

| 위치 | 효과 | 일반적 사용 |
|------|------|-------------|
| **Cross-Attention** (Q, K, V, Out) | 텍스트-이미지 연결 수정 | **항상 적용** |
| **Self-Attention** (Q, K, V, Out) | 이미지 내부 구조 수정 | 대부분 적용 |
| **FFN/MLP** | 세부 디테일 수정 | 선택적 |
| **Text Encoder** | 프롬프트 해석 방식 수정 | 캐릭터/컨셉 학습 시 |

**FLUX/SD3 시대의 LoRA:**

[FLUX](../13-stable-diffusion/05-flux.md)와 [SD3](../13-stable-diffusion/06-sd3-future.md)는 U-Net 대신 Transformer 블록을 사용합니다. LoRA의 원리는 동일하지만, 적용 위치가 **이중 스트림/단일 스트림 블록의 어텐션과 MLP**로 바뀝니다.

### 개념 4: 데이터셋 준비 — LoRA 학습의 80%

> 💡 **비유**: LoRA 학습은 **요리**와 비슷합니다. 아무리 좋은 조리법(학습 파라미터)이 있어도, 재료(데이터셋)가 나쁘면 맛없는 요리가 나와요. 반대로 좋은 재료만 있으면 간단한 조리법으로도 훌륭한 요리가 됩니다.

**필요한 이미지 수:**

| 학습 목표 | 최소 이미지 | 권장 이미지 |
|-----------|-------------|-------------|
| 간단한 스타일 | 10~20장 | 30~50장 |
| 캐릭터/인물 | 15~30장 | 50~100장 |
| 복잡한 컨셉 | 30~50장 | 100~200장 |

**좋은 데이터셋의 조건:**

1. **다양성**: 같은 포즈/각도만 있으면 그것만 학습됨
2. **일관성**: 학습하려는 특성은 모든 이미지에 일관되게 존재해야 함
3. **고품질**: 흐릿하거나 노이즈가 많은 이미지는 제외
4. **적절한 해상도**: 512×512 (SD 1.5) 또는 1024×1024 (SDXL/FLUX)

**캡션 작성 — 이미지만큼 중요!**

각 이미지에는 캡션(텍스트 설명)이 필요합니다. 캡션은 모델에게 "이 이미지에서 뭘 배워야 하는지" 알려주는 역할을 합니다.

```
# 좋은 캡션 예시 (캐릭터 LoRA)
sks_character, 1girl, red hair, green eyes, smiling, portrait, white background

# 나쁜 캡션 예시
pretty anime girl  # 너무 일반적, 특징이 없음
```

**트리거 워드(Trigger Word)**는 LoRA를 활성화하는 특별한 단어입니다. 위 예시의 `sks_character`처럼 모델이 모르는 고유한 단어를 사용합니다.

> 💡 **알고 계셨나요?** "sks"라는 트리거 워드는 DreamBooth 논문에서 처음 사용되었는데, 원래는 총기 브랜드 이름입니다. 연구자들이 "모델이 절대 모를 단어"를 찾다가 우연히 선택한 거예요. 지금은 관례처럼 많은 사람들이 sks를 사용합니다.

## 실습: LoRA 학습과 적용

### 방법 1: HuggingFace Diffusers로 LoRA 학습

```python
# diffusers와 peft를 이용한 LoRA 학습 (간략 버전)
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
import torch

# 기본 모델 로드
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

# LoRA 설정
lora_config = LoraConfig(
    r=32,                          # Rank
    lora_alpha=16,                 # Alpha (보통 rank/2)
    target_modules=[               # 어텐션 레이어에 적용
        "to_q", "to_k", "to_v", "to_out.0"
    ],
    lora_dropout=0.05,             # 과적합 방지
)

# U-Net에 LoRA 적용
unet = get_peft_model(pipe.unet, lora_config)

print(f"학습 가능 파라미터: {unet.num_parameters(only_trainable=True):,}")
print(f"전체 파라미터: {unet.num_parameters():,}")
# 출력 예: 학습 가능: 약 380만, 전체: 약 8.6억 (0.4%만 학습!)
```

### 방법 2: Kohya sd-scripts로 LoRA 학습 (실전)

Kohya의 sd-scripts는 커뮤니티에서 가장 많이 사용하는 LoRA 학습 도구입니다.

```bash
# Kohya 설치 (Linux/WSL)
git clone https://github.com/kohya-ss/sd-scripts
cd sd-scripts
pip install -r requirements.txt

# 데이터셋 폴더 구조
# train_data/
# ├── 10_sks_style/           # 반복횟수_트리거워드_클래스
# │   ├── image1.png
# │   ├── image1.txt          # 캡션 파일
# │   ├── image2.png
# │   └── image2.txt
```

```python
# Kohya LoRA 학습 명령어 (예시)
"""
accelerate launch --num_cpu_threads_per_process=2 train_network.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --train_data_dir="./train_data" \
    --output_dir="./output" \
    --output_name="my_style_lora" \
    --network_module=networks.lora \
    --network_dim=32 \
    --network_alpha=16 \
    --resolution=512 \
    --train_batch_size=1 \
    --max_train_epochs=10 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW8bit" \
    --mixed_precision="fp16" \
    --save_model_as=safetensors
"""
```

**주요 파라미터 설명:**

| 파라미터 | 권장값 | 설명 |
|---------|--------|------|
| `network_dim` | 32 | Rank 값 |
| `network_alpha` | 16 | Alpha 값 (dim/2 권장) |
| `learning_rate` | 1e-4 ~ 5e-5 | 학습률 |
| `max_train_epochs` | 5~20 | 에폭 수 |
| `resolution` | 512/1024 | SD1.5=512, SDXL=1024 |

### 방법 3: 학습된 LoRA 적용하기

```python
# 학습된 LoRA 로드 및 적용
from diffusers import StableDiffusionPipeline
import torch

# 기본 모델 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# LoRA 로드 (safetensors 또는 pt 파일)
pipe.load_lora_weights("./my_style_lora.safetensors")

# 이미지 생성 (트리거 워드 포함)
prompt = "sks_style, a beautiful landscape, sunset over mountains"
image = pipe(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("lora_result.png")
```

```python
# LoRA 가중치 조절 (강도 조절)
pipe.load_lora_weights(
    "./my_style_lora.safetensors",
    adapter_name="my_style"
)

# LoRA 강도를 0.7로 설정 (기본값 1.0)
pipe.set_adapters(["my_style"], adapter_weights=[0.7])

# 여러 LoRA 조합 사용
pipe.load_lora_weights("./style_lora.safetensors", adapter_name="style")
pipe.load_lora_weights("./character_lora.safetensors", adapter_name="character")

# 두 LoRA를 다른 강도로 조합
pipe.set_adapters(
    ["style", "character"],
    adapter_weights=[0.8, 0.5]
)

image = pipe("sks_style, sks_character, portrait").images[0]
```

## 더 깊이 알아보기

### LoRA의 수학적 배경

LoRA가 작동하는 이유는 **과잉 매개변수화(Overparameterization)** 가설에 있습니다. 거대한 사전 학습 모델의 가중치 업데이트 $\Delta W$는 실제로 **저차원 다양체(Low-Rank Manifold)**에 존재한다는 것이죠.

**행렬 랭크의 의미:**
- 1000×2000 행렬의 최대 랭크는 1000
- 하지만 실제로 의미 있는 정보는 훨씬 적은 차원에 존재
- 랭크 32~64면 대부분의 스타일 변화를 표현할 수 있음

이 가설은 2021년 Microsoft 연구진이 GPT-3 파인튜닝 연구에서 처음 제안했습니다. 당시 1750억 파라미터의 GPT-3를 풀 파인튜닝하는 것은 불가능에 가까웠고, LoRA는 이 문제를 우아하게 해결했습니다.

### LoRA 변형들

| 변형 | 특징 | 용도 |
|------|------|------|
| **LoRA** | 원본, A는 Gaussian 초기화, B는 0 초기화 | 일반적 학습 |
| **LoHa** (Hadamard) | 하다마드 곱을 사용, 더 표현력 높음 | 복잡한 스타일 |
| **LoKr** (Kronecker) | 크로네커 곱 사용, 매우 작은 파일 | 경량화 필요 시 |
| **LyCORIS** | LoHa, LoKr 등 다양한 기법 통합 라이브러리 | 고급 사용자 |
| **DoRA** | 방향과 크기를 분리하여 학습 | 2024년 최신 기법 |

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Rank가 높을수록 무조건 좋다" — Rank가 너무 높으면 **과적합**되어 학습 이미지만 복제하고 새로운 이미지를 생성하지 못합니다. 데이터셋 크기에 맞는 적절한 Rank를 선택하세요.

> 🔥 **실무 팁**: LoRA가 과적합되었는지 확인하려면, 학습에 사용하지 않은 프롬프트로 이미지를 생성해보세요. 결과가 학습 이미지와 너무 비슷하다면 Rank를 낮추거나 학습을 일찍 멈추세요.

> 💡 **알고 계셨나요?** LoRA는 원래 LLM(대규모 언어 모델)을 위해 개발되었습니다. 2021년 Microsoft 연구진이 GPT-3 파인튜닝을 위해 발표한 후, 2022년 @cloneofsimo가 Stable Diffusion에 최초로 적용했죠. 이후 커뮤니티의 폭발적인 반응과 함께 이미지 생성의 핵심 기술로 자리잡았습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **LoRA 원리** | 원본 가중치를 동결하고 저랭크 행렬 BA로 변화량을 학습 |
| **Rank** | LoRA의 용량 결정, 높을수록 복잡한 스타일 학습 가능 (32~64 권장) |
| **Alpha** | LoRA 강도 조절, 보통 Rank의 절반으로 설정 |
| **데이터셋** | 10~100장의 고품질 이미지 + 적절한 캡션 필요 |
| **트리거 워드** | LoRA를 활성화하는 고유 키워드 (예: sks_style) |
| **파라미터 절감** | 전체 파인튜닝 대비 99%+ 파라미터 절감 |

## 다음 섹션 미리보기

다음 [DreamBooth](./02-dreambooth.md)에서는 LoRA와 함께 자주 언급되는 또 다른 파인튜닝 기법을 배웁니다. DreamBooth는 "내 얼굴"이나 "내 반려동물"처럼 **특정 주체(Subject)**를 모델에 학습시키는 데 특화되어 있어요. LoRA와 DreamBooth를 결합한 "LoRA + DreamBooth" 기법도 함께 다룹니다.

## 참고 자료

- [LoRA: Low-Rank Adaptation of Large Language Models (arXiv)](https://arxiv.org/abs/2106.09685) - LoRA 원논문 (Microsoft, 2021)
- [Using LoRA for Efficient Stable Diffusion Fine-Tuning](https://huggingface.co/blog/lora) - HuggingFace 공식 튜토리얼
- [Kohya sd-scripts GitHub](https://github.com/kohya-ss/sd-scripts) - 커뮤니티 표준 LoRA 학습 도구
- [LoRA Training Parameters Wiki](https://github.com/bmaltais/kohya_ss/wiki/LoRA-training-parameters) - Kohya 파라미터 상세 설명
- [Detailed LoRA Training Guide](https://www.viewcomfy.com/blog/detailed-LoRA-training-guide-for-Stable-Diffusion) - 실전 LoRA 학습 가이드
- [Understanding LoRA Training: LR, Dim, Alpha](https://medium.com/@dreamsarereal/understanding-lora-training-part-1-learning-rate-schedulers-network-dimension-and-alpha-c88a8658beb7) - Rank와 Alpha 상세 설명
