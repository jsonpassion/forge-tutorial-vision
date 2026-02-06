# DreamBooth

> 개인화된 이미지 생성

## 개요

[LoRA 학습](./01-lora.md)에서 저랭크 분해를 통한 효율적인 파인튜닝을 배웠습니다. 이번에는 **DreamBooth** — 단 3~5장의 사진으로 **특정 인물, 반려동물, 물건**을 모델에 각인시키는 기술을 다룹니다. "내 얼굴로 우주복을 입고 있는 이미지"나 "우리 강아지가 왕관을 쓴 초상화"를 만들 수 있게 해주는 개인화의 끝판왕입니다.

**선수 지식**: [LoRA 학습](./01-lora.md), [SD 아키텍처](../13-stable-diffusion/01-sd-architecture.md)
**학습 목표**:
- DreamBooth의 핵심 개념(고유 식별자, Prior Preservation)을 이해한다
- DreamBooth와 LoRA의 차이점과 조합 방법을 안다
- 효과적인 DreamBooth 학습을 위한 데이터셋을 준비할 수 있다
- Diffusers로 DreamBooth 학습을 실행할 수 있다

## 왜 알아야 할까?

LoRA가 **스타일**을 학습하는 데 뛰어나다면, DreamBooth는 **특정 주체(Subject)**를 학습하는 데 특화되어 있습니다. "이 사람의 얼굴", "이 특정 캐릭터", "이 강아지"처럼 정체성이 중요한 경우에 DreamBooth가 더 적합하죠. 물론 현대의 실전에서는 DreamBooth + LoRA를 **결합**하여 두 장점을 모두 취하는 경우가 많습니다.

## 핵심 개념

### 개념 1: DreamBooth의 핵심 아이디어

> 💡 **비유**: DreamBooth는 **모델의 사전에 새 단어를 추가**하는 것과 같습니다. 모델이 "dog"라는 일반 명사는 알지만 "내 강아지 Max"는 모르잖아요? DreamBooth는 모델에게 "sks dog = Max라는 특별한 강아지"라고 가르치는 겁니다.

DreamBooth는 2022년 Google Research에서 발표한 논문 "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation"에서 소개되었습니다.

**핵심 메커니즘:**

1. **고유 식별자(Unique Identifier)**: 모델이 모르는 희귀한 단어(예: "sks")를 특정 주체와 연결
2. **클래스 명사(Class Noun)**: 주체가 속한 일반 카테고리(예: "dog", "person")
3. **전체 파인튜닝**: LoRA와 달리 모델 가중치를 직접 수정

**프롬프트 구조:**

```
[고유 식별자] [클래스 명사]
예: "sks dog", "sks person", "sks toy"
```

학습 후에는 이렇게 사용:
```
"sks dog wearing a party hat"
"sks person in a spacesuit on Mars"
"sks toy in a museum display case"
```

### 개념 2: Prior Preservation — 망각 방지의 핵심

> 💡 **비유**: Prior Preservation은 **외국어를 배우면서 모국어를 잊지 않도록** 하는 것과 같습니다. 새로운 언어(특정 주체)를 배우느라 기존에 알던 언어(일반적인 개, 사람)를 잊어버리면 안 되니까요.

DreamBooth의 가장 큰 위험은 **언어 드리프트(Language Drift)**입니다. "sks dog"을 학습하느라 **일반적인 "dog"**를 생성하는 능력을 잃어버리는 현상이죠.

**Prior Preservation Loss의 작동 방식:**

1. 학습 전에 원본 모델로 **클래스 이미지(Regularization Images)** 생성
   - 예: "a dog" 프롬프트로 200~500장의 일반 개 이미지 생성
2. 학습 중에 두 가지를 동시에 학습:
   - **주체 이미지**: "sks dog" → 내 강아지 Max
   - **클래스 이미지**: "dog" → 일반적인 개 (망각 방지)

**수식으로 표현하면:**

$$\mathcal{L} = \mathcal{L}_{subject} + \lambda \cdot \mathcal{L}_{prior}$$

여기서:
- $\mathcal{L}_{subject}$: 주체 이미지에 대한 손실 (내 강아지 학습)
- $\mathcal{L}_{prior}$: 클래스 이미지에 대한 손실 (일반 개 유지)
- $\lambda$: Prior Preservation 가중치 (보통 1.0)

> ⚠️ **흔한 오해**: "클래스 이미지 없이도 학습할 수 있다" — 가능하지만, 과적합 위험이 높습니다. 특히 학습 데이터가 적을수록 클래스 이미지가 더 중요해집니다.

### 개념 3: DreamBooth vs LoRA — 언제 무엇을 선택할까?

| 비교 항목 | DreamBooth | LoRA |
|-----------|------------|------|
| **주요 용도** | **특정 주체** (얼굴, 캐릭터) | **스타일** (화풍, 색감) |
| **학습 방식** | 모델 전체 파인튜닝 | 저랭크 행렬만 학습 |
| **결과물 크기** | 2~8GB (전체 모델) | 1~100MB |
| **VRAM 요구** | 17~24GB+ | 12~16GB |
| **학습 시간** | 20~60분 | 8~30분 |
| **얼굴 품질** | **매우 우수** (95% 정확도) | 양호 (가끔 불안정) |
| **스타일 품질** | 양호 | **우수** |
| **조합 가능성** | 조합 어려움 | 여러 LoRA 조합 가능 |

**현대의 최선책: DreamBooth + LoRA**

DreamBooth의 높은 주체 충실도와 LoRA의 효율성을 결합한 **DreamBooth LoRA**가 현재 가장 인기 있는 방식입니다:

- DreamBooth 방식으로 학습하되
- LoRA 형태로 가중치 변화만 저장
- 결과: 수십 MB의 파일로 높은 품질의 주체 생성

### 개념 4: 효과적인 데이터셋 준비

**필요한 이미지 수:**

| 주체 유형 | 최소 | 권장 | 클래스 이미지 |
|-----------|------|------|---------------|
| 인물 얼굴 | 3~5장 | 10~20장 | 200~500장 |
| 반려동물 | 5~10장 | 15~30장 | 200~500장 |
| 제품/물건 | 5~10장 | 20~40장 | 100~300장 |
| 스타일 캐릭터 | 10~20장 | 30~50장 | 200~500장 |

**좋은 주체 이미지의 조건:**

1. **다양한 각도**: 정면, 측면, 3/4 각도 등
2. **다양한 조명**: 자연광, 실내조명, 다양한 밝기
3. **다양한 배경**: 배경이 다양할수록 배경과 주체를 분리 학습
4. **선명한 품질**: 흐릿하거나 가려진 이미지 제외
5. **일관된 주체**: 동일한 대상만 포함 (다른 사람/동물 제외)

**인물 사진 특별 가이드:**

```
✅ 좋은 예:
- 정면 얼굴, 자연스러운 표정
- 약간 고개 돌린 각도
- 미소 짓는 표정
- 진지한 표정
- 야외에서 찍은 사진

❌ 피할 것:
- 선글라스, 모자 착용
- 과도한 필터/보정
- 흐릿하거나 저해상도
- 그룹 사진
- 비슷한 포즈만 반복
```

> 🔥 **실무 팁**: 인물 학습 시 **배경이 다양한 이미지**를 사용하세요. 배경이 모두 같으면 모델이 배경까지 주체의 일부로 학습해버립니다.

### 개념 5: 고유 식별자 선택 — 왜 "sks"인가?

> 💡 **알고 계셨나요?** "sks"라는 트리거 워드는 DreamBooth 논문에서 처음 사용되었습니다. 연구자들이 "모델이 절대 모를 희귀한 단어"를 찾다가 무기 브랜드 이름 중 하나를 선택한 거예요. CLIP 텍스트 인코더의 어휘에는 있지만, 이미지와의 연결이 약한 단어라 학습에 적합했습니다.

**좋은 식별자의 조건:**

1. **희귀함**: 모델이 기존에 알지 못하는 단어
2. **토큰 효율성**: CLIP에서 1~2 토큰으로 인코딩
3. **충돌 없음**: 기존 개념과 겹치지 않음

**추천 식별자 예시:**

| 좋은 식별자 | 나쁜 식별자 | 이유 |
|-------------|-------------|------|
| sks | cute | "cute"는 형용사로 자주 사용됨 |
| ohwx | dog | 학습하려는 클래스와 충돌 |
| xyz123 | realistic | 스타일 키워드와 충돌 |
| qwe | beautiful | 일반적인 수식어 |

## 실습: DreamBooth 학습과 적용

### 방법 1: Diffusers로 기본 DreamBooth 학습

```python
# DreamBooth 기본 학습 스크립트 (개념 설명용)
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.training_utils import set_seed
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class DreamBoothDataset(Dataset):
    """DreamBooth 학습용 데이터셋"""
    def __init__(
        self,
        instance_data_dir,      # 주체 이미지 폴더
        instance_prompt,        # "sks dog"
        class_data_dir=None,    # 클래스 이미지 폴더 (Prior Preservation)
        class_prompt=None,      # "dog"
        size=512
    ):
        self.instance_images = self.load_images(instance_data_dir)
        self.instance_prompt = instance_prompt

        # Prior Preservation용 클래스 이미지
        self.class_images = []
        if class_data_dir:
            self.class_images = self.load_images(class_data_dir)
        self.class_prompt = class_prompt
        self.size = size

    def load_images(self, folder):
        images = []
        for f in os.listdir(folder):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(folder, f))
                images.append(img.convert('RGB'))
        return images

    def __len__(self):
        return max(len(self.instance_images), len(self.class_images))

    def __getitem__(self, idx):
        # 주체 이미지와 프롬프트
        instance_img = self.instance_images[idx % len(self.instance_images)]

        result = {
            "instance_image": instance_img,
            "instance_prompt": self.instance_prompt
        }

        # Prior Preservation 활성화 시 클래스 이미지도 반환
        if self.class_images:
            class_img = self.class_images[idx % len(self.class_images)]
            result["class_image"] = class_img
            result["class_prompt"] = self.class_prompt

        return result

# 데이터셋 생성 예시
dataset = DreamBoothDataset(
    instance_data_dir="./my_dog_photos",    # 내 강아지 사진 폴더
    instance_prompt="sks dog",              # 고유 식별자 + 클래스
    class_data_dir="./dog_regularization",  # 일반 개 이미지 폴더
    class_prompt="dog"                      # 클래스 프롬프트
)

print(f"주체 이미지: {len(dataset.instance_images)}장")
print(f"클래스 이미지: {len(dataset.class_images)}장")
```

### 방법 2: Accelerate로 실제 학습 실행

```bash
# Diffusers의 DreamBooth 학습 스크립트 사용
# 먼저 클래스 이미지 생성
accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="./my_dog_photos" \
  --class_data_dir="./dog_regularization" \
  --output_dir="./dreambooth_dog" \
  --instance_prompt="sks dog" \
  --class_prompt="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --num_class_images=200 \
  --mixed_precision="fp16"
```

### 방법 3: DreamBooth + LoRA (권장)

```bash
# DreamBooth LoRA 학습 (더 효율적!)
accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="./my_dog_photos" \
  --class_data_dir="./dog_regularization" \
  --output_dir="./dreambooth_lora_dog" \
  --instance_prompt="sks dog" \
  --class_prompt="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=500 \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --num_class_images=200 \
  --rank=32 \
  --mixed_precision="fp16"
```

### 방법 4: 학습된 모델 사용

```python
# DreamBooth로 학습한 모델 로드 및 사용
from diffusers import StableDiffusionPipeline
import torch

# 전체 DreamBooth 모델 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "./dreambooth_dog",  # 학습된 모델 폴더
    torch_dtype=torch.float16
)
pipe.to("cuda")

# 다양한 상황에서 내 강아지 생성
prompts = [
    "sks dog wearing a crown, royal portrait",
    "sks dog as an astronaut on the moon",
    "sks dog in a flower field, sunset",
    "sks dog wearing a detective hat, noir style"
]

for i, prompt in enumerate(prompts):
    image = pipe(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]
    image.save(f"dreambooth_result_{i}.png")
    print(f"생성 완료: {prompt}")
```

```python
# DreamBooth LoRA 로드 및 사용
from diffusers import StableDiffusionPipeline
import torch

# 기본 모델 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# DreamBooth LoRA 로드
pipe.load_lora_weights("./dreambooth_lora_dog")

# 이미지 생성
image = pipe(
    "sks dog wearing a superhero cape, flying in the sky",
    num_inference_steps=30,
).images[0]

image.save("dreambooth_lora_result.png")
```

## 더 깊이 알아보기

### DreamBooth의 탄생 스토리

DreamBooth는 2022년 Google Research에서 발표되었습니다. 논문 제목의 "DreamBooth"는 사진관(Photo Booth)에서 영감을 받았어요. 마치 사진관에서 다양한 배경으로 사진을 찍듯이, 특정 주체를 다양한 상황에 배치할 수 있다는 의미입니다.

재미있는 점은, DreamBooth 연구진이 처음에는 **Imagen** 모델로 연구를 진행했다는 것입니다. 하지만 Imagen은 공개되지 않아서, 커뮤니티는 이 기술을 **Stable Diffusion**에 적용했죠. 오픈소스의 힘입니다!

### Textual Inversion과의 비교

DreamBooth 이전에는 **Textual Inversion**이라는 기법이 있었습니다:

| 비교 항목 | Textual Inversion | DreamBooth |
|-----------|-------------------|------------|
| **수정 대상** | 텍스트 임베딩만 | 전체 모델 가중치 |
| **결과물 크기** | ~10KB | ~2GB |
| **학습 시간** | 수 시간 | 20~60분 |
| **주체 충실도** | 중간 | **매우 높음** |
| **과적합 위험** | 낮음 | 높음 (Prior Preservation 필요) |

Textual Inversion은 텍스트 임베딩만 학습하므로 표현력에 한계가 있지만, DreamBooth는 모델 자체를 수정하여 훨씬 높은 충실도를 달성합니다.

### 클래스 불일치 문제

주체에 맞는 올바른 클래스를 선택하는 것이 중요합니다:

```
✅ 올바른 클래스:
- 골든 리트리버 → "dog" (아종보다 일반 클래스가 나음)
- 특정 인물 → "person" 또는 "man"/"woman"
- 장난감 로봇 → "toy" 또는 "robot"

❌ 잘못된 클래스:
- 골든 리트리버 → "golden retriever" (너무 구체적)
- 특정 인물 → "celebrity" (잘못된 연관)
- 장난감 로봇 → "transformer" (다른 의미와 충돌)
```

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "DreamBooth는 LoRA보다 무조건 좋다" — 용도가 다릅니다. **특정 인물/동물**은 DreamBooth가 우세하지만, **스타일**이나 **개념** 학습은 LoRA가 더 효율적입니다. 최선은 둘을 결합한 DreamBooth LoRA입니다.

> 🔥 **실무 팁**: 인물 DreamBooth 학습 시 **배경을 미리 제거**하면 더 좋은 결과를 얻을 수 있습니다. rembg 라이브러리로 쉽게 배경을 제거할 수 있어요.

> 💡 **알고 계셨나요?** DreamBooth 논문의 저자 중 한 명인 Nataniel Ruiz는 나중에 LoRA와 DreamBooth를 결합한 연구도 진행했습니다. 이 조합이 현재 커뮤니티의 표준이 된 셈이죠.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **고유 식별자** | 특정 주체를 나타내는 희귀 단어 (예: "sks") |
| **클래스 명사** | 주체가 속한 일반 카테고리 (예: "dog", "person") |
| **Prior Preservation** | 클래스 이미지로 일반 개념 망각 방지 |
| **클래스 이미지** | 망각 방지용 정규화 이미지 (200~500장 권장) |
| **DreamBooth LoRA** | 두 기법의 장점을 결합한 현대적 접근 |
| **용도 차이** | 주체=DreamBooth, 스타일=LoRA |

## 다음 섹션 미리보기

다음 [ControlNet](./03-controlnet.md)에서는 생성 이미지를 **정밀하게 제어**하는 기술을 배웁니다. LoRA와 DreamBooth가 "무엇을" 생성할지 정하는 기술이라면, ControlNet은 "어떻게" 생성할지(포즈, 구도, 깊이)를 제어합니다. 세 기술을 조합하면 상상할 수 있는 거의 모든 이미지를 만들 수 있어요.

## 참고 자료

- [DreamBooth: Fine Tuning Text-to-Image Diffusion Models (arXiv)](https://arxiv.org/abs/2208.12242) - DreamBooth 원논문 (Google, 2022)
- [DreamBooth Project Page](https://dreambooth.github.io/) - 공식 프로젝트 페이지
- [DreamBooth Training with Diffusers](https://huggingface.co/docs/diffusers/training/dreambooth) - HuggingFace 공식 튜토리얼
- [Fine-tuning SDXL with DreamBooth and LoRA](https://www.datacamp.com/tutorial/fine-tuning-stable-diffusion-xl-with-dreambooth-and-lora) - SDXL DreamBooth LoRA 가이드
- [DreamBooth vs LoRA Comparison](https://www.pykaso.ai/resources/blog/dreambooth-vs-lora-comparison) - 두 기법 비교 분석
- [Dreambooth Using Diffusers - LearnOpenCV](https://learnopencv.com/dreambooth-using-diffusers/) - 실전 DreamBooth 튜토리얼
