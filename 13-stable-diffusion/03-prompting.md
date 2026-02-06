# 프롬프트 엔지니어링

> 효과적인 프롬프트 작성법

## 개요

Stable Diffusion의 이미지 품질은 프롬프트에 크게 좌우됩니다. 같은 모델이라도 프롬프트를 어떻게 쓰느냐에 따라 아마추어 수준과 프로 수준의 결과가 나뉘죠. 이번 섹션에서는 프롬프트의 구조, 가중치 조절, 네거티브 프롬프트까지 체계적으로 다룹니다.

**선수 지식**: [SD 아키텍처](./01-sd-architecture.md), [CFG](../12-diffusion-models/05-cfg.md), [CLIP](../10-vision-language/02-clip.md)
**학습 목표**:
- 효과적인 프롬프트의 구조와 패턴을 이해한다
- 토큰 가중치와 우선순위 조절법을 배운다
- 네거티브 프롬프트의 활용법을 파악한다
- 모델별 프롬프트 전략의 차이를 안다

## 왜 알아야 할까?

프롬프트는 [CLIP 텍스트 인코더](../10-vision-language/02-clip.md)를 통해 벡터로 변환되고, 이 벡터가 [크로스 어텐션](../12-diffusion-models/04-unet-architecture.md)을 통해 U-Net에 주입됩니다. 프롬프트 엔지니어링은 결국 "CLIP이 이해하기 좋은 형태로 의도를 전달하는 기술"이에요.

## 핵심 개념

### 개념 1: 프롬프트의 기본 구조

> 💡 **비유**: 좋은 프롬프트는 **레스토랑 주문**과 같습니다. "맛있는 거 주세요"보다 "미디엄 레어 스테이크, 머쉬룸 소스, 사이드로 구운 감자 부탁드려요"가 원하는 결과를 얻을 확률이 높죠.

효과적인 프롬프트의 구성 요소:

**1. 주제 (Subject)** — 무엇을 그릴지
> "a golden retriever puppy", "futuristic cityscape", "portrait of an elderly woman"

**2. 스타일 (Style)** — 어떤 화풍/분위기로
> "oil painting", "anime style", "photorealistic", "watercolor", "cyberpunk"

**3. 구도와 카메라 (Composition)** — 어떤 각도에서
> "close-up", "wide angle", "bird's eye view", "rule of thirds"

**4. 조명 (Lighting)** — 어떤 빛으로
> "golden hour lighting", "dramatic shadows", "studio lighting", "neon glow"

**5. 품질 키워드 (Quality)** — 품질 부스터
> "masterpiece", "highly detailed", "8k", "sharp focus", "professional photography"

### 개념 2: 토큰 제한과 우선순위

CLIP 텍스트 인코더는 **최대 77개 토큰**만 처리합니다. 이 제한은 중요한 함의를 가져요:

- 프롬프트 앞부분의 토큰이 **더 큰 영향력**을 가짐
- 77 토큰을 넘으면 뒷부분이 **잘려나감**
- 가장 중요한 설명을 앞에 배치해야 함

**가중치 조절 문법** (WebUI/ComfyUI 기준):
- `(keyword:1.3)` — 가중치 1.3배 강조
- `(keyword:0.7)` — 가중치 0.7배로 약화
- `((keyword))` — 약 1.21배 강조 (괄호 중첩)

```python
# 토큰 수 확인 예시
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
prompt = "a beautiful sunset over the ocean with dramatic clouds"
tokens = tokenizer(prompt)
print(f"토큰 수: {len(tokens['input_ids'])}")  # 시작/끝 토큰 포함
print(f"최대 허용: 77개")
```

### 개념 3: 네거티브 프롬프트 — 원치 않는 것 제거

[CFG](../12-diffusion-models/05-cfg.md)에서 배운 것처럼, 네거티브 프롬프트는 생성 방향을 **반대로** 유도합니다:

**범용 네거티브 프롬프트** (대부분의 상황에서 유용):
> "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, logo"

**인물 사진용 추가**:
> "extra fingers, mutated hands, bad proportions, extra limbs, cross-eyed"

**풍경/건축용 추가**:
> "oversaturated, unrealistic colors, bad perspective, cropped"

> 🔥 **실무 팁**: 네거티브 프롬프트에 너무 많은 키워드를 넣으면 오히려 역효과가 날 수 있어요. 핵심적인 5~10개 키워드에 집중하는 것이 좋습니다.

### 개념 4: 모델별 프롬프트 전략

| 모델 | 프롬프트 스타일 | 팁 |
|------|---------------|-----|
| **SD 1.5** | 품질 키워드 중요 ("masterpiece, best quality") | 구체적 설명이 효과적 |
| **SDXL** | 자연어에 강함, 품질 키워드 덜 필요 | 더 긴 서술적 프롬프트 가능 |
| **FLUX** | 자연어를 가장 잘 이해 | 일상 문장처럼 작성해도 됨 |

SD 1.5에서는 "masterpiece, best quality, 1girl, long hair, blue eyes, school uniform, cherry blossom" 같은 **태그 나열** 스타일이 효과적이지만, FLUX에서는 "A young woman with long flowing hair and blue eyes wearing a school uniform, standing under cherry blossom trees" 같은 **자연어 서술**이 더 잘 작동합니다.

> ⚠️ **흔한 오해**: "프롬프트를 길게 쓸수록 좋다" — 77 토큰 제한 안에서 핵심을 명확히 전달하는 것이 중요합니다. 불필요한 수식어를 나열하면 오히려 각 키워드의 영향력이 분산됩니다.

### 개념 5: 프롬프트 디버깅 — 어텐션 맵 분석

프롬프트의 각 단어가 이미지의 **어느 영역**에 영향을 미치는지 시각화할 수 있습니다. 이것은 [크로스 어텐션](../12-diffusion-models/04-unet-architecture.md)의 가중치를 추출하는 것이죠.

"cat" 토큰의 어텐션이 이미지의 고양이 영역에 집중되어 있다면, 프롬프트가 잘 작동하는 것입니다. 반대로 어텐션이 분산되어 있다면, 프롬프트를 더 구체적으로 수정해야 해요.

## 더 깊이 알아보기

### CLIP의 학습 데이터가 프롬프트에 미치는 영향

CLIP은 인터넷에서 수집된 이미지-텍스트 쌍으로 학습되었습니다. 따라서 인터넷에서 자주 등장하는 형태의 설명("stock photo of...", "digital art by...", "photography by...")이 더 효과적인 경우가 있어요.

예를 들어 "a photo of a cat"보다 "professional photography of a cat, Canon EOS R5, f/1.8, bokeh background"가 더 사실적인 결과를 내는데, 이는 CLIP이 카메라 관련 키워드를 고품질 사진과 연관시켜 학습했기 때문이죠.

## 흔한 오해와 팁

> 💡 **알고 계셨나요?**: "프롬프트 엔지니어링"이라는 직업이 실제로 존재합니다. AI 이미지 생성의 대중화와 함께, 프롬프트를 전문적으로 작성하고 판매하는 마켓플레이스(PromptBase 등)까지 등장했죠. 기술적 이해와 예술적 감각이 모두 필요한 새로운 영역입니다.

> 🔥 **실무 팁**: 좋은 프롬프트를 빨리 찾는 방법 — Civitai나 PromptHero 같은 커뮤니티에서 원하는 스타일과 비슷한 이미지를 찾고, 그 프롬프트를 **기반으로 수정**하는 것이 처음부터 작성하는 것보다 효율적입니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 프롬프트 구조 | 주제 + 스타일 + 구도 + 조명 + 품질 |
| 토큰 제한 | CLIP 최대 77 토큰, 앞부분이 더 중요 |
| 가중치 조절 | `(keyword:1.3)`으로 특정 요소 강조/약화 |
| 네거티브 프롬프트 | 원치 않는 요소를 명시하여 제거 |
| 모델별 차이 | SD 1.5는 태그 나열, FLUX는 자연어 서술 |

## 다음 섹션 미리보기

프롬프트만큼 중요한 것이 **샘플러 선택**입니다. 다음 섹션 [샘플러 가이드](./04-samplers.md)에서는 Euler, DPM++, UniPC 등 다양한 샘플러의 차이와 최적의 설정을 알아봅니다.

## 참고 자료

- [Stable Diffusion Prompt Guide (stable-diffusion-art.com)](https://stable-diffusion-art.com/prompt-guide/) - 프롬프트 작성 종합 가이드
- [CLIP Interrogator (GitHub)](https://github.com/pharmapsychotic/clip-interrogator) - 이미지에서 프롬프트를 역추적하는 도구
- [PromptHero](https://prompthero.com/) - 프롬프트 공유 커뮤니티
- [Civitai](https://civitai.com/) - 모델/프롬프트 커뮤니티
