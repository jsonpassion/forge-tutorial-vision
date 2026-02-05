# DETR과 Transformer 기반 탐지

> End-to-End 객체 탐지

## 개요

2020년, Facebook AI Research(현 Meta AI)는 객체 탐지에 **Transformer**를 도입한 **DETR(DEtection TRansformer)**을 발표합니다. DETR은 지금까지 "당연하다"고 여겨졌던 것들 — 앵커 박스, NMS, 후보 영역 추출 — 을 **모두 제거**하고, 순수한 Transformer의 집합 예측(Set Prediction)으로 객체 탐지를 풀어냈습니다. 구조가 놀라울 정도로 단순하면서도, Faster R-CNN에 필적하는 성능을 보여준 혁신적 모델입니다.

**선수 지식**: [객체 탐지 기초](./01-detection-basics.md), [Anchor-Free 탐지기](./04-anchor-free.md)
**학습 목표**:
- DETR의 핵심 아이디어(집합 예측 + 이분 매칭)를 이해한다
- Object Query와 헝가리안 매칭의 역할을 설명할 수 있다
- DETR의 한계와 이를 개선한 후속 모델들을 안다
- RT-DETR로 실시간 Transformer 기반 탐지를 경험할 수 있다

## 왜 알아야 할까?

지금까지 본 모든 탐지기에는 공통적인 **후처리 단계**가 있었습니다 — NMS로 중복 박스를 제거하는 것이죠. NMS는 단순하지만, IoU 임계값이라는 하이퍼파라미터에 민감하고, 밀집된 객체에서 문제가 생기며, End-to-End 학습을 방해합니다.

DETR은 이 문제를 **근본적으로** 해결했습니다. "각 객체에 정확히 하나의 예측만 매칭시키는" 방식으로 중복 자체가 발생하지 않도록 설계한 것이죠. 이 아이디어는 이후 RT-DETR이 **YOLO를 속도와 정확도 모두에서 이기는** 결과로 이어졌습니다.

## 핵심 개념

### 1. DETR의 핵심 아이디어 — 탐지를 "집합 예측"으로 바꾸기

> 💡 **비유**: 결혼식 자리 배치를 생각해보세요. 하객 100명에 대해 테이블 100개를 준비했습니다. **각 하객을 정확히 하나의 테이블에 앉히되**, 전체적으로 가장 합리적인 배치를 찾아야 합니다. DETR은 이미지 속 객체(하객)를 예측 슬롯(테이블)에 **1:1로 매칭**시키는 것과 같습니다. 한 객체에 두 예측이 붙는 일(NMS가 필요한 상황)이 애초에 없는 거죠!

**DETR의 3가지 구성 요소:**

1. **CNN 백본**: 이미지에서 특징 맵 추출 (ResNet-50 등)
2. **Transformer 인코더-디코더**: 전역 문맥을 반영한 특징 추출 및 객체 예측
3. **FFN(Feed-Forward Network)**: 각 예측에 대해 클래스 + 박스 좌표 출력

**핵심 메커니즘 — Object Query:**

DETR의 디코더에는 **N개의 학습 가능한 Object Query**가 입력됩니다 (보통 N=100). 각 쿼리는 "나는 이미지에서 특정 객체를 담당하는 슬롯이다"라는 역할을 합니다. Transformer의 어텐션 메커니즘을 통해, 각 쿼리가 이미지의 어느 부분을 볼지 스스로 학습합니다.

> ⚠️ **흔한 오해**: "Object Query 100개면 100개만 탐지 가능하다"고 생각할 수 있는데, 맞습니다! N개의 쿼리가 있으면 최대 N개 객체를 탐지할 수 있습니다. 하지만 일반적인 이미지에서 객체가 100개를 넘는 경우는 드물기 때문에, 대부분의 쿼리는 "no-object" (객체 없음)를 예측하게 됩니다.

### 2. 헝가리안 매칭 — 1:1 매칭의 마법

기존 탐지기에서는 하나의 정답 박스에 여러 예측이 매칭될 수 있었고, 이 중복을 NMS로 제거했습니다. DETR은 아예 **1:1 매칭**을 강제합니다.

**헝가리안 알고리즘(Hungarian Algorithm)**은 N개의 예측과 M개의 정답 사이에서 **전체 비용을 최소화하는 최적의 1:1 매칭**을 찾습니다.

매칭 비용은 다음 세 가지의 조합입니다:
- **분류 비용**: 예측 클래스와 정답 클래스의 불일치 정도
- **L1 거리**: 예측 박스와 정답 박스의 좌표 차이
- **GIoU 손실**: 박스 겹침 정도

$$\mathcal{L}_{match} = -\lambda_{cls}\hat{p}_{\sigma(i)}(c_i) + \lambda_{L1}\|b_i - \hat{b}_{\sigma(i)}\|_1 + \lambda_{giou}\mathcal{L}_{GIoU}(b_i, \hat{b}_{\sigma(i)})$$

- $\hat{p}_{\sigma(i)}(c_i)$: 매칭된 예측의 클래스 확률
- $b_i, \hat{b}_{\sigma(i)}$: 정답과 예측의 바운딩 박스
- $\sigma$: 최적의 매칭 순열 (헝가리안 알고리즘이 찾음)

간단히 말하면: "100개 예측 슬롯 중 어떤 것이 어떤 정답 객체를 담당할지, **전역적으로 가장 좋은 배정**을 찾는 것"입니다.

### 3. DETR 전체 파이프라인

**Step 1 — 특징 추출:**

CNN 백본(ResNet-50)으로 이미지에서 특징 맵을 추출하고, 2D 위치를 인코딩한 **Positional Encoding**을 더합니다.

**Step 2 — Transformer 인코더:**

특징 맵을 1D 시퀀스로 펼치고, Transformer 인코더에 통과시킵니다. 인코더의 Self-Attention은 **이미지 전체의 문맥**을 파악합니다 — "왼쪽에 사람이 있으면 그 옆의 물체는 자전거일 가능성이 높다" 같은 전역적 추론이 가능해지죠.

**Step 3 — Transformer 디코더:**

N개의 Object Query가 디코더에 입력되고, Cross-Attention으로 인코더 출력(이미지 특징)을 참조합니다. 각 쿼리는 "내가 담당할 객체"를 이미지에서 찾아갑니다.

**Step 4 — 예측 헤드:**

각 쿼리의 출력이 FFN을 통과해 **(클래스, 바운딩 박스)** 또는 **"no-object"**를 예측합니다.

### 4. DETR의 한계와 후속 모델들

DETR은 혁신적이었지만, 몇 가지 실용적 한계가 있었습니다:

| 한계 | 원인 |
|------|------|
| **느린 수렴** | 500 에포크 필요 (Faster R-CNN은 12~36 에포크) |
| **작은 객체 약함** | 단일 스케일 특징 사용 |
| **높은 계산 비용** | 전체 특징 맵에 대한 글로벌 어텐션 |

이를 해결하기 위해 후속 모델들이 빠르게 등장했습니다:

| 모델 | 연도 | 핵심 개선 | COCO AP |
|------|------|-----------|---------|
| **Deformable DETR** | 2021 | 변형 가능한 어텐션으로 수렴 10배 가속 | 46.2% |
| **DN-DETR** | 2022 | 디노이징 학습으로 수렴 가속 | 48.6% |
| **DINO** | 2022 | 개선된 쿼리 초기화 + 디노이징 | 49.4% |
| **RT-DETR** | 2024 | 효율적 하이브리드 인코더로 **실시간** 달성 | 53.0% |
| **RT-DETRv2** | 2024 | Bag-of-Freebies로 추가 성능 향상 | 54.3% |

> 💡 **알고 계셨나요?**: RT-DETR 논문의 제목은 **"DETRs Beat YOLOs on Real-time Object Detection"** — "DETR이 실시간 탐지에서 YOLO를 이겼다"입니다! Transformer 기반 탐지기가 "느리다"는 편견을 완전히 깨부순 결과였습니다. RT-DETR-L은 **COCO AP 53.0%에 114 FPS**를 달성하며, 같은 속도 대비 YOLOv8보다 높은 정확도를 보여줬습니다.

### 5. DETR vs 기존 탐지기 — 패러다임 비교

| 특성 | Faster R-CNN | YOLO | DETR |
|------|-------------|------|------|
| 후보 영역 | RPN | 그리드 | Object Query |
| 앵커 박스 | 필요 | v7까지 필요 | **불필요** |
| NMS | 필요 | 필요 | **불필요** |
| End-to-End | 부분적 | 부분적 | **완전** |
| 전역 문맥 | 제한적 | 제한적 | **Transformer로 완전** |
| 작은 객체 | 보통 | 보통 | 약함 → RT-DETR에서 개선 |

## 실습: RT-DETR로 실시간 탐지 경험하기

Ultralytics를 통해 RT-DETR을 쉽게 사용할 수 있습니다:

```python
from ultralytics import RTDETR

# RT-DETR 모델 로드 (자동 다운로드)
model = RTDETR("rtdetr-l.pt")  # Large 모델

# 이미지에서 객체 탐지
results = model("bus.jpg")

# 결과 확인
for result in results:
    boxes = result.boxes
    print(f"탐지된 객체 수: {len(boxes)}")

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        name = model.names[cls]
        print(f"  {name}: {conf:.2f} @ [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
```

HuggingFace Transformers로 원본 DETR을 사용하는 코드도 살펴봅시다:

```python
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image

# DETR 모델과 프로세서 로드
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()

# 이미지 로드 및 전처리
# image = Image.open("test.jpg")  # 실제 이미지 사용 시
# 여기서는 랜덤 이미지로 시연
import numpy as np
image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

inputs = processor(images=image, return_tensors="pt")

# 추론 — NMS 없이 바로 결과!
with torch.no_grad():
    outputs = model(**inputs)

# 후처리: 신뢰도 임계값 적용
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.7
)

# 결과 출력
result = results[0]
for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
    box = box.tolist()
    class_name = model.config.id2label[label.item()]
    print(f"  {class_name}: {score:.3f} @ [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
```

YOLO와 RT-DETR을 비교하는 것도 간단합니다:

```python
from ultralytics import YOLO, RTDETR

# 두 모델 로드
yolo = YOLO("yolo11s.pt")       # YOLO11 Small
rtdetr = RTDETR("rtdetr-l.pt")   # RT-DETR Large

# 같은 데이터셋에서 검증
yolo_metrics = yolo.val(data="coco128.yaml")
rtdetr_metrics = rtdetr.val(data="coco128.yaml")

print(f"YOLO11s  - mAP@0.5: {yolo_metrics.box.map50:.3f}, "
      f"mAP@0.5:0.95: {yolo_metrics.box.map:.3f}")
print(f"RT-DETR-L - mAP@0.5: {rtdetr_metrics.box.map50:.3f}, "
      f"mAP@0.5:0.95: {rtdetr_metrics.box.map:.3f}")
```

> 🔥 **실무 팁**: RT-DETR의 독특한 장점은 **디코더 레이어 수를 조절해 속도-정확도 trade-off를 제어**할 수 있다는 것입니다. 재학습 없이도, 추론 시 디코더 레이어를 줄이면 속도가 빨라지고, 늘리면 정확도가 올라갑니다.

## 더 깊이 알아보기

### DETR의 탄생 — Transformer가 CV에 온 순간

DETR(2020)은 NLP에서 혁명을 일으킨 Transformer 아키텍처를 컴퓨터 비전의 **객체 탐지**에 처음 성공적으로 적용한 모델입니다. 논문의 제1저자 Nicolas Carion은 Facebook AI Research(FAIR)에서 이 연구를 수행했으며, ECCV 2020에서 발표되었습니다.

흥미로운 점은 DETR 논문이 ViT(Vision Transformer, 2020년 10월)보다 **5개월 먼저** 나왔다는 것입니다. 즉, "Transformer를 비전에 쓸 수 있다"는 것을 DETR이 먼저 증명한 셈이죠. DETR은 분류가 아닌 탐지에 Transformer를 적용했지만, 이 성공이 ViT와 후속 연구들에 큰 영감을 주었습니다.

DETR 논문의 가장 인상적인 부분은 **코드의 단순함**입니다. 논문에 포함된 핵심 구현 코드가 **50줄도 안 됩니다**. Faster R-CNN의 복잡한 파이프라인(RPN, 앵커, NMS 등)과 비교하면 놀라울 정도로 깔끔하죠. 이 단순함이 DETR의 가장 큰 매력 중 하나입니다.

### Deformable Attention — DETR의 치명적 약점 해결

원본 DETR의 가장 큰 문제는 **수렴이 느리다**는 것이었습니다. 전체 특징 맵에 대한 글로벌 어텐션은 강력하지만 비효율적이거든요.

**Deformable DETR**은 각 쿼리가 **전체 특징 맵이 아닌, 소수의 핵심 포인트만** 참조하도록 변경했습니다. 이를 **Deformable Attention**이라 하는데, 참조 포인트의 위치를 학습 가능하게 만들어, 정말 중요한 곳만 보도록 합니다.

결과:
- 수렴 속도 **10배 향상** (500 에포크 → 50 에포크)
- 멀티스케일 특징 맵 활용으로 **작은 객체 성능 대폭 개선**
- 계산 비용 절감

이 기법은 이후 DINO, RT-DETR 등 모든 DETR 계열 모델의 핵심 기반이 되었습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "DETR은 느려서 실용적이지 않다"는 초기 DETR에 한한 이야기입니다. 2024년의 **RT-DETR은 114 FPS(T4 GPU)**로 YOLOv8보다 빠르면서도 더 정확합니다. Transformer 기반 탐지 = 느림이라는 공식은 더 이상 성립하지 않습니다.

> 💡 **알고 계셨나요?**: DETR의 GitHub 저장소에는 핵심 모델 코드가 **단 50줄**의 PyTorch 코드로 구현되어 있습니다. Nicolas Carion은 "간결함이 DETR의 가장 큰 장점"이라고 강조했는데, 복잡한 후처리 파이프라인 없이 Transformer만으로 탐지가 완성되기 때문입니다.

> 🔥 **실무 팁**: DETR 계열 모델을 선택할 때 — **정확도 최우선**이면 DINO, **실시간 + 정확도 균형**이면 RT-DETR, **연구/교육 목적**이면 원본 DETR을 추천합니다. 특히 RT-DETR은 Ultralytics에서 YOLO와 동일한 인터페이스로 사용할 수 있어 실무 전환이 쉽습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **DETR** | Transformer로 NMS 없이 End-to-End 탐지. 집합 예측 패러다임 |
| **Object Query** | N개의 학습 가능한 슬롯. 각각이 하나의 객체(또는 배경)를 담당 |
| **헝가리안 매칭** | 예측과 정답의 최적 1:1 매칭을 찾는 알고리즘 |
| **Deformable DETR** | 변형 가능 어텐션으로 수렴 10배 가속 + 작은 객체 개선 |
| **RT-DETR** | 실시간 DETR. YOLO를 속도-정확도 모두 능가 (CVPR 2024) |
| **NMS-Free** | 1:1 매칭으로 중복 예측이 구조적으로 불가능 |

## 다음 섹션 미리보기

이것으로 Chapter 07 객체 탐지를 마무리합니다! 바운딩 박스라는 사각형으로 객체의 "위치"를 찾았다면, 다음 Chapter 08 [이미지 분할](../08-segmentation/01-semantic-segmentation.md)에서는 **픽셀 단위로** 더 정밀하게 객체를 분리하는 방법을 배웁니다. "이 사진에서 고양이는 어디까지가 고양이인가?" — 에 답하는 기술이죠.

## 참고 자료

- [End-to-End Object Detection with Transformers (DETR, Carion et al., 2020)](https://arxiv.org/abs/2005.12872) - DETR 원논문, Transformer를 탐지에 도입
- [Deformable DETR (Zhu et al., 2021)](https://arxiv.org/abs/2010.04159) - 변형 가능 어텐션으로 DETR의 수렴 문제 해결
- [DETRs Beat YOLOs on Real-time Object Detection (RT-DETR, 2024)](https://arxiv.org/abs/2304.08069) - 실시간 DETR, CVPR 2024
- [Ultralytics RT-DETR 문서](https://docs.ultralytics.com/models/rtdetr/) - RT-DETR 사용 가이드
- [HuggingFace DETR 문서](https://huggingface.co/docs/transformers/model_doc/detr) - 원본 DETR 사용 가이드
- [DigitalOcean — DETR: The Role of Hungarian Algorithm](https://www.digitalocean.com/community/tutorials/introduction-detr-hungarian-algorithm-2) - 헝가리안 매칭 상세 설명
