# 객체 탐지 기초

> 바운딩 박스, IoU, NMS 이해

## 개요

지금까지 우리는 이미지 하나에 대해 "이건 고양이다"라고 **하나의 라벨**을 붙이는 분류(Classification) 문제를 풀었습니다. 하지만 현실 세계의 이미지에는 고양이도, 강아지도, 사람도 **동시에 여러 개** 존재하죠. **객체 탐지(Object Detection)**는 이미지 안에서 **무엇이 어디에 있는지**를 동시에 알아내는 기술입니다. 자율주행 자동차가 보행자와 차량을 인식하고, 스마트폰 카메라가 얼굴을 찾아 초점을 맞추는 것 — 모두 객체 탐지 덕분입니다.

**선수 지식**: [이미지 분류 실전](../06-image-classification/01-mnist.md), [CNN 아키텍처](../05-cnn-architectures/03-resnet.md)
**학습 목표**:
- 분류와 객체 탐지의 차이를 이해한다
- 바운딩 박스 표현 방식과 IoU 계산법을 알고 구현할 수 있다
- NMS(비최대 억제)의 원리와 필요성을 설명할 수 있다
- mAP 평가 지표를 이해하고 해석할 수 있다

## 왜 알아야 할까?

이미지 분류는 "이 사진에 뭐가 있나?"에 답하지만, 객체 탐지는 "뭐가 **어디에** 있나?"까지 답합니다. 이 차이가 엄청나거든요. 분류만으로는 자율주행차가 보행자의 **위치**를 알 수 없고, CCTV가 침입자의 **좌표**를 특정할 수 없습니다.

| 태스크 | 입력 | 출력 | 예시 |
|--------|------|------|------|
| **분류** | 이미지 1장 | 클래스 1개 | "고양이" |
| **객체 탐지** | 이미지 1장 | (클래스 + 위치) × N개 | "고양이(120,80,300,250), 강아지(400,100,550,320)" |
| **인스턴스 분할** | 이미지 1장 | (클래스 + 픽셀 마스크) × N개 | 각 객체의 정확한 윤곽선 |

객체 탐지는 컴퓨터 비전의 **가장 실용적인 태스크** 중 하나이며, 분류에서 분할로 가는 징검다리이기도 합니다. 이번 챕터에서는 이 핵심 기술의 기초 개념부터 최신 모델까지 차근차근 살펴봅니다.

## 핵심 개념

### 1. 바운딩 박스 — 객체의 "주소"를 표현하는 방법

> 💡 **비유**: 친구에게 "거실 왼쪽 아래 구석 쯤에 있는 리모컨 좀 가져다줘"라고 말하는 것처럼, 객체 탐지도 이미지 안에서 객체의 **위치를 사각형으로 지정**합니다. 이 사각형이 바로 **바운딩 박스(Bounding Box)**입니다.

바운딩 박스는 객체를 감싸는 최소 크기의 직사각형으로, 보통 두 가지 형식으로 표현합니다:

| 형식 | 표현 | 설명 |
|------|------|------|
| **(x1, y1, x2, y2)** | 좌상단 + 우하단 좌표 | Pascal VOC, torchvision 기본 형식 |
| **(cx, cy, w, h)** | 중심점 + 너비/높이 | YOLO, COCO 형식 |

> ⚠️ **흔한 오해**: "좌표계가 다 같다"고 생각하기 쉽지만, 데이터셋과 프레임워크마다 바운딩 박스 형식이 다릅니다! COCO는 **(x, y, w, h)** (좌상단 + 크기), Pascal VOC는 **(x1, y1, x2, y2)** (양 끝 좌표), YOLO는 **(cx, cy, w, h)** (중심 + 크기, 정규화). 모델을 학습하기 전에 **반드시 형식 변환을 확인**하세요. 이거 때문에 삽질하는 경우가 정말 많습니다.

두 형식 간의 변환은 간단합니다:

```python
import torch

def xyxy_to_cxcywh(boxes):
    """(x1, y1, x2, y2) → (cx, cy, w, h) 변환"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2  # 중심 x
    cy = (y1 + y2) / 2  # 중심 y
    w = x2 - x1          # 너비
    h = y2 - y1          # 높이
    return torch.stack([cx, cy, w, h], dim=-1)

def cxcywh_to_xyxy(boxes):
    """(cx, cy, w, h) → (x1, y1, x2, y2) 변환"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2  # 좌상단 x
    y1 = cy - h / 2  # 좌상단 y
    x2 = cx + w / 2  # 우하단 x
    y2 = cy + h / 2  # 우하단 y
    return torch.stack([x1, y1, x2, y2], dim=-1)

# 예시: (100, 50, 300, 250) → 중심(200, 150), 크기(200, 200)
box_xyxy = torch.tensor([[100., 50., 300., 250.]])
box_cxcywh = xyxy_to_cxcywh(box_xyxy)
print(f"xyxy: {box_xyxy[0].tolist()}")      # [100, 50, 300, 250]
print(f"cxcywh: {box_cxcywh[0].tolist()}")  # [200, 150, 200, 200]
```

### 2. IoU — 두 박스가 얼마나 겹치는지 측정하기

> 💡 **비유**: 두 장의 종이를 겹쳐 놓았을 때, **겹치는 부분**이 **합쳐진 전체 영역**에서 차지하는 비율이 바로 IoU입니다. 종이가 완전히 겹치면 IoU = 1, 전혀 겹치지 않으면 IoU = 0이 됩니다.

**IoU(Intersection over Union)**는 객체 탐지에서 **가장 중요한 지표**입니다. 예측한 바운딩 박스가 정답(Ground Truth)과 얼마나 일치하는지를 0~1 사이의 숫자로 표현합니다.

$$IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{교집합 영역}}{\text{합집합 영역}}$$

- $|A \cap B|$: 두 박스가 겹치는 영역의 넓이
- $|A \cup B|$: 두 박스를 합친 전체 영역의 넓이 = $|A| + |B| - |A \cap B|$

IoU 값의 의미를 직관적으로 보면:

| IoU 범위 | 의미 | 판정 |
|----------|------|------|
| 0.0 | 전혀 겹치지 않음 | 완전 빗나감 |
| 0.0 ~ 0.3 | 약간 겹침 | 나쁨 |
| 0.3 ~ 0.5 | 절반 정도 겹침 | 보통 |
| **0.5 이상** | 꽤 잘 겹침 | **일반적 정답 기준** |
| 0.75 이상 | 매우 잘 겹침 | 엄격한 기준 |
| 1.0 | 완벽하게 일치 | 이상적 |

IoU를 직접 구현하면 그 원리를 확실히 이해할 수 있습니다:

```python
import torch

def compute_iou(box1, box2):
    """
    두 바운딩 박스 세트의 IoU를 계산합니다.

    Args:
        box1: (N, 4) 형태, (x1, y1, x2, y2) 포맷
        box2: (M, 4) 형태, (x1, y1, x2, y2) 포맷
    Returns:
        iou: (N, M) 형태의 IoU 행렬
    """
    # 교집합의 좌상단 좌표 = 두 박스 좌상단 중 큰 값
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    # 교집합의 우하단 좌표 = 두 박스 우하단 중 작은 값
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    # 교집합 넓이 (겹치지 않으면 0)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # 각 박스의 넓이
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 합집합 = 넓이1 + 넓이2 - 교집합
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-6)  # 0으로 나누기 방지

# 테스트: 두 박스의 IoU 계산
gt_box = torch.tensor([[50., 50., 200., 200.]])     # 정답 박스
pred_box = torch.tensor([[80., 60., 220., 210.]])    # 예측 박스
iou = compute_iou(gt_box, pred_box)
print(f"IoU: {iou.item():.4f}")  # 약 0.56 — 꽤 잘 맞춘 편!
```

실무에서는 직접 구현하는 것보다 **torchvision의 내장 함수**를 사용하는 게 더 빠르고 안전합니다:

```python
from torchvision.ops import box_iou

gt_boxes = torch.tensor([[50., 50., 200., 200.],
                          [300., 100., 450., 300.]])
pred_boxes = torch.tensor([[80., 60., 220., 210.],
                            [310., 110., 440., 290.],
                            [0., 0., 50., 50.]])

# (2, 3) 크기의 IoU 행렬 — 모든 쌍에 대해 한 번에 계산
iou_matrix = box_iou(gt_boxes, pred_boxes)
print(f"IoU 행렬:\n{iou_matrix}")
# gt_box[0]과 pred_box[0]의 IoU, gt_box[0]과 pred_box[1]의 IoU, ...
```

> 💡 **알고 계셨나요?**: IoU는 **Jaccard Index(자카드 지수)**라고도 불리는데, 19세기 스위스 식물학자 Paul Jaccard가 식물 종의 유사도를 측정하기 위해 고안한 것입니다. 두 집합의 유사도를 측정하는 이 단순한 아이디어가 100년 넘게 살아남아 딥러닝 시대에도 핵심 지표로 쓰이고 있다니, 좋은 수학은 시대를 초월하는 것 같습니다.

### 3. NMS — 중복 탐지를 제거하는 후처리

> 💡 **비유**: 한 사람의 사진을 여러 명이 동시에 찍었다고 상상해보세요. 거의 같은 각도에서 찍은 사진이 10장이면, 그 중 **가장 잘 나온 1장**만 남기고 나머지는 지우는 게 합리적이죠? NMS(Non-Maximum Suppression, 비최대 억제)가 하는 일이 정확히 이겁니다 — **같은 객체에 대한 중복 박스 중 가장 자신감 있는 것만 남깁니다.**

객체 탐지 모델은 보통 하나의 객체에 대해 **수십~수백 개의 바운딩 박스**를 예측합니다. 이 중 대부분은 같은 객체를 가리키는 중복 예측이에요. NMS는 이 중복을 정리하는 후처리 알고리즘입니다.

**NMS 알고리즘 단계:**

1. 모든 예측 박스를 **신뢰도(confidence) 점수** 기준으로 내림차순 정렬
2. 가장 점수가 높은 박스를 **최종 결과에 추가**
3. 나머지 박스 중, 선택된 박스와 IoU가 **임계값(보통 0.5) 이상**인 박스를 **제거**
4. 남은 박스 중 다시 가장 점수가 높은 박스를 선택
5. 박스가 없을 때까지 2~4를 반복

직접 구현해보면 이해가 쉽습니다:

```python
import torch

def nms_manual(boxes, scores, iou_threshold=0.5):
    """
    NMS를 직접 구현합니다.

    Args:
        boxes: (N, 4) — (x1, y1, x2, y2) 형식
        scores: (N,) — 각 박스의 신뢰도 점수
        iou_threshold: IoU 임계값 (이 이상 겹치면 제거)
    Returns:
        keep: 살아남은 박스의 인덱스 리스트
    """
    # 점수 기준 내림차순 정렬
    order = scores.argsort(descending=True)
    keep = []

    while len(order) > 0:
        # 1) 가장 높은 점수의 박스를 선택
        best_idx = order[0]
        keep.append(best_idx.item())

        if len(order) == 1:
            break

        # 2) 나머지 박스들과의 IoU 계산
        remaining = order[1:]
        ious = compute_iou(
            boxes[best_idx].unsqueeze(0),
            boxes[remaining]
        ).squeeze(0)

        # 3) IoU가 임계값 미만인 박스만 남김 (겹치는 건 제거)
        mask = ious < iou_threshold
        order = remaining[mask]

    return keep

# 예시: 같은 고양이에 대한 여러 예측
boxes = torch.tensor([
    [100., 100., 300., 300.],  # 박스 0: 점수 0.9
    [110., 105., 310., 305.],  # 박스 1: 점수 0.85 (박스0과 많이 겹침)
    [105., 98., 295., 295.],   # 박스 2: 점수 0.7  (박스0과 많이 겹침)
    [400., 200., 550., 400.],  # 박스 3: 점수 0.8  (다른 객체!)
])
scores = torch.tensor([0.9, 0.85, 0.7, 0.8])

keep = nms_manual(boxes, scores, iou_threshold=0.5)
print(f"NMS 전: 박스 {len(boxes)}개")      # 4개
print(f"NMS 후: 박스 {len(keep)}개")       # 2개 (박스 0, 3만 생존)
print(f"살아남은 인덱스: {keep}")            # [0, 3]
```

실무에서는 **torchvision.ops.nms**를 쓰면 C++ 커널 덕분에 훨씬 빠릅니다:

```python
from torchvision.ops import nms

# torchvision의 NMS — GPU 가속 가능!
keep_indices = nms(boxes, scores, iou_threshold=0.5)
print(f"torchvision NMS 결과: {keep_indices.tolist()}")  # [0, 3]
```

> 🔥 **실무 팁**: NMS의 IoU 임계값은 결과에 큰 영향을 줍니다. **임계값이 낮으면** (예: 0.3) 박스를 공격적으로 제거해서 밀집된 객체를 놓칠 수 있고, **임계값이 높으면** (예: 0.7) 중복 박스가 많이 남아 같은 객체에 여러 박스가 붙습니다. 일반적으로 **0.45~0.5**가 좋은 시작점이에요.

### 4. 평가 지표 — 탐지 모델이 얼마나 잘하는지 측정하기

분류에서는 정확도(Accuracy) 하나로 성능을 측정했지만, 객체 탐지에서는 **위치까지 맞춰야** 하기 때문에 더 복잡한 지표가 필요합니다.

#### Precision과 Recall

먼저 예측 결과를 분류합니다:

| 판정 | 조건 | 의미 |
|------|------|------|
| **TP** (True Positive) | IoU ≥ 임계값인 정답 매칭 | 올바른 탐지 |
| **FP** (False Positive) | IoU < 임계값 또는 중복 매칭 | 잘못된 탐지 (오탐) |
| **FN** (False Negative) | 매칭되지 못한 정답 박스 | 놓친 객체 (미탐) |

이를 바탕으로:

$$Precision = \frac{TP}{TP + FP}$$

- "탐지한 것 중 실제 맞는 비율" — **정밀도가 높으면** 오탐이 적습니다

$$Recall = \frac{TP}{TP + FN}$$

- "실제 객체 중 찾아낸 비율" — **재현율이 높으면** 놓치는 게 적습니다

#### AP와 mAP

Precision과 Recall은 신뢰도 임계값에 따라 변합니다. 임계값을 높이면 Precision은 올라가지만 Recall이 떨어지고, 낮추면 그 반대죠. **AP(Average Precision)**는 이 trade-off를 하나의 숫자로 요약합니다 — Precision-Recall 곡선 아래의 면적을 구하는 거예요.

**mAP(Mean Average Precision)**는 모든 클래스의 AP를 평균한 값입니다:

$$mAP = \frac{1}{C}\sum_{i=1}^{C} AP_i$$

여기서 $C$는 클래스 수입니다.

데이터셋별로 mAP 계산 기준이 다르다는 점을 꼭 기억하세요:

| 데이터셋 | 기준 | 표기 |
|----------|------|------|
| **Pascal VOC** | IoU ≥ 0.5에서만 계산 | mAP@0.5 |
| **COCO** | IoU 0.5~0.95까지 0.05 간격으로 10번 계산 후 평균 | mAP@[.5:.95] |
| COCO (추가) | 객체 크기별 분리 평가 (small/medium/large) | AP_S, AP_M, AP_L |

> ⚠️ **흔한 오해**: 논문에서 "mAP 50%"와 "mAP@0.5에서 80%"를 혼동하기 쉽습니다. **COCO mAP는 매우 엄격한 지표**라서, COCO에서 mAP 50이면 상당히 좋은 모델입니다. 같은 모델이 Pascal VOC 기준(mAP@0.5)에서는 75% 이상 나올 수 있어요. 논문을 읽을 때 **어떤 기준의 mAP인지** 반드시 확인하세요!

### 5. 객체 탐지의 두 갈래 — Two-Stage vs One-Stage

객체 탐지 모델은 크게 두 가지 접근 방식으로 나뉩니다:

**Two-Stage Detector (2단계 탐지기)**
1. 먼저 "객체가 있을 법한 후보 영역(Region Proposal)"을 추출
2. 각 후보 영역을 분류하고 박스를 정밀 조정

대표 모델: **R-CNN → Fast R-CNN → Faster R-CNN**
- 장점: 높은 정확도
- 단점: 느린 속도 (두 단계를 거치므로)

**One-Stage Detector (1단계 탐지기)**
1. 이미지를 그리드로 나누고, 각 그리드에서 **바로** 클래스와 위치를 예측
2. 한 번에 끝! 별도의 후보 영역 추출 없음

대표 모델: **YOLO, SSD, RetinaNet**
- 장점: 빠른 속도 (실시간 가능)
- 단점: 작은 객체나 밀집 객체에서 정확도가 다소 낮음 (점점 개선 중)

| 특성 | Two-Stage | One-Stage |
|------|-----------|-----------|
| 속도 | 느림 (5~20 FPS) | 빠름 (30~150+ FPS) |
| 정확도 | 높음 | 약간 낮음 → 최근 비슷 |
| 대표 모델 | Faster R-CNN | YOLO, SSD |
| 용도 | 의료 영상, 정밀 분석 | 자율주행, 실시간 감시 |

> 💡 **알고 계셨나요?**: 최근에는 **Transformer 기반의 DETR**이 등장하면서, "Two-Stage vs One-Stage" 구분 자체가 무색해지고 있습니다. DETR은 NMS조차 필요 없는 **End-to-End** 방식으로, 이후 섹션에서 자세히 다룹니다.

## 실습: 객체 탐지 파이프라인 체험하기

전체 탐지 파이프라인을 코드로 체험해봅시다. torchvision의 사전 학습된 Faster R-CNN을 사용해 실제 이미지에서 객체를 탐지합니다:

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# COCO 데이터셋으로 학습된 Faster R-CNN 로드
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

# 전처리 함수 (weights에서 제공)
preprocess = weights.transforms()

# 랜덤 이미지로 테스트 (실제로는 PIL 이미지를 로드합니다)
dummy_image = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
input_tensor = preprocess(dummy_image)

# 추론 실행
with torch.no_grad():
    predictions = model([input_tensor])

# 결과 확인
pred = predictions[0]
print(f"탐지된 박스 수: {len(pred['boxes'])}")
print(f"박스 형태: {pred['boxes'].shape}")     # (N, 4) — x1, y1, x2, y2
print(f"라벨 형태: {pred['labels'].shape}")    # (N,) — 클래스 ID
print(f"점수 형태: {pred['scores'].shape}")    # (N,) — 신뢰도 0~1

# 신뢰도 0.5 이상만 필터링
high_conf = pred['scores'] > 0.5
boxes = pred['boxes'][high_conf]
labels = pred['labels'][high_conf]
scores = pred['scores'][high_conf]

# COCO 클래스 이름 매핑 (일부)
coco_names = weights.meta["categories"]
for i in range(len(boxes)):
    class_name = coco_names[labels[i]]
    score = scores[i].item()
    box = boxes[i].tolist()
    print(f"  {class_name}: {score:.2f} @ [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
```

## 더 깊이 알아보기

### 객체 탐지의 60년 역사

객체 탐지의 역사는 딥러닝 이전으로 거슬러 올라갑니다. Zou et al.의 서베이 논문 *"Object Detection in 20 Years"*에 따르면, 객체 탐지는 크게 세 시대로 나뉩니다:

1. **전통적 방법 시대 (2001~2012)**: Viola-Jones 얼굴 검출기(2001), HOG + SVM(2005), DPM(2008) 등 **수작업 특징(Hand-crafted Features)**에 의존
2. **CNN 기반 시대 (2012~2020)**: R-CNN(2014)이 시작을 열고, Faster R-CNN, YOLO, SSD 등이 연이어 등장. ImageNet에서 폭발한 딥러닝이 탐지 분야를 완전히 뒤바꿈
3. **Transformer 시대 (2020~현재)**: DETR(2020)이 Transformer를 도입하면서, NMS 없는 End-to-End 탐지가 가능해짐

특히 2001년 Paul Viola와 Michael Jones가 발표한 **Viola-Jones 검출기**는 웹캠에서 **실시간 얼굴 인식**을 최초로 가능하게 한 획기적인 알고리즘이었습니다. Haar-like 특징과 Cascade 구조를 사용했는데, 이 기술은 지금도 OpenCV의 `cv2.CascadeClassifier`로 사용할 수 있을 만큼 오랫동안 사랑받고 있습니다.

### GIoU, DIoU, CIoU — IoU의 진화

기본 IoU에는 한 가지 문제가 있습니다 — 두 박스가 전혀 겹치지 않으면 IoU = 0이라서 **"얼마나 멀리 떨어져 있는지"** 구분이 안 됩니다. 이를 보완한 변형들이 있습니다:

| 지표 | 특징 | 활용 |
|------|------|------|
| **GIoU** | 두 박스를 감싸는 최소 박스를 고려 | 손실 함수에 사용 |
| **DIoU** | 중심점 간의 거리를 추가로 고려 | 더 빠른 수렴 |
| **CIoU** | 거리 + 종횡비(aspect ratio)까지 고려 | YOLO에서 자주 사용 |

이런 변형들은 **손실 함수**로 사용할 때 특히 중요합니다. "얼마나 틀렸는지"를 더 세밀하게 알려주기 때문이죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "IoU 0.5면 절반이 겹친다"고 생각하기 쉽지만, 실제로 IoU 0.5는 교집합이 합집합의 절반이라는 뜻이지, 면적의 절반이 겹친다는 뜻이 아닙니다. 시각적으로 보면 IoU 0.5도 **꽤 많이 겹쳐 보입니다**.

> 🔥 **실무 팁**: COCO 데이터셋에서 모델을 평가할 때, `pycocotools` 라이브러리의 `COCOeval`을 사용하면 표준화된 방식으로 mAP를 계산할 수 있습니다. 직접 구현하면 미묘한 차이가 생길 수 있으니, 공식 도구를 쓰는 것이 안전합니다.

> 💡 **알고 계셨나요?**: Soft-NMS라는 변형은 겹치는 박스를 완전히 제거하는 대신 **점수를 낮추는** 방식입니다. 밀집된 객체(사람 군중 등)를 탐지할 때 성능이 더 좋은 경우가 많습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **바운딩 박스** | 객체 위치를 사각형 좌표로 표현. (x1,y1,x2,y2) 또는 (cx,cy,w,h) 형식 |
| **IoU** | 두 박스의 겹침 비율 (0~1). 탐지 정확도의 핵심 지표 |
| **NMS** | 중복 박스 제거 알고리즘. 가장 자신감 높은 박스만 남김 |
| **mAP** | 클래스별 AP의 평균. COCO는 여러 IoU 임계값의 평균 |
| **Two-Stage** | 후보 영역 → 분류 (정확하지만 느림) |
| **One-Stage** | 한 번에 분류+위치 예측 (빠르지만 정확도 trade-off) |

## 다음 섹션 미리보기

이제 객체 탐지의 기초 개념을 단단히 잡았으니, 다음 섹션에서는 **딥러닝 기반 객체 탐지의 문을 연 [R-CNN 계열](./02-rcnn-family.md)**을 살펴봅니다. R-CNN에서 Faster R-CNN까지, "느리지만 정확한" Two-Stage 탐지기의 진화 과정을 따라가 봅시다.

## 참고 자료

- [Object Detection in 20 Years: A Survey (Zou et al., 2023)](https://arxiv.org/abs/1905.05055) - 객체 탐지 60년 역사를 정리한 종합 서베이
- [torchvision.ops — NMS, IoU 공식 문서](https://pytorch.org/vision/stable/ops.html) - PyTorch 공식 탐지 관련 함수 문서
- [LearnOpenCV — NMS Theory and Implementation](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/) - NMS의 이론과 PyTorch 구현 튜토리얼
- [COCO Detection Evaluation](https://cocodataset.org/#detection-eval) - COCO 데이터셋 공식 평가 기준 설명
- [PseudoLab 객체 탐지 소개](https://pseudo-lab.github.io/Tutorial-Book/chapters/object-detection/Ch1-Object-Detection.html) - 한국어 객체 탐지 입문 자료
