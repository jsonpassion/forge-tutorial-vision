# Anchor-Free 탐지기

> FCOS, CenterNet, CornerNet

## 개요

지금까지 본 Faster R-CNN과 초기 YOLO는 모두 **앵커 박스(Anchor Box)**를 사용했습니다. 미리 정해둔 다양한 크기와 비율의 기준 박스를 깔아놓고, 이를 조정하여 객체를 찾는 방식이었죠. 하지만 2018~2019년, 연구자들은 "앵커 없이도 탐지할 수 있지 않을까?"라는 도전적인 질문을 던졌습니다. 그 결과 **CornerNet, CenterNet, FCOS** 같은 Anchor-Free 탐지기가 등장했고, 이 패러다임은 이후 YOLOv8에도 채택될 만큼 큰 영향을 미쳤습니다.

**선수 지식**: [객체 탐지 기초](./01-detection-basics.md), [R-CNN 계열](./02-rcnn-family.md)의 앵커 박스 개념
**학습 목표**:
- 앵커 박스의 문제점과 Anchor-Free 접근의 동기를 이해한다
- CornerNet, CenterNet, FCOS 각각의 핵심 아이디어를 설명할 수 있다
- 키포인트 기반 vs 밀집 예측 기반 Anchor-Free 방식의 차이를 안다
- Anchor-Free가 현대 탐지기에 미친 영향을 이해한다

## 왜 알아야 할까?

앵커 기반 탐지기에는 몇 가지 골치 아픈 문제가 있습니다:

| 문제 | 설명 |
|------|------|
| **하이퍼파라미터 지옥** | 앵커 크기, 비율, 수를 얼마로 할지 — 데이터셋마다 달라야 함 |
| **양/음 샘플 불균형** | 수만 개 앵커 중 실제 객체와 매칭되는 건 극소수 |
| **IoU 기반 매칭의 모호함** | 같은 객체에 여러 앵커가 매칭되거나, 어디에도 매칭 안 되는 경우 |
| **계산 낭비** | 대부분의 앵커는 배경 — 쓸데없는 계산이 많음 |

Anchor-Free 탐지기는 이 모든 문제를 **앵커 자체를 없애버림**으로써 해결합니다. 더 단순하고, 하이퍼파라미터가 적고, 직관적인 설계가 가능해진 거죠.

## 핵심 개념

### 1. CornerNet (2018) — 모서리로 객체를 찾자

> 💡 **비유**: 바닥에 놓인 상자를 보고 "이건 가로 30cm, 세로 20cm 상자야"라고 말하는 대신, **"왼쪽 위 모서리가 여기, 오른쪽 아래 모서리가 저기"**라고 두 점만 찍으면 상자가 완벽하게 정의됩니다. CornerNet이 바로 이 아이디어입니다!

CornerNet(Hao Law, Jia Deng, 2018)은 앵커 박스를 완전히 제거한 최초의 경쟁력 있는 탐지기입니다.

**핵심 아이디어:** 바운딩 박스를 **좌상단 코너 + 우하단 코너**, 두 개의 키포인트로 탐지

**CornerNet의 동작 방식:**

1. 이미지를 백본 네트워크(Hourglass Network)에 통과
2. **Top-Left 히트맵**: 좌상단 코너 위치 예측 (클래스별)
3. **Bottom-Right 히트맵**: 우하단 코너 위치 예측 (클래스별)
4. **Embedding 벡터**: 같은 객체의 두 코너를 매칭 (유사한 임베딩 = 같은 객체)
5. 매칭된 코너 쌍으로 바운딩 박스 생성

> ⚠️ **흔한 오해**: "코너 두 개만 찾으면 간단하겠다"고 생각할 수 있지만, 실제로 **어떤 좌상단 코너와 어떤 우하단 코너가 같은 객체인지 매칭하는 게** 핵심 난제입니다. CornerNet은 이를 위해 Associative Embedding이라는 기법을 사용합니다.

CornerNet은 MS COCO에서 **42.2% AP**를 달성하며, 앵커 없이도 경쟁력 있는 성능이 가능함을 처음으로 증명했습니다.

### 2. CenterNet (2019) — 객체는 하나의 점이다

> 💡 **비유**: 지도에서 도시를 표시할 때, 도시의 경계를 다 그리는 대신 **중심에 점 하나**를 찍고 크기 정보를 옆에 적으면 충분하죠? CenterNet은 이 발상으로 객체를 탐지합니다 — **객체의 중심점 하나**를 찾고, 거기서 크기를 예측하면 끝!

CenterNet("Objects as Points", Xingyi Zhou et al., 2019)은 CornerNet보다 훨씬 간결한 접근을 제시합니다.

**핵심 아이디어:** 객체 = 중심점(Center Point). 중심을 찾으면 크기는 거기서 회귀하면 된다.

**CenterNet의 동작 방식:**

1. 이미지를 백본(ResNet, DLA, Hourglass 등)에 통과
2. **Center 히트맵**: 각 클래스별 객체 중심점의 확률 맵 예측
3. **크기 회귀**: 중심점 위치에서 (w, h)를 예측
4. **오프셋 보정**: 다운샘플링에 의한 좌표 오차 보정
5. 히트맵의 피크(peak)를 추출하면 바로 탐지 결과!

**CenterNet의 혁신적인 점들:**

- **NMS가 필요 없음**: 히트맵의 피크가 곧 객체이므로, 중복 제거가 내재적으로 해결됨
- **단순한 구조**: 클래스 히트맵 + 크기 + 오프셋, 단 3개의 출력 헤드
- **범용성**: 같은 프레임워크로 2D 탐지, 3D 탐지, 포즈 추정까지 가능

```python
import torch
import torch.nn as nn

class SimpleCenterNetHead(nn.Module):
    """
    CenterNet의 핵심 아이디어를 보여주는 간소화된 탐지 헤드.
    실제 CenterNet은 더 정교한 구조를 사용합니다.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 1) 클래스별 중심점 히트맵 (H/4 × W/4 × C)
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),
            nn.Sigmoid()  # 0~1 확률값
        )
        # 2) 바운딩 박스 크기 (H/4 × W/4 × 2)
        self.size = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)  # (w, h) 예측
        )
        # 3) 오프셋 보정 (H/4 × W/4 × 2)
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)  # (dx, dy) 보정
        )

    def forward(self, features):
        hm = self.heatmap(features)    # 중심점 확률
        wh = self.size(features)       # 너비, 높이
        off = self.offset(features)    # 좌표 보정값
        return hm, wh, off

# 사용 예시
head = SimpleCenterNetHead(in_channels=256, num_classes=80)
fake_features = torch.randn(1, 256, 120, 160)  # 백본 출력 가정

heatmap, sizes, offsets = head(fake_features)
print(f"히트맵: {heatmap.shape}")    # [1, 80, 120, 160] — 클래스별 중심점 확률
print(f"크기: {sizes.shape}")        # [1, 2, 120, 160] — 각 위치의 (w, h)
print(f"오프셋: {offsets.shape}")    # [1, 2, 120, 160] — 좌표 보정값
```

히트맵에서 피크를 추출하는 것이 곧 NMS를 대체합니다:

```python
def extract_peaks(heatmap, kernel_size=3, threshold=0.3):
    """
    히트맵에서 로컬 피크를 추출합니다 (NMS 대체).
    맥스 풀링으로 로컬 최대값만 남깁니다.
    """
    # 맥스 풀링으로 로컬 최대값 찾기
    pad = (kernel_size - 1) // 2
    hmax = torch.nn.functional.max_pool2d(
        heatmap, kernel_size, stride=1, padding=pad
    )
    # 로컬 최대값인 위치만 남김
    keep = (hmax == heatmap).float()
    peaks = heatmap * keep

    # 임계값 이상인 피크만 선택
    mask = peaks > threshold
    return peaks, mask

# 예시: 80개 클래스의 히트맵에서 피크 추출
peaks, mask = extract_peaks(heatmap)
num_detections = mask.sum().item()
print(f"탐지된 객체 수: {num_detections}")
```

### 3. FCOS (2019) — 모든 픽셀이 탐지기다

> 💡 **비유**: CornerNet이 "모서리 두 개를 찾아라", CenterNet이 "중심 하나를 찾아라"였다면, FCOS는 **"모든 픽셀이 자기가 어떤 객체 안에 있는지 판단하라"**입니다. 마치 **시맨틱 분할(Semantic Segmentation)처럼** 객체 탐지를 푸는 거죠!

FCOS(Fully Convolutional One-Stage, Zhi Tian et al., 2019)는 객체 탐지를 **픽셀 단위 예측** 문제로 재정의했습니다.

**핵심 아이디어:** 특징 맵의 각 픽셀이 자신이 속한 객체의 경계까지의 **4방향 거리(l, t, r, b)**를 예측

각 픽셀에서 예측하는 값:
- **(l, t, r, b)**: 좌/상/우/하 경계까지의 거리 → 이 4개 값으로 바운딩 박스 복원
- **클래스 점수**: C개 클래스에 대한 확률
- **Center-ness**: 중심에 가까울수록 1, 멀수록 0 → 품질 낮은 예측 억제

**FCOS의 동작 방식:**

1. **FPN 백본**: 다양한 스케일의 특징 맵 생성 (P3~P7)
2. **각 픽셀에서 예측**: (l, t, r, b) 거리 + 클래스 확률 + center-ness
3. **FPN 레벨별 크기 제한**: 작은 객체는 P3, 큰 객체는 P7에서 탐지
4. **Center-ness로 필터링**: 객체 경계에 가까운 저품질 예측 억제

> 💡 **알고 계셨나요?**: FCOS의 "center-ness" 아이디어는 간단하지만 강력합니다. 물체의 중심에서 멀어진 픽셀일수록 예측 품질이 떨어지는 경향이 있는데, center-ness 점수를 분류 점수에 곱해주면 **이런 저품질 예측을 자동으로 억제**할 수 있습니다. 이 아이디어는 이후 많은 탐지기에 채택되었습니다.

### 4. 세 가지 접근의 비교

| 모델 | 접근 방식 | 핵심 출력 | NMS 필요 | 성능 (COCO AP) |
|------|-----------|-----------|----------|---------------|
| **CornerNet** | 좌상단 + 우하단 코너 | 코너 히트맵 + 임베딩 | 예 | 42.2% |
| **CenterNet** | 객체 중심점 | 중심 히트맵 + 크기 | **아니오** | 45.1% |
| **FCOS** | 픽셀별 거리 예측 | 4방향 거리 + center-ness | 예 | 44.7% |

**어떤 것을 선택할까?**

- **최대한 단순한 구조**를 원한다면 → **CenterNet** (NMS도 불필요)
- **FPN과 결합한 강력한 성능**을 원한다면 → **FCOS**
- **키포인트 기반 접근에 관심**이 있다면 → **CornerNet**

## 실습: torchvision의 FCOS 사용하기

torchvision은 FCOS의 공식 구현을 제공합니다:

```python
import torch
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

# 사전 학습된 FCOS 모델 로드
weights = FCOS_ResNet50_FPN_Weights.DEFAULT
model = fcos_resnet50_fpn(weights=weights)
model.eval()

# 전처리
preprocess = weights.transforms()

# 테스트 이미지 (랜덤)
dummy_image = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
input_tensor = preprocess(dummy_image)

# 추론
with torch.no_grad():
    predictions = model([input_tensor])

pred = predictions[0]
print(f"탐지된 박스 수: {len(pred['boxes'])}")

# 신뢰도 필터링
mask = pred['scores'] > 0.5
boxes = pred['boxes'][mask]
labels = pred['labels'][mask]
scores = pred['scores'][mask]

categories = weights.meta["categories"]
for i in range(min(5, len(boxes))):  # 상위 5개만 출력
    name = categories[labels[i]]
    print(f"  {name}: {scores[i]:.3f}")
```

Faster R-CNN과 FCOS를 나란히 비교하면:

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
import time

def benchmark_model(model, preprocess, name, num_runs=50):
    """모델의 추론 속도를 벤치마크합니다."""
    model.eval()
    dummy = preprocess(torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8))

    # 워밍업
    with torch.no_grad():
        for _ in range(5):
            model([dummy])

    # 측정
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model([dummy])
    elapsed = (time.time() - start) / num_runs

    print(f"{name}: {elapsed*1000:.1f}ms/image ({1/elapsed:.1f} FPS)")

# Faster R-CNN vs FCOS 비교
frcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
frcnn = fasterrcnn_resnet50_fpn_v2(weights=frcnn_weights)

fcos_weights = FCOS_ResNet50_FPN_Weights.DEFAULT
fcos = fcos_resnet50_fpn(weights=fcos_weights)

benchmark_model(frcnn, frcnn_weights.transforms(), "Faster R-CNN")
benchmark_model(fcos, fcos_weights.transforms(), "FCOS")
# 일반적으로 FCOS가 Faster R-CNN보다 빠릅니다 (2-stage vs 1-stage)
```

## 더 깊이 알아보기

### Anchor-Free의 탄생 배경 — DenseBox에서 FCOS까지

사실 Anchor-Free 탐지의 아이디어는 CornerNet(2018)보다 훨씬 이전인 **2015년의 DenseBox**까지 거슬러 올라갑니다. DenseBox는 각 픽셀에서 바운딩 박스를 직접 예측했지만, 당시에는 앵커 기반 방법(Faster R-CNN, SSD)에 성능이 밀려서 주목받지 못했습니다.

그러다 2018년 CornerNet이 "앵커 없이도 경쟁력 있는 성능을 낼 수 있다"는 것을 증명하면서, Anchor-Free 연구가 폭발적으로 늘어났습니다. FCOS의 저자 Zhi Tian은 FCOS 논문에서 "왜 이전의 Anchor-Free 방법들이 성능이 낮았는지" 분석하고, **FPN + Center-ness**라는 두 가지 기법으로 문제를 해결했습니다.

이 연구들의 영향은 거대했습니다 — **YOLOv8이 앵커 기반에서 Anchor-Free로 전환**한 것이 가장 상징적인 사례입니다. 현재 최신 탐지기 대부분이 Anchor-Free 방식을 채택하고 있습니다.

### 키포인트 기반 vs 밀집 예측 기반

Anchor-Free 탐지기는 크게 두 갈래로 나뉩니다:

**키포인트 기반 (Keypoint-based):**
- CornerNet (좌상단 + 우하단 코너)
- CenterNet (중심점)
- ExtremeNet (상/하/좌/우 극단점)
- 특징: 특정 키포인트를 찾아 박스 구성

**밀집 예측 기반 (Dense Prediction):**
- FCOS (각 픽셀에서 거리 예측)
- FoveaBox (중심 영역에서 박스 예측)
- 특징: 시맨틱 분할처럼 모든 위치에서 예측

최근 추세는 두 접근의 장점을 합치는 방향으로 가고 있으며, YOLO11의 구조도 이 융합의 결과물입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Anchor-Free가 항상 앵커 기반보다 좋다"는 아닙니다. 특히 **밀집된 작은 객체**(예: 항공 사진의 차량)에서는 앵커 기반이 더 안정적인 경우도 있습니다. 데이터셋 특성에 따라 적합한 방법이 다릅니다.

> 🔥 **실무 팁**: torchvision에서 Anchor-Free 모델을 쓰려면 `fcos_resnet50_fpn`을 사용하세요. Faster R-CNN과 API가 동일해서 **코드 한 줄만 바꾸면** 바로 비교 실험을 할 수 있습니다.

> 💡 **알고 계셨나요?**: CenterNet의 저자 Xingyi Zhou는 같은 "Objects as Points" 프레임워크로 **2D 탐지, 3D 탐지, 포즈 추정**을 모두 수행할 수 있음을 보여줬습니다. 중심점에서 예측하는 속성만 바꾸면 되거든요 — (w, h)를 예측하면 2D 탐지, (w, h, depth, orientation)을 예측하면 3D 탐지, 관절 좌표를 예측하면 포즈 추정!

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **Anchor-Free** | 미리 정의된 앵커 박스 없이 객체를 탐지하는 패러다임 |
| **CornerNet** | 좌상단 + 우하단 코너 2개로 객체 탐지 (키포인트 기반) |
| **CenterNet** | 객체 중심점 1개 + 크기 회귀. NMS 불필요 |
| **FCOS** | 각 픽셀에서 4방향 거리 + center-ness 예측 (밀집 기반) |
| **Center-ness** | 중심에서 먼 저품질 예측을 억제하는 FCOS의 핵심 기법 |
| **현대적 영향** | YOLOv8이 Anchor-Free 채택 → 업계 표준으로 자리잡음 |

## 다음 섹션 미리보기

Anchor-Free 탐지기가 "앵커"를 제거했다면, 다음에 소개할 [DETR](./05-detr.md)은 **NMS까지** 제거합니다. Transformer를 객체 탐지에 도입하여, 후보 추출도, 중복 제거도 없이 **End-to-End로** 탐지하는 혁신적 접근을 살펴봅시다.

## 참고 자료

- [CornerNet: Detecting Objects as Paired Keypoints (Law & Deng, 2018)](https://arxiv.org/abs/1808.01244) - 앵커 없는 탐지의 시작
- [Objects as Points (CenterNet, Zhou et al., 2019)](https://arxiv.org/abs/1904.07850) - 객체를 점으로 모델링하는 혁신적 접근
- [FCOS: Fully Convolutional One-Stage Object Detection (Tian et al., 2019)](https://arxiv.org/abs/1904.01355) - 픽셀 단위 Anchor-Free 탐지
- [LearnOpenCV — FCOS Explained](https://learnopencv.com/fcos-anchor-free-object-detection-explained/) - FCOS 아키텍처 상세 설명
- [LearnOpenCV — CenterNet Explained](https://learnopencv.com/centernet-anchor-free-object-detection-explained/) - CenterNet 동작 원리 튜토리얼
- [torchvision FCOS 문서](https://pytorch.org/vision/stable/models/fcos.html) - PyTorch 공식 FCOS 가이드
