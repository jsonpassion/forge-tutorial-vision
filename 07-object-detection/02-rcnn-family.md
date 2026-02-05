# R-CNN 계열

> R-CNN, Fast R-CNN, Faster R-CNN

## 개요

딥러닝 기반 객체 탐지의 역사는 **R-CNN**에서 시작되었습니다. 2014년 Ross Girshick이 CNN을 객체 탐지에 적용하면서, Pascal VOC 벤치마크에서 mAP를 **30% 이상 끌어올리는** 혁명을 일으켰거든요. 하지만 R-CNN은 너무 느렸습니다. 이후 Fast R-CNN, Faster R-CNN으로 진화하며 속도와 정확도를 동시에 개선했는데, 이 진화 과정을 따라가면 객체 탐지의 핵심 아이디어를 자연스럽게 이해할 수 있습니다.

**선수 지식**: [객체 탐지 기초](./01-detection-basics.md)에서 배운 바운딩 박스, IoU, NMS
**학습 목표**:
- R-CNN → Fast R-CNN → Faster R-CNN의 진화 과정과 각 단계의 핵심 개선점을 이해한다
- Region Proposal과 RoI Pooling의 개념을 설명할 수 있다
- Anchor Box와 Region Proposal Network(RPN)의 원리를 이해한다
- torchvision의 Faster R-CNN을 사용해 커스텀 데이터에 적용할 수 있다

## 왜 알아야 할까?

R-CNN 계열은 **Two-Stage Detector의 교과서**입니다. "먼저 후보를 찾고, 각 후보를 분류한다"는 이 아이디어는 단순하지만 강력해서, 2014년부터 지금까지도 Faster R-CNN이 **기준 모델(baseline)**로 쓰이고 있습니다. 의료 영상에서 종양을 찾거나, 위성 사진에서 건물을 탐지하는 등 **정확도가 최우선**인 분야에서는 여전히 Faster R-CNN 기반 모델을 선택하는 경우가 많습니다.

또한 R-CNN의 진화 과정은 딥러닝 연구의 전형적인 패턴을 보여줍니다 — "먼저 되게 만들고, 그다음 빠르게 만들고, 그다음 더 빠르게 만든다."

## 핵심 개념

### 1. R-CNN (2014) — "일단 되게 만들자"

> 💡 **비유**: 편지를 분류하는 우체부를 상상해보세요. R-CNN 방식은 이렇습니다: (1) 편지 더미에서 **편지처럼 생긴 것 2,000개를 골라내고** (Region Proposal), (2) 각각을 한 장씩 꺼내서 **자세히 읽어보고** (CNN 특징 추출), (3) 어느 지역으로 보낼지 **판단합니다** (SVM 분류). 한 장씩 읽어봐야 하니 당연히 느리겠죠?

R-CNN의 핵심 아이디어는 단순합니다: **"CNN이 이미지 분류를 잘한다면, 이미지의 각 부분에도 CNN을 적용하면 되지 않을까?"**

**R-CNN 처리 과정:**

1. **Region Proposal (후보 영역 추출)**: Selective Search 알고리즘으로 객체가 있을 법한 **~2,000개의 후보 영역**을 추출
2. **CNN 특징 추출**: 각 후보를 227×227로 리사이즈하고, AlexNet에 통과시켜 **4,096차원 특징 벡터** 추출
3. **분류**: SVM으로 각 후보의 클래스 판별
4. **위치 보정**: 선형 회귀로 바운딩 박스 좌표 미세 조정

**R-CNN의 문제점:**

| 문제 | 원인 |
|------|------|
| **매우 느림** (이미지 1장에 ~47초) | 2,000개 후보마다 CNN을 개별 실행 |
| **학습이 복잡** | CNN, SVM, 박스 회귀를 따로 훈련 |
| **디스크 공간 낭비** | 모든 후보의 특징 벡터를 저장 |

그래도 R-CNN은 Pascal VOC 2012에서 mAP **53.3%**를 달성하며, 이전 최고 기록보다 30% 이상 높은 성과를 보여줬습니다. "CNN + 객체 탐지"라는 방향성이 옳다는 것을 증명한 거죠.

### 2. Fast R-CNN (2015) — "한 번만 보면 되잖아!"

> 💡 **비유**: R-CNN이 2,000통의 편지를 **한 장씩** 읽었다면, Fast R-CNN은 **모든 편지를 한 번에 복사기에 넣고 스캔**한 뒤, 필요한 부분만 잘라서 분류합니다. 스캔을 한 번만 하니 훨씬 빠르죠!

Fast R-CNN의 핵심 개선: **CNN은 전체 이미지에 한 번만 실행**하고, 후보 영역에 해당하는 특징만 꺼내 쓴다.

**Fast R-CNN의 핵심 기술 — RoI Pooling:**

전체 이미지를 CNN에 통과시키면 **특징 맵(Feature Map)**이 나옵니다. 각 후보 영역의 좌표를 이 특징 맵에 매핑하면, 후보 영역에 해당하는 특징만 추출할 수 있어요. 하지만 후보마다 크기가 다르기 때문에, **RoI Pooling**으로 모든 후보를 **고정 크기(예: 7×7)**로 변환합니다.

**Fast R-CNN 처리 과정:**

1. **전체 이미지** → CNN → **특징 맵** (한 번만 실행!)
2. Selective Search로 **후보 영역 추출**
3. 각 후보를 특징 맵에서 **RoI Pooling**으로 고정 크기 특징 추출
4. FC 레이어에서 **분류 + 박스 회귀를 동시에** 수행 (Multi-task Loss)

**개선 성과:**

| 항목 | R-CNN | Fast R-CNN | 개선 |
|------|-------|------------|------|
| 학습 시간 | ~84시간 | ~9.5시간 | **9배** 빨라짐 |
| 추론 시간 (이미지당) | ~47초 | ~0.32초 | **146배** 빨라짐 |
| mAP (VOC 2007) | 58.5% | 66.9% | **8.4%p** 향상 |

하지만 Fast R-CNN에는 여전히 병목이 남아 있었습니다 — **Selective Search**입니다. 후보 영역을 추출하는 이 전통적 알고리즘이 전체 시간의 대부분을 차지했거든요.

### 3. Faster R-CNN (2015) — "후보 추출도 신경망으로!"

> 💡 **비유**: Fast R-CNN까지는 "어디를 볼지"를 정하는 사람(Selective Search)과 "무엇인지"를 판단하는 AI(CNN)가 따로 일했습니다. Faster R-CNN은 **AI가 "어디를 볼지"까지 스스로 결정**하도록 만들었습니다. 사람 없이 AI만으로 전 과정이 돌아가는 거죠!

Faster R-CNN의 핵심 혁신: **Region Proposal Network(RPN)** — 후보 영역 추출까지 신경망으로 대체.

**Anchor Box 개념:**

RPN은 특징 맵의 각 위치에서 **미리 정의된 기준 박스(Anchor)**를 기반으로 객체 유무를 판단합니다.

| 설정 | 값 | 설명 |
|------|------|------|
| **스케일** | 128², 256², 512² | 작은/중간/큰 객체 커버 |
| **종횡비** | 1:1, 1:2, 2:1 | 정사각형, 세로형, 가로형 |
| **위치당 앵커 수** | 3 × 3 = **9개** | 모든 조합 |

특징 맵이 40×60이면, 총 40 × 60 × 9 = **21,600개의 앵커**가 생성됩니다. RPN은 각 앵커에 대해:
- **객체 있음/없음 확률** (2-class classification)
- **박스 보정 값** (4개의 offset: dx, dy, dw, dh)

을 예측합니다.

**Faster R-CNN 전체 구조:**

1. **Backbone** (ResNet 등): 이미지 → 특징 맵 추출
2. **RPN**: 특징 맵 → 후보 영역(Region Proposal) 생성
3. **RoI Pooling/Align**: 후보별 특징을 고정 크기로 추출
4. **Detection Head**: 분류 + 박스 회귀

> ⚠️ **흔한 오해**: "Faster R-CNN은 실시간이다"라고 생각하기 쉽지만, 실제로는 **5~15 FPS** 정도입니다. 이름에 "Faster"가 붙어서 헷갈리지만, 이건 "Fast R-CNN보다 빠르다"는 의미이지 "빠른 모델이다"라는 뜻이 아닙니다. 실시간(30+ FPS)이 필요하면 YOLO 계열을 고려해야 해요.

**최종 성능 비교:**

| 모델 | 후보 추출 | 공유 특징 | End-to-End | 속도 (FPS) | mAP |
|------|-----------|-----------|------------|-----------|-----|
| R-CNN | Selective Search | X | X | ~0.02 | 58.5% |
| Fast R-CNN | Selective Search | O | 부분적 | ~3 | 66.9% |
| **Faster R-CNN** | **RPN (학습)** | **O** | **O** | **~15** | **69.9%** |

## 실습: Faster R-CNN으로 커스텀 객체 탐지기 만들기

torchvision의 사전 학습된 Faster R-CNN을 커스텀 클래스에 맞게 수정하는 코드입니다:

```python
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_faster_rcnn(num_classes, pretrained=True):
    """
    커스텀 클래스 수에 맞춘 Faster R-CNN 모델을 생성합니다.

    Args:
        num_classes: 탐지할 클래스 수 + 1 (배경 포함)
        pretrained: COCO 사전 학습 가중치 사용 여부
    """
    if pretrained:
        # COCO로 사전 학습된 모델 로드
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    else:
        model = fasterrcnn_resnet50_fpn_v2(weights=None)

    # 분류 헤드를 커스텀 클래스 수에 맞게 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# 3개 클래스 탐지기 (배경 포함 4개)
# 예: 고양이, 강아지, 새 → num_classes = 4 (배경 + 3클래스)
model = create_faster_rcnn(num_classes=4)

# 모델 구조 확인
print(f"분류 헤드 출력: {model.roi_heads.box_predictor.cls_score.out_features}개 클래스")
print(f"박스 회귀 출력: {model.roi_heads.box_predictor.bbox_pred.out_features}개 좌표값")
```

학습 루프도 간단합니다. torchvision의 Faster R-CNN은 **학습 모드에서 자동으로 손실(loss)을 계산**해줍니다:

```python
import torch
from torch.optim import SGD

def train_one_epoch(model, data_loader, optimizer, device):
    """Faster R-CNN 1에포크 학습"""
    model.train()
    total_loss = 0

    for images, targets in data_loader:
        # images: 이미지 텐서 리스트
        # targets: [{'boxes': Tensor, 'labels': Tensor}, ...] 리스트
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 학습 모드에서는 loss_dict를 반환
        loss_dict = model(images, targets)
        # loss_dict 예시:
        #   'loss_classifier': 분류 손실
        #   'loss_box_reg': 박스 회귀 손실
        #   'loss_objectness': RPN 객체 유무 손실
        #   'loss_rpn_box_reg': RPN 박스 회귀 손실

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)

# 추론 예시
def detect_objects(model, image, device, threshold=0.5):
    """학습된 모델로 객체 탐지"""
    model.eval()
    with torch.no_grad():
        predictions = model([image.to(device)])

    pred = predictions[0]
    # 신뢰도 필터링
    mask = pred['scores'] > threshold
    return {
        'boxes': pred['boxes'][mask],
        'labels': pred['labels'][mask],
        'scores': pred['scores'][mask]
    }
```

> 🔥 **실무 팁**: Faster R-CNN 학습 시 **배치 크기를 크게 잡기 어렵습니다** (보통 2~4). 이미지마다 크기가 다르고, 후보 영역 수도 다르기 때문입니다. 학습 속도를 높이려면 **Gradient Accumulation**을 활용하세요.

## 더 깊이 알아보기

### Ross Girshick과 R-CNN의 탄생

R-CNN의 창시자 **Ross Girshick**은 UC Berkeley에서 Jitendra Malik 교수의 지도 아래 박사 과정을 밟던 중 R-CNN을 개발했습니다. 당시 2012년 AlexNet이 ImageNet 분류에서 놀라운 성과를 보이며 CNN의 잠재력을 증명한 직후였죠. Girshick은 "분류에서 이렇게 잘하는 CNN이 탐지에서도 먹히지 않을까?"라는 질문에서 출발했습니다.

결과는 대성공이었습니다. R-CNN은 CVPR 2014에서 발표되었고, Pascal VOC에서 기존 최고 성적을 **30% 이상** 갈아치우며 단숨에 주목받았습니다. 논문 제목 *"Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation"*에서 알 수 있듯이, CNN이 추출하는 **풍부한 특징 계층(Rich Feature Hierarchies)**이 핵심이었죠.

이후 Girshick은 Microsoft Research로 옮겨 Fast R-CNN을 단독 저자로 발표하고, Shaoqing Ren, Kaiming He 등과 함께 Faster R-CNN을 완성했습니다. 그리고 2017년에는 Mask R-CNN(인스턴스 분할)까지 발전시켰죠. 한 사람이 객체 탐지의 핵심 논문 4편을 연달아 주도한 셈입니다.

### Feature Pyramid Network (FPN) — 다양한 크기의 객체 잡기

Faster R-CNN의 약점 중 하나는 **작은 객체 탐지**였습니다. 이를 해결한 것이 **FPN(Feature Pyramid Network)**입니다. CNN의 여러 레이어에서 나오는 다양한 스케일의 특징 맵을 **피라미드 형태로 결합**하여, 작은 객체는 고해상도 특징에서, 큰 객체는 저해상도 특징에서 탐지할 수 있게 합니다.

torchvision의 `fasterrcnn_resnet50_fpn`에서 **fpn**이 바로 이 Feature Pyramid Network를 의미합니다. FPN이 없는 Faster R-CNN보다 특히 작은 객체에서 성능이 크게 향상됩니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "R-CNN 계열은 구식이라 안 쓴다"고 생각할 수 있지만, Faster R-CNN은 **2025년 현재에도** 많은 벤치마크의 기준 모델입니다. 특히 COCO 데이터셋에서 새로운 탐지기의 성능을 비교할 때 거의 항상 Faster R-CNN이 비교 대상에 포함됩니다.

> 💡 **알고 계셨나요?**: Faster R-CNN 논문(Ren et al., 2015)은 Google Scholar 기준 **인용 수 5만 회 이상**으로, 컴퓨터 비전 분야에서 가장 많이 인용된 논문 중 하나입니다. 이 논문 하나가 현대 객체 탐지의 기반을 놓았다고 해도 과언이 아닙니다.

> 🔥 **실무 팁**: 커스텀 데이터로 Faster R-CNN을 학습할 때, `num_classes`에 **배경 클래스**를 반드시 포함해야 합니다! 예를 들어 "고양이"와 "강아지" 두 클래스를 탐지하려면 `num_classes=3` (배경 + 고양이 + 강아지)으로 설정하세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **R-CNN** | CNN을 객체 탐지에 최초 적용. 2,000개 후보에 각각 CNN 실행 (느림) |
| **Fast R-CNN** | CNN 1회 실행 + RoI Pooling. 속도 146배 향상 |
| **Faster R-CNN** | RPN으로 후보 추출도 학습. 완전한 End-to-End |
| **RoI Pooling** | 다양한 크기의 후보를 고정 크기 특징으로 변환 |
| **RPN** | 앵커 기반으로 객체 후보를 신경망이 직접 생성 |
| **Anchor Box** | 위치당 9개 (3 스케일 × 3 종횡비)의 기준 박스 |

## 다음 섹션 미리보기

R-CNN 계열은 정확하지만 실시간 처리에는 한계가 있었습니다. 다음 섹션에서는 **"실시간 객체 탐지"**의 대명사 [YOLO 시리즈](./03-yolo.md)를 살펴봅니다. "You Only Look Once" — 한 번만 보면 된다는 이름답게, 이미지를 **단 한 번** 통과시켜 모든 객체를 탐지하는 혁신적 접근을 배워봅시다.

## 참고 자료

- [Rich Feature Hierarchies for Accurate Object Detection (R-CNN, Girshick et al., 2014)](https://arxiv.org/abs/1311.2524) - R-CNN 원논문, 딥러닝 기반 객체 탐지의 시작
- [Fast R-CNN (Girshick, 2015)](https://arxiv.org/abs/1504.08083) - RoI Pooling과 멀티태스크 학습 도입
- [Faster R-CNN (Ren et al., 2015)](https://arxiv.org/abs/1506.01497) - RPN 도입으로 End-to-End 객체 탐지 완성
- [torchvision Faster R-CNN 공식 문서](https://pytorch.org/vision/stable/models/faster_rcnn.html) - PyTorch에서 Faster R-CNN 사용 가이드
- [Lil'Log: Object Detection Part 3 — R-CNN Family](https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/) - R-CNN 계열을 체계적으로 정리한 블로그
- [Faster R-CNN: Still the Benchmark in 2025](https://www.thinkautonomous.ai/blog/faster-rcnn/) - 2025년 관점에서 본 Faster R-CNN의 가치
