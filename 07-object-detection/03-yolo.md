# YOLO 시리즈

> YOLOv5부터 YOLOv11까지

## 개요

**"You Only Look Once"** — 한 번만 보면 된다. 이 이름 자체가 YOLO의 핵심 아이디어를 담고 있습니다. [R-CNN 계열](./02-rcnn-family.md)이 "먼저 후보를 찾고 → 각각 분류"하는 2단계 접근이었다면, YOLO는 이미지를 **단 한 번** 신경망에 통과시켜 모든 객체의 위치와 클래스를 **동시에** 예측합니다. 이 단순하면서도 강력한 아이디어 덕분에, YOLO는 **실시간 객체 탐지의 대명사**가 되었습니다.

**선수 지식**: [객체 탐지 기초](./01-detection-basics.md), [R-CNN 계열](./02-rcnn-family.md)
**학습 목표**:
- YOLO의 핵심 아이디어(그리드 기반 1-stage 탐지)를 이해한다
- YOLOv1부터 YOLOv11까지의 진화 과정과 주요 개선점을 안다
- Ultralytics 라이브러리로 YOLO 모델을 사용할 수 있다
- 실시간 탐지와 정확도 간의 trade-off를 이해한다

## 왜 알아야 할까?

Faster R-CNN이 ~15 FPS였던 반면, YOLOv1은 **45 FPS**, Fast YOLO는 무려 **155 FPS**를 달성했습니다. 이는 자율주행, 드론, 로봇, 실시간 영상 분석 등 **속도가 생명인** 응용 분야의 문을 열어젖혔죠.

2024년 기준으로 YOLO는 GitHub에서 가장 많이 사용되는 객체 탐지 프레임워크이며, Ultralytics의 `ultralytics` 패키지는 **누적 다운로드 수 1억 회 이상**을 기록했습니다. 산업계에서 객체 탐지가 필요하면, 대부분 **"일단 YOLO부터"** 시작합니다.

## 핵심 개념

### 1. YOLO의 핵심 아이디어 — 탐지를 회귀 문제로 바꾸기

> 💡 **비유**: 교실에서 선생님이 학생들에게 "이 사진에서 동물을 찾아보세요"라고 했다고 합시다. R-CNN 방식은 학생이 **돋보기로 구석구석 2,000군데를 살피는** 것이고, YOLO 방식은 사진을 **격자로 나눈 뒤 각 칸을 담당하는 학생이 동시에 자기 구역을 보는** 것입니다. 30명이 동시에 보니 당연히 빠르죠!

**YOLOv1의 동작 원리:**

1. 이미지를 **S × S 그리드**로 나눔 (기본 7×7)
2. 각 그리드 셀이 **B개의 바운딩 박스**와 **C개 클래스 확률**을 예측
3. 각 바운딩 박스는 **(x, y, w, h, confidence)** 5개 값을 가짐
4. 최종 출력: **S × S × (B × 5 + C)** 텐서 — 한 번에 예측!

예를 들어, 7×7 그리드에 2개 박스, 20개 클래스(Pascal VOC)라면:
- 출력 크기: 7 × 7 × (2 × 5 + 20) = 7 × 7 × 30

이 전체 과정이 **CNN 한 번의 순전파(forward pass)**로 끝납니다. 별도의 후보 영역 추출이 없으니, 속도가 비약적으로 빨라진 것이죠.

### 2. YOLO 버전별 진화 — 10년의 여정

YOLO의 버전 히스토리를 정리하면:

| 버전 | 연도 | 주요 저자 | 핵심 개선 |
|------|------|-----------|-----------|
| **v1** | 2016 | Joseph Redmon | 1-stage 탐지의 시작. 45 FPS |
| **v2** | 2016 | Redmon, Farhadi | Batch Norm, 앵커 박스 도입, Darknet-19 |
| **v3** | 2018 | Redmon, Farhadi | Darknet-53, 멀티스케일 예측 (FPN 유사) |
| **v4** | 2020 | Bochkovskiy et al. | CSPDarknet, Mish 활성화, Mosaic 증강 |
| **v5** | 2020 | Ultralytics | PyTorch 전환, 편의성 극대화, 자동 앵커 |
| **v6** | 2022 | Meituan | 경량 모델 + 양자화/증류에 집중 |
| **v7** | 2022 | Wang et al. | E-ELAN 구조, 보조 헤드 학습 |
| **v8** | 2023 | Ultralytics | **Anchor-Free** 전환, C2f 모듈, 디커플 헤드 |
| **v9** | 2024 | Wang et al. | PGI(프로그래밍 가능 그래디언트), GELAN |
| **v10** | 2024 | Tsinghua Univ. | NMS-Free, 일관된 듀얼 할당 |
| **v11** | 2024 | Ultralytics | C3k2 + C2PSA 어텐션, 22% 파라미터 감소 |

> 💡 **알고 계셨나요?**: YOLO 버전 번호가 순서대로 나온 게 아닙니다! v6보다 v7이 먼저 나왔고, v5는 논문 없이 코드만 공개되었으며, v4의 저자는 원 창시자와 다른 사람입니다. 한 마디로 YOLO는 **오픈소스 커뮤니티가 함께 만들어가는** 프로젝트입니다.

### 3. YOLOv8 — 현대 YOLO의 기준점

YOLOv8은 YOLO 시리즈에서 가장 큰 패러다임 전환을 이뤘습니다:

**핵심 변화들:**

- **Anchor-Free**: 이전 버전의 앵커 박스를 제거하고, 각 위치에서 직접 박스를 예측
- **C2f 모듈**: YOLOv5의 C3 모듈을 개선한 새 블록으로 더 풍부한 특징 추출
- **디커플 헤드(Decoupled Head)**: 분류와 박스 회귀를 **별도 헤드**로 분리하여 성능 향상
- **다중 태스크 지원**: 탐지, 분할, 포즈 추정, 분류를 하나의 프레임워크에서

**모델 사이즈별 변형:**

| 모델 | 파라미터 수 | mAP@0.5:0.95 | 속도 (FPS) | 용도 |
|------|------------|-------------|-----------|------|
| YOLOv8n | 3.2M | 37.3 | 가장 빠름 | 엣지/모바일 |
| YOLOv8s | 11.2M | 44.9 | 빠름 | 균형 |
| YOLOv8m | 25.9M | 50.2 | 보통 | 일반 서버 |
| YOLOv8l | 43.7M | 52.9 | 느림 | 고성능 서버 |
| YOLOv8x | 68.2M | 53.9 | 가장 느림 | 최고 정확도 |

### 4. YOLO11 — 최신 세대 (2024)

YOLO11은 YOLOv8 대비 효율성에 집중합니다:

- **C3k2 블록**: C2f를 대체하는 더 경량화된 블록
- **C2PSA (Partial Self-Attention)**: 공간 어텐션을 추가해 특징 선택력 향상
- **22% 파라미터 감소**: 같은 정확도에서 더 가벼운 모델
- **SPPF 유지**: 다양한 수용 영역(receptive field)을 위한 공간 피라미드 풀링

## 실습: Ultralytics로 YOLO 사용하기

YOLO의 가장 큰 장점 중 하나는 **사용이 매우 쉽다**는 것입니다. Ultralytics 라이브러리 덕분에 몇 줄 코드로 탐지가 가능합니다:

```python
# Ultralytics 설치: pip install ultralytics
from ultralytics import YOLO

# 사전 학습된 YOLO11 모델 로드 (자동 다운로드)
model = YOLO("yolo11n.pt")  # nano 모델 (가장 가벼움)

# 이미지에서 객체 탐지
results = model("bus.jpg")

# 결과 확인
for result in results:
    boxes = result.boxes  # 바운딩 박스
    print(f"탐지된 객체 수: {len(boxes)}")

    for box in boxes:
        # 좌표, 신뢰도, 클래스
        xyxy = box.xyxy[0].tolist()      # [x1, y1, x2, y2]
        conf = box.conf[0].item()         # 신뢰도
        cls = int(box.cls[0].item())      # 클래스 ID
        name = model.names[cls]           # 클래스 이름
        print(f"  {name}: {conf:.2f} @ [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
```

커스텀 데이터셋으로 학습하는 것도 간단합니다:

```python
from ultralytics import YOLO

# 사전 학습된 모델에서 파인 튜닝
model = YOLO("yolo11n.pt")

# 커스텀 데이터셋으로 학습
# data.yaml에 클래스 정보와 이미지 경로 정의
results = model.train(
    data="data.yaml",       # 데이터셋 설정 파일
    epochs=100,             # 학습 에포크
    imgsz=640,              # 입력 이미지 크기
    batch=16,               # 배치 크기
    device=0,               # GPU 0번 사용
    patience=20,            # 조기 종료 인내심
    lr0=0.01,               # 초기 학습률
)

# 검증 세트에서 평가
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
```

data.yaml 파일의 형식은 다음과 같습니다:

```yaml
# data.yaml 예시
path: /path/to/dataset     # 데이터셋 루트 경로
train: images/train         # 학습 이미지 경로 (path 기준 상대)
val: images/val             # 검증 이미지 경로

# 클래스 정의
names:
  0: cat
  1: dog
  2: bird
```

모델을 다양한 형식으로 내보내는 것도 한 줄이면 됩니다:

```python
# ONNX로 내보내기 (웹/모바일 배포용)
model.export(format="onnx")

# TensorRT로 내보내기 (NVIDIA GPU 최적화)
model.export(format="engine")

# CoreML로 내보내기 (iOS 배포용)
model.export(format="coreml")
```

> 🔥 **실무 팁**: YOLO 모델 선택 가이드 — 엣지 디바이스(라즈베리 파이, 모바일)에서는 **nano(n)**, 일반 GPU 서버에서는 **small(s) 또는 medium(m)**, 최고 정확도가 필요하면 **large(l) 또는 extra(x)**를 선택하세요. 대부분의 실무 프로젝트에서는 **small**이 가성비가 가장 좋습니다.

## 더 깊이 알아보기

### Joseph Redmon의 결단 — AI 윤리와 연구자의 책임

YOLO의 창시자 Joseph Redmon은 워싱턴 대학교에서 Ali Farhadi 교수의 지도 아래 박사 과정 중 YOLOv1을 개발했습니다. 그는 2016년 CVPR에서 OpenCV People's Choice Award를 수상하고, 2017년에는 YOLOv2로 Best Paper Honorable Mention까지 받으며 학계의 스타로 떠올랐습니다.

하지만 2020년 2월, Redmon은 트위터에 충격적인 선언을 합니다: **"컴퓨터 비전 연구를 그만두겠다."** 그 이유는 자신의 연구가 **군사 무기와 감시 시스템에 악용**되고 있다는 것이었습니다.

> "과학은 비정치적이고 연구는 주제와 무관하게 도덕적으로 선하다고 믿었던 적이 있었는데, 그것이 부끄럽습니다."

Redmon의 은퇴 이후, YOLO의 개발은 다양한 팀이 이어받았습니다. Alexey Bochkovskiy가 v4를, Ultralytics의 Glenn Jocher가 v5와 v8을, Chien-Yao Wang 등이 v7과 v9을 주도했죠. **한 사람이 시작한 프로젝트가 커뮤니티의 힘으로 계속 진화하는** 오픈소스의 아름다운 모습입니다.

이 이야기는 AI 연구자에게 중요한 질문을 던집니다 — 기술의 발전과 윤리적 책임은 어떻게 균형을 맞출 수 있을까요?

### YOLO의 학습 전략 — Mosaic 증강과 Multi-Scale Training

YOLOv4부터 도입된 **Mosaic 데이터 증강**은 4장의 이미지를 한 장에 합치는 기법입니다. 이렇게 하면:
- 배치 크기를 키우는 효과 (4배의 이미지 정보)
- 배경 다양성 증가
- 작은 객체 학습에 유리

또한 **Multi-Scale Training**은 학습 중 입력 이미지 크기를 랜덤하게 변경합니다 (예: 320~640 사이). 이를 통해 모델이 다양한 스케일의 객체에 강건해집니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "YOLO 버전이 높으면 무조건 좋다"고 생각하기 쉽지만, 반드시 그렇지는 않습니다. YOLOv8이 v5보다 정확하지만, v5가 더 가볍고 배포 안정성이 검증되어 있어 **프로덕션 환경에서 v5를 여전히 선호**하는 경우도 많습니다. 최신 버전보다 **본인 태스크에 맞는 버전**을 선택하세요.

> 💡 **알고 계셨나요?**: YOLO의 공식 웹사이트(pjreddie.com)에서 Redmon은 자신의 이력서에 "다른 사람의 컴퓨터에 신경망을 몰래 설치하는 사람"이라고 유머러스하게 적어두었고, Darknet(YOLO의 C 프레임워크) README에는 "나이트클럽 사진에서 물체를 탐지하는" 예시를 넣는 등 특유의 유머를 보여줬습니다.

> 🔥 **실무 팁**: Ultralytics YOLO로 학습할 때, `imgsz=640`이 기본값인데, 작은 객체가 많은 데이터셋에서는 `imgsz=1280`으로 올리면 AP_S(작은 객체 정확도)가 크게 향상됩니다. 대신 GPU 메모리를 4배 더 사용하니 배치 크기를 줄여야 합니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **1-Stage 탐지** | 후보 추출 없이 한 번에 분류 + 위치 예측 |
| **그리드 분할** | 이미지를 S×S로 나누고 각 셀이 객체 담당 |
| **YOLOv8** | Anchor-Free 전환, C2f 모듈, 디커플 헤드 |
| **YOLO11** | C3k2 + C2PSA 어텐션, 22% 파라미터 감소 |
| **Ultralytics** | YOLO 공식 프레임워크, 3줄로 탐지 가능 |
| **모델 사이즈** | n(엣지) → s(균형) → m(서버) → l/x(최고 정확도) |

## 다음 섹션 미리보기

YOLO는 앵커 박스(v1~v7)에서 Anchor-Free(v8~)로 진화했습니다. 다음 섹션에서는 이 **앵커 없는 탐지**를 처음부터 설계한 [Anchor-Free 탐지기](./04-anchor-free.md) — FCOS, CenterNet, CornerNet을 살펴봅니다. "왜 앵커가 문제였는지", 그리고 "앵커 없이 어떻게 객체를 찾는지"를 깊이 이해해봅시다.

## 참고 자료

- [You Only Look Once (YOLOv1, Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) - YOLO의 시작, 원 논문
- [Ultralytics YOLO 공식 문서](https://docs.ultralytics.com/) - YOLO11/v8 사용법 공식 가이드
- [YOLO Evolution: Transforming Object Detection 2015-2024 (viso.ai)](https://viso.ai/computer-vision/yolo-explained/) - YOLO 전체 진화 타임라인
- [The Ultimate Guide to YOLO Models (Roboflow, 2026)](https://blog.roboflow.com/guide-to-yolo-models/) - 모든 YOLO 버전의 종합 가이드
- [LearnOpenCV — YOLO11 Tutorial](https://learnopencv.com/yolo11/) - YOLO11 아키텍처와 실습 튜토리얼
- [YOLOv1 to YOLOv11: A Comprehensive Survey](https://arxiv.org/html/2508.02067) - 최신 서베이 논문
