# 에지 검출

> Sobel, Canny, Laplacian 에지 검출

## 개요

이미지에서 **에지(Edge)**란 밝기가 급격하게 변하는 경계를 말합니다. 물체의 윤곽선, 그림자의 경계, 텍스트의 획 — 이 모든 것이 에지입니다. 에지를 잘 찾아내면 이미지에서 "무엇이 어디에 있는지"를 파악하는 첫 단추를 끼울 수 있습니다.

**선수 지식**: [필터와 커널](./02-filters-kernels.md) — 커널, 합성곱 연산의 이해
**학습 목표**:
- 에지가 수학적으로 무엇을 의미하는지(미분) 이해한다
- Sobel, Laplacian, Canny 에지 검출기의 차이를 설명할 수 있다
- OpenCV로 에지 검출을 적용하고 결과를 비교할 수 있다

## 왜 알아야 할까?

에지 검출은 **객체 탐지, 이미지 분할, 도형 인식, OCR(글자 인식)** 등 수많은 CV 작업의 전처리 단계입니다. 딥러닝 시대에도 Canny 에지는 **ControlNet** 같은 생성 AI의 입력 조건으로 활발히 사용됩니다.

## 핵심 개념

### 1. 에지란 무엇인가?

> 💡 **비유**: 하얀 종이 위에 검은 글자가 있다고 상상하세요. 종이 부분은 밝기가 균일하고, 글자와 만나는 경계에서 밝기가 **급격히 변합니다.** 이 급격한 변화 지점이 바로 에지입니다.

수학적으로 에지는 이미지 밝기의 **기울기(Gradient)**가 큰 곳입니다. 기울기를 구하려면 **미분**이 필요합니다.

> **밝기가 일정한 영역** → 기울기 ≈ 0 → 에지 아님
> **밝기가 급변하는 영역** → 기울기 큼 → 에지!

### 2. Sobel 에지 검출 — 방향별 기울기

> 💡 **비유**: Sobel은 이미지를 두 번 조사하는 **탐정**입니다. 한 번은 **가로 방향**으로 밝기 변화를 확인하고, 한 번은 **세로 방향**으로 확인합니다. 두 결과를 합치면 모든 방향의 에지를 찾을 수 있습니다.

Sobel 필터는 이미지의 1차 미분(기울기)을 근사합니다.

**Sobel 커널:**

수평 방향(Gx) — 좌우 밝기 차이 검출:

| | 열1 | 열2 | 열3 |
|---|:---:|:---:|:---:|
| **행1** | -1 | 0 | 1 |
| **행2** | -2 | 0 | 2 |
| **행3** | -1 | 0 | 1 |

수직 방향(Gy) — 상하 밝기 차이 검출:

| | 열1 | 열2 | 열3 |
|---|:---:|:---:|:---:|
| **행1** | -1 | -2 | -1 |
| **행2** | 0 | 0 | 0 |
| **행3** | 1 | 2 | 1 |

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

# 수평 에지 (좌우 밝기 변화)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# 수직 에지 (상하 밝기 변화)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 두 방향 합치기 (에지 크기)
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

print(f"수평 에지: min={sobel_x.min():.0f}, max={sobel_x.max():.0f}")
print(f"수직 에지: min={sobel_y.min():.0f}, max={sobel_y.max():.0f}")
```

### 3. Laplacian 에지 검출 — 한 번에 모든 방향

> 💡 **비유**: Sobel이 가로·세로를 따로 조사하는 탐정이라면, Laplacian은 **한 번에 사방을 둘러보는 탐정**입니다. 모든 방향의 밝기 변화를 한꺼번에 잡아냅니다.

Laplacian은 이미지의 **2차 미분**을 계산합니다. 에지에서 2차 미분은 0을 지나가는데, 이 **영교차(Zero-Crossing)** 지점이 에지 위치입니다.

```python
import cv2

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

# 노이즈 제거를 위해 블러 먼저 적용
blurred = cv2.GaussianBlur(img, (3, 3), 0)

# Laplacian 적용
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian = np.uint8(np.abs(laplacian))

print(f"Laplacian 결과: shape={laplacian.shape}")
```

> ⚠️ **주의**: Laplacian은 노이즈에 민감합니다. 반드시 **가우시안 블러를 먼저** 적용하세요.

### 4. Canny 에지 검출 — 가장 인기 있는 방법

> 💡 **비유**: Canny는 에지 검출의 **올인원 패키지**입니다. 노이즈 제거 → 기울기 계산 → 얇은 선으로 정리 → 진짜 에지만 남기기까지 4단계를 자동으로 처리합니다.

Canny 알고리즘의 4단계:

1. **가우시안 블러**: 노이즈 제거
2. **기울기 계산**: Sobel 필터로 방향과 크기 구함
3. **비최대 억제(NMS)**: 에지를 1픽셀 두께로 가늘게 만듦
4. **이중 임계값**: 강한 에지는 유지, 약한 에지는 강한 에지와 연결된 것만 남김

```python
import cv2

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

# Canny 에지 검출
# threshold1: 하한 임계값, threshold2: 상한 임계값
edges = cv2.Canny(img, 100, 200)

print(f"에지 맵: shape={edges.shape}, dtype={edges.dtype}")
print(f"에지 픽셀 비율: {(edges > 0).sum() / edges.size * 100:.1f}%")
```

**임계값에 따른 변화:**

| 임계값 조합 | 효과 |
|------------|------|
| 낮은 값 (50, 100) | 에지를 많이 검출 (노이즈 포함 가능) |
| 중간 값 (100, 200) | 균형 잡힌 결과 (일반적 권장) |
| 높은 값 (200, 300) | 강한 에지만 검출 (세부 정보 손실) |

```python
import cv2

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

# 임계값별 비교
for low, high in [(50, 100), (100, 200), (200, 300)]:
    edges = cv2.Canny(img, low, high)
    edge_ratio = (edges > 0).sum() / edges.size * 100
    print(f"임계값 ({low:3d}, {high:3d}): 에지 비율 = {edge_ratio:.1f}%")
```

### 5. 세 가지 방법 비교

| 방법 | 원리 | 장점 | 단점 |
|------|------|------|------|
| **Sobel** | 1차 미분 (기울기) | 방향별 에지 분리 가능 | 두꺼운 에지, 파라미터 조절 필요 |
| **Laplacian** | 2차 미분 | 모든 방향 한번에 검출 | 노이즈에 매우 민감 |
| **Canny** | 멀티스테이지 | 가늘고 깨끗한 에지, 가장 실용적 | 임계값 두 개 설정 필요 |

> **실무 추천**: 대부분의 경우 **Canny**를 먼저 사용하세요. 방향별 에지 정보가 필요할 때만 Sobel을 쓰면 됩니다.

## 실습: 직접 해보기

### 세 가지 에지 검출 한눈에 비교

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 세 가지 에지 검출 적용
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.uint8(np.clip(np.sqrt(sobel_x**2 + sobel_y**2), 0, 255))

laplacian = np.uint8(np.abs(cv2.Laplacian(blurred, cv2.CV_64F)))

canny = cv2.Canny(blurred, 100, 200)

# 결과 통계 비교
for name, result in [("Sobel", sobel), ("Laplacian", laplacian), ("Canny", canny)]:
    edge_pixels = (result > 0).sum()
    print(f"{name:10s}: 에지 픽셀 {edge_pixels:>7,}개 ({edge_pixels/result.size*100:.1f}%)")
```

## 더 깊이 알아보기

> 💡 **알고 계셨나요?**: **Sobel 연산자**에는 흥미로운 뒷이야기가 있습니다. Irwin Sobel은 1968년 스탠퍼드 AI 연구소(Stanford AI Lab)에서 발표한 짧은 강연에서 이 연산자를 소개했는데, **정식 논문을 출판하지 않았습니다.** 강연을 들은 동료 연구자들이 입소문으로 퍼뜨려서 널리 알려지게 된 거죠. 수십 년간 컴퓨터 비전의 기본 도구로 사용되면서도 정식 출판물이 없었던 셈이에요. 나중에야 Sobel 본인이 회고록 형식의 글을 남겼을 정도입니다.

반면 **Canny 에지 검출기**는 완전히 다른 이야기입니다. John Canny가 **MIT 석사 논문**(1983~1986)으로 개발한 이 알고리즘은, "좋은 에지 검출기가 만족해야 할 세 가지 조건"을 수학적으로 정의하고 이를 최적화한 것이었거든요.

Canny가 제시한 세 가지 조건:
1. **좋은 검출(Good Detection)**: 진짜 에지를 놓치지 말 것
2. **좋은 위치(Good Localization)**: 검출된 에지가 실제 위치에 가까울 것
3. **하나의 응답(Single Response)**: 하나의 에지에 하나의 결과만 나올 것

이 논문은 현재까지 **48,000회 이상 인용**되어, 컴퓨터 과학 역사상 가장 많이 인용된 논문 중 하나로 꼽힙니다. 40년이 지난 지금도 Canny 에지가 표준으로 사용되는 이유는 바로 이 탄탄한 수학적 근거 덕분이죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Canny가 Sobel보다 무조건 좋다" — 사실 **용도가 다릅니다.** Sobel은 에지의 **기울기 방향과 크기** 정보를 제공하는 반면, Canny는 **이진 에지 맵**(에지다/아니다)만 출력합니다. 기울기 방향이 필요한 작업(HOG 특징 추출, 광학 흐름 등)에서는 Sobel이 오히려 더 적합해요. "어떤 것이 더 좋은가"가 아니라 "무엇이 필요한가"를 먼저 생각하세요.

> 💡 **알고 계셨나요?**: Canny 알고리즘 내부에서 **Sobel을 사용합니다!** Canny의 4단계를 다시 보면 — 가우시안 블러 → **Sobel로 기울기 계산** → 비최대 억제 → 이중 임계값 — 두 번째 단계에서 Sobel 필터가 동작하고 있거든요. 즉 Canny는 Sobel의 "대체품"이 아니라, Sobel 위에 후처리를 더한 "확장판"인 셈이죠.

> 🔥 **실무 팁**: Canny의 두 임계값(low, high) 설정이 어렵다면, **비율 1:2 또는 1:3**을 경험적 시작점으로 사용해 보세요. 예를 들어 50:150, 100:200, 100:300 같은 조합입니다. 또한 이미지의 중간값(median)을 기준으로 자동 설정하는 방법도 있습니다: `low = 0.66 * median`, `high = 1.33 * median`. 매번 수동으로 조절하는 것보다 훨씬 편리하거든요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **에지** | 이미지에서 밝기가 급격히 변하는 경계 |
| **기울기(Gradient)** | 밝기 변화의 방향과 크기 (1차 미분) |
| **Sobel** | 가로/세로 방향별 1차 미분 커널 |
| **Laplacian** | 모든 방향 2차 미분, 노이즈에 민감 |
| **Canny** | 4단계 자동 처리, 가장 깨끗한 에지 검출 |
| **임계값** | Canny에서 에지 강도 기준을 결정하는 두 값 |

## 다음 섹션 미리보기

에지가 이미지의 "경계"를 찾는 것이라면, **특징점(Feature)**은 이미지에서 "특별한 점"을 찾는 것입니다. 다음 섹션 **[특징점 검출](./04-feature-detection.md)**에서는 SIFT, ORB 등 이미지 매칭과 인식에 쓰이는 핵심 기법을 배웁니다.

## 참고 자료

- [OpenCV 공식 블로그 - Edge Detection Using OpenCV](https://opencv.org/blog/edge-detection-using-opencv/) - Sobel, Canny, Laplacian 비교 가이드
- [OpenCV 공식 튜토리얼 - Canny Edge Detection](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html) - Canny 알고리즘 4단계 설명
- [DEV Community - Edge Detection Step-by-Step Guide](https://dev.to/chidoziemanagwu/implementing-edge-detection-with-python-and-opencv-a-step-by-step-guide-57ab) - 초보자용 단계별 구현 가이드
- [GeeksforGeeks - Real-Time Edge Detection](https://www.geeksforgeeks.org/python/real-time-edge-detection-using-opencv-python/) - 실시간 에지 검출 예제
