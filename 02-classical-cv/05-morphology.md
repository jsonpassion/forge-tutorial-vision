# 형태학적 연산

> 침식, 팽창, 열기, 닫기 연산

## 개요

이진 이미지(흑백)에서 물체의 **형태를 다듬는** 연산입니다. 노이즈 제거, 물체 분리, 구멍 메우기 등 이미지 전처리에 매우 자주 사용됩니다. 이름은 어렵지만 원리는 단순합니다.

**선수 지식**: [필터와 커널](./02-filters-kernels.md) — 커널 개념
**학습 목표**:
- 침식, 팽창의 원리와 효과를 이해한다
- 열기(Opening)와 닫기(Closing)의 용도를 구분할 수 있다
- 실제 이미지 전처리에 형태학 연산을 적용할 수 있다

## 왜 알아야 할까?

에지 검출이나 이진화 후에 결과가 깔끔하지 않은 경우가 대부분입니다. 작은 잡 노이즈, 물체 내부의 구멍, 붙어있는 물체를 분리하는 등 **후처리 정리** 작업에 형태학 연산은 필수입니다. 의료 영상 분석, OCR, 산업 검사 등에서 핵심적으로 활용됩니다.

## 핵심 개념

### 1. 구조 요소(Structuring Element)

> 💡 **비유**: 형태학 연산에서 커널은 **도장**에 해당합니다. 이 도장을 이미지 위에 찍으면서 주변 픽셀을 확인하고, 규칙에 따라 해당 픽셀을 유지하거나 변경합니다.

```python
import cv2
import numpy as np

# 다양한 모양의 구조 요소 생성
rect   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))    # 사각형
ellip  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # 타원형
cross  = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))   # 십자형

# 직접 만들 수도 있음
custom = np.ones((3, 3), dtype=np.uint8)

print("사각형 커널:")
print(rect)
```

### 2. 침식(Erosion) — 깎아내기

> 💡 **비유**: 해변의 파도가 모래성을 조금씩 **깎아내는** 것과 같습니다. 물체의 가장자리가 안쪽으로 줄어들고, 작은 돌출부는 사라집니다.

커널 아래의 **모든 픽셀이 흰색(1)일 때만** 중심 픽셀을 흰색으로 유지합니다. 하나라도 검정이면 검정으로 바꿉니다.

**침식의 효과:**
- 물체의 경계가 **안쪽으로 줄어듦**
- 작은 흰색 노이즈가 **제거됨**
- 가느다란 연결 부분이 **끊어짐** (물체 분리에 유용)

```python
import cv2
import numpy as np

img = cv2.imread("binary_image.png", cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5, 5), dtype=np.uint8)

# 침식 적용
eroded = cv2.erode(img, kernel, iterations=1)

# 반복 횟수를 늘리면 더 많이 깎임
eroded_2x = cv2.erode(img, kernel, iterations=2)
eroded_3x = cv2.erode(img, kernel, iterations=3)

print(f"원본 흰색 픽셀: {(img > 0).sum()}")
print(f"1회 침식 후: {(eroded > 0).sum()}")
print(f"3회 침식 후: {(eroded_3x > 0).sum()}")
```

### 3. 팽창(Dilation) — 부풀리기

> 💡 **비유**: 빵 반죽이 **발효되면서 부풀어 오르는** 것과 같습니다. 물체의 가장자리가 바깥으로 확장되고, 작은 구멍은 메워집니다.

커널 아래의 픽셀 중 **하나라도 흰색(1)이면** 중심 픽셀을 흰색으로 만듭니다. 침식의 정반대입니다.

**팽창의 효과:**
- 물체의 경계가 **바깥으로 확장됨**
- 작은 검정 구멍이 **메워짐**
- 끊어진 부분이 **연결됨**

```python
import cv2
import numpy as np

img = cv2.imread("binary_image.png", cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5, 5), dtype=np.uint8)

# 팽창 적용
dilated = cv2.dilate(img, kernel, iterations=1)

print(f"원본 흰색 픽셀: {(img > 0).sum()}")
print(f"팽창 후: {(dilated > 0).sum()}")
```

### 4. 열기(Opening) — 노이즈 제거

> 💡 **비유**: 먼저 **사포로 표면을 갈아낸 뒤** (침식), 다시 **페인트를 칠하는** (팽창) 것입니다. 작은 돌기(노이즈)는 사라지지만, 큰 물체는 거의 원래 크기로 돌아옵니다.

**열기 = 침식 → 팽창** (순서가 핵심!)

```python
import cv2
import numpy as np

img = cv2.imread("noisy_binary.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), dtype=np.uint8)

# 열기 (노이즈 제거)
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

print("열기: 작은 흰색 노이즈 점이 제거됩니다")
```

### 5. 닫기(Closing) — 구멍 메우기

> 💡 **비유**: 먼저 **퍼티로 구멍을 메운 뒤** (팽창), **여분을 깎아내는** (침식) 것입니다. 물체 내부의 작은 구멍은 메워지지만, 전체 크기는 유지됩니다.

**닫기 = 팽창 → 침식** (열기의 반대 순서)

```python
import cv2
import numpy as np

img = cv2.imread("holes_binary.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), dtype=np.uint8)

# 닫기 (구멍 메우기)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

print("닫기: 물체 내부의 작은 검정 구멍이 메워집니다")
```

### 6. 네 가지 연산 비교

| 연산 | 순서 | 효과 | 주요 용도 |
|------|------|------|----------|
| **침식** | 깎기 | 물체 축소, 노이즈 제거 | 물체 분리, 얇은 연결 끊기 |
| **팽창** | 부풀리기 | 물체 확장, 구멍 메움 | 끊어진 부분 연결 |
| **열기** | 침식 → 팽창 | 작은 노이즈 제거 + 크기 유지 | 배경 노이즈 정리 |
| **닫기** | 팽창 → 침식 | 작은 구멍 메움 + 크기 유지 | 물체 내부 정리 |

### 7. 추가 연산

```python
import cv2
import numpy as np

img = cv2.imread("binary_image.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), dtype=np.uint8)

# 그래디언트: 팽창 - 침식 = 물체의 윤곽선만 남음
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Top Hat: 원본 - 열기 = 밝은 작은 점 추출
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# Black Hat: 닫기 - 원본 = 어두운 작은 구멍 추출
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
```

| 연산 | 공식 | 추출하는 것 |
|------|------|-----------|
| **그래디언트** | 팽창 − 침식 | 물체의 **윤곽선** |
| **Top Hat** | 원본 − 열기 | 배경보다 **밝은** 작은 요소 |
| **Black Hat** | 닫기 − 원본 | 배경보다 **어두운** 작은 구멍 |

## 실습: 직접 해보기

### 이진화 + 형태학 연산 파이프라인

```python
import cv2
import numpy as np

# 이미지 읽기 & 흑백 변환
img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

# 이진화 (임계값 기준으로 흑백 분리)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 노이즈 제거 파이프라인
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 1단계: 열기로 작은 노이즈 제거
step1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 2단계: 닫기로 내부 구멍 메우기
step2 = cv2.morphologyEx(step1, cv2.MORPH_CLOSE, kernel)

print(f"원본 흰색 픽셀: {(binary > 0).sum():,}")
print(f"열기 후: {(step1 > 0).sum():,}")
print(f"닫기 후: {(step2 > 0).sum():,}")
```

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **구조 요소** | 형태학 연산에 쓰는 커널 (사각형, 타원, 십자) |
| **침식** | 경계를 깎아 물체 축소, 작은 노이즈 제거 |
| **팽창** | 경계를 확장해 물체 부풀리기, 구멍 메움 |
| **열기** | 침식 → 팽창. 노이즈 제거에 최적 |
| **닫기** | 팽창 → 침식. 내부 구멍 메우기에 최적 |
| **그래디언트** | 팽창 − 침식 = 윤곽선 추출 |

## 다음 섹션 미리보기

이것으로 **Chapter 02: 전통적 컴퓨터 비전**이 마무리됩니다! OpenCV의 기본 도구들을 모두 익혔습니다. 다음 챕터 **[딥러닝 기초](../03-deep-learning-basics/01-neural-network.md)**에서는 드디어 신경망의 세계에 발을 들입니다. 뉴런, 레이어, 가중치 — 딥러닝의 근간을 처음부터 배워봅시다.

## 참고 자료

- [OpenCV 공식 튜토리얼 - Morphological Transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) - 침식, 팽창, 열기, 닫기 공식 가이드
- [PyImageSearch - OpenCV Morphological Operations](https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/) - 실용적 예제와 시각적 비교
- [GeeksforGeeks - Morphological Operations](https://www.geeksforgeeks.org/python/python-opencv-morphological-operations/) - Python 코드 예제 모음
- [OpenCV 공식 - Eroding and Dilating](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html) - 침식과 팽창의 원리 상세 설명
