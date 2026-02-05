# 특징점 검출

> SIFT, SURF, ORB 특징 추출기

## 개요

두 장의 사진에서 같은 건물을 찾아 연결하거나, 파노라마 사진을 만들려면 "이미지 A의 이 부분이 이미지 B의 저 부분과 같다"는 것을 알아내야 합니다. 이를 가능하게 하는 것이 **특징점(Keypoint)** 검출과 **기술자(Descriptor)** 추출입니다.

**선수 지식**: [에지 검출](./03-edge-detection.md) — 기울기, 에지의 개념
**학습 목표**:
- 특징점과 기술자가 무엇인지 이해한다
- SIFT, ORB의 차이와 적합한 상황을 구분할 수 있다
- 두 이미지 사이의 특징점 매칭을 수행할 수 있다

## 왜 알아야 할까?

특징점 검출은 **이미지 매칭, 파노라마 합성, 3D 복원, 객체 인식, AR(증강현실)** 등의 핵심 기술입니다. 딥러닝이 등장하기 전에는 SIFT가 CV의 중심이었고, 지금도 3D 비전([카메라 기하학](../16-3d-vision/03-camera-geometry.md), [SLAM](../16-3d-vision/04-slam.md))에서 활발히 사용됩니다.

## 핵심 개념

### 1. 특징점(Keypoint)과 기술자(Descriptor)

> 💡 **비유**: 친구에게 건물 사진 속 특정 위치를 알려준다고 생각해 보세요. "3층 왼쪽에서 두 번째 창문"이라고 하면 **위치**가 특징점이고, "파란 프레임, 사각형, 커튼 있음"이라는 **묘사**가 기술자입니다. 다른 각도에서 찍은 사진에서도 같은 묘사를 찾으면 같은 창문임을 알 수 있습니다.

| 개념 | 비유 | 실제 의미 |
|------|------|----------|
| **특징점 (Keypoint)** | "3층 왼쪽 창문" | 이미지에서 고유하게 식별 가능한 점의 **위치와 크기** |
| **기술자 (Descriptor)** | "파란 프레임, 사각형" | 그 점 주변의 패턴을 숫자로 요약한 **벡터** |

좋은 특징점의 조건:
- **반복성**: 같은 장면을 다른 각도/조명에서 찍어도 같은 점이 검출됨
- **구별성**: 다른 점들과 헷갈리지 않는 고유한 패턴을 가짐
- **안정성**: 크기 변화, 회전에도 잘 유지됨

### 2. SIFT — 스케일에 불변한 특징

> 💡 **비유**: SIFT는 **어디서 봐도 알아볼 수 있는 지문**을 만드는 알고리즘입니다. 사진을 확대하든, 축소하든, 회전하든 같은 지문이 나옵니다.

**S**cale-**I**nvariant **F**eature **T**ransform의 약자로, 2004년 David Lowe가 발표했습니다.

SIFT의 동작 과정:
1. **스케일 공간 구축**: 이미지를 여러 크기로 블러하여 다양한 스케일에서 특징 탐색
2. **극값 검출**: 주변보다 튀는 점(극대/극소)을 특징점 후보로 선정
3. **방향 할당**: 각 특징점에 주요 방향을 부여 (회전 불변성)
4. **기술자 생성**: 16×16 영역을 4×4 블록 16개로 나누고, 각 블록에서 8방향 히스토그램 → **128차원 벡터**

```python
import cv2

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

# SIFT 생성 (opencv-contrib-python 필요)
sift = cv2.SIFT_create()

# 특징점 검출 + 기술자 추출
keypoints, descriptors = sift.detectAndCompute(img, None)

print(f"검출된 특징점 수: {len(keypoints)}")
print(f"기술자 형태: {descriptors.shape}")  # (N, 128)
print(f"기술자 타입: {descriptors.dtype}")   # float32

# 특징점 정보 확인 (첫 번째)
kp = keypoints[0]
print(f"\n첫 번째 특징점:")
print(f"  위치: ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})")
print(f"  크기: {kp.size:.1f}")
print(f"  방향: {kp.angle:.1f}°")
```

### 3. ORB — 빠르고 무료인 대안

> 💡 **비유**: SIFT가 정밀한 전문 지문 감식이라면, ORB는 **빠른 현장 지문 대조**입니다. 정확도는 약간 낮지만 **훨씬 빠르고** 무료(특허 없음)입니다.

**O**riented FAST and **R**otated **B**RIEF의 약자입니다. FAST로 특징점을 빠르게 찾고, BRIEF로 기술자를 효율적으로 만듭니다.

| 비교 항목 | SIFT | ORB |
|----------|------|-----|
| 기술자 크기 | 128차원 (float) | 32차원 (binary) |
| 속도 | 느림 | **~100배 빠름** |
| 라이선스 | 특허 만료 (현재 무료) | 무료 |
| 스케일 불변 | 뛰어남 | 보통 |
| 회전 불변 | 뛰어남 | 좋음 |
| 적합 용도 | 정밀 매칭, 3D 복원 | 실시간 앱, 모바일 |

```python
import cv2

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

# ORB 생성 (기본 500개 특징점)
orb = cv2.ORB_create(nfeatures=1000)

# 특징점 검출 + 기술자 추출
keypoints, descriptors = orb.detectAndCompute(img, None)

print(f"검출된 특징점 수: {len(keypoints)}")
print(f"기술자 형태: {descriptors.shape}")  # (N, 32)
print(f"기술자 타입: {descriptors.dtype}")   # uint8 (바이너리)
```

### 4. 특징점 매칭 — 두 이미지 연결하기

특징점을 찾았으면, 두 이미지의 기술자를 **비교하여 같은 점을 찾습니다.**

```python
import cv2

img1 = cv2.imread("photo1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("photo2.jpg", cv2.IMREAD_GRAYSCALE)

# ORB로 특징점 추출
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# BFMatcher로 매칭 (바이너리 기술자 → 해밍 거리)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 거리순 정렬 (작을수록 유사)
matches = sorted(matches, key=lambda x: x.distance)

print(f"이미지1 특징점: {len(kp1)}개")
print(f"이미지2 특징점: {len(kp2)}개")
print(f"매칭된 쌍: {len(matches)}개")
print(f"최고 매칭 거리: {matches[0].distance:.1f}")
print(f"최저 매칭 거리: {matches[-1].distance:.1f}")
```

```python
import cv2

# SIFT 매칭 (float 기술자 → L2 거리 + KNN)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# Lowe의 비율 테스트 — 좋은 매칭만 남기기
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # 최근접과 차근접의 비율
        good_matches.append(m)

print(f"전체 매칭: {len(matches)}개")
print(f"좋은 매칭: {len(good_matches)}개")
```

### 5. 매칭 방법 비교

| 매칭 방법 | 기술자 타입 | 거리 척도 | 특징 |
|----------|-----------|----------|------|
| **BFMatcher** (Brute-Force) | 모두 가능 | HAMMING 또는 L2 | 모든 쌍 비교, 정확하지만 느림 |
| **FLANN** | float 권장 | L2 | 근사 최근접 탐색, 대량 매칭에 빠름 |

> **실무 팁**: ORB → BFMatcher(HAMMING), SIFT → FLANN 또는 BFMatcher(L2)를 사용하세요.

## 실습: 직접 해보기

### SIFT vs ORB 속도 비교

```python
import cv2
import time

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

# SIFT 속도 측정
sift = cv2.SIFT_create()
start = time.time()
kp_sift, des_sift = sift.detectAndCompute(img, None)
time_sift = time.time() - start

# ORB 속도 측정
orb = cv2.ORB_create(nfeatures=len(kp_sift))  # 같은 수로 맞춤
start = time.time()
kp_orb, des_orb = orb.detectAndCompute(img, None)
time_orb = time.time() - start

print(f"SIFT: {len(kp_sift)}개, {time_sift*1000:.1f}ms, 기술자 {des_sift.shape}")
print(f"ORB:  {len(kp_orb)}개, {time_orb*1000:.1f}ms, 기술자 {des_orb.shape}")
print(f"속도 차이: ORB가 {time_sift/time_orb:.1f}배 빠름")
```

## 더 깊이 알아보기

> 💡 **알고 계셨나요?**: SIFT의 역사에는 **20년간의 특허 전쟁**이 있었습니다. David Lowe가 1999년 SIFT를 발표한 후, 소속 대학인 **UBC(University of British Columbia)**가 특허를 출원했거든요. 이 특허 때문에 SIFT를 상업적으로 사용하려면 라이선스 비용을 내야 했고, OpenCV에서도 별도 모듈(`opencv-contrib`)에 격리되어 있었습니다.

이 특허 문제가 바로 **ORB가 탄생한 직접적인 이유**입니다. 2011년 OpenCV Labs(Willow Garage)의 연구원들이 "SIFT/SURF처럼 강력하면서도 **특허에서 자유롭고 빠른** 대안"을 목표로 ORB를 개발했어요. FAST(특징점 검출) + BRIEF(기술자)에 방향 정보를 추가해서, 실시간 애플리케이션에서도 쓸 수 있게 만든 거죠.

그리고 2020년 3월, SIFT 특허가 드디어 **만료**되었습니다! 이후 OpenCV **4.4.0** 버전부터 SIFT가 메인 모듈로 이동하여, `opencv-contrib` 없이도 자유롭게 사용할 수 있게 되었죠. 20년 만에 풀린 족쇄인 셈입니다. 지금은 SIFT든 ORB든 상황에 맞게 자유롭게 선택할 수 있는 좋은 시대가 되었어요.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "SIFT는 딥러닝에 의해 완전히 대체되었다" — 전혀 그렇지 않습니다! **3D 복원(Structure from Motion)**, **SLAM**, **파노라마 합성** 같은 기하학적 비전 분야에서 SIFT는 지금도 활발히 사용되고 있어요. 특히 학습 데이터가 없는 새로운 환경에서의 매칭에는 여전히 SIFT가 강력합니다. SuperPoint, SuperGlue 같은 딥러닝 기반 매칭 기법도 등장했지만, 범용성 면에서 SIFT를 완전히 대체하기엔 아직 갈 길이 멀죠.

> ⚠️ **흔한 오해**: "ORB는 SIFT의 완벽한 대체품이다" — ORB는 **속도에서 압도적**이지만, **시점 변화나 조명 변화가 큰 상황**에서는 SIFT보다 덜 강건합니다. ORB의 기술자는 32바이트 바이너리로 SIFT의 512바이트(128 x float32) 대비 매우 컴팩트하지만, 그만큼 표현력에 한계가 있거든요. 실시간이 중요하면 ORB, 정확도가 중요하면 SIFT — 이렇게 용도에 맞게 선택하는 것이 정답입니다.

> 💡 **알고 계셨나요?**: SIFT 특허가 **2020년 3월에 만료**되면서, OpenCV **4.4.0**(2020년 7월)부터 SIFT가 메인 모듈로 이동했습니다. 이전에는 `opencv-contrib-python`을 별도로 설치해야 했지만, 이제는 기본 `opencv-python` 패키지만으로도 `cv2.SIFT_create()`를 사용할 수 있어요. 오래된 튜토리얼에서 "SIFT를 쓰려면 contrib를 설치하세요"라는 안내를 보더라도, 4.4.0 이상이라면 무시하셔도 됩니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **특징점 (Keypoint)** | 이미지에서 고유하게 식별 가능한 점 (위치, 크기, 방향) |
| **기술자 (Descriptor)** | 특징점 주변 패턴을 숫자 벡터로 요약한 것 |
| **SIFT** | 128차원, 스케일/회전 불변, 정밀하지만 느림 |
| **ORB** | 32차원 바이너리, 매우 빠름, 실시간 적합 |
| **매칭** | 두 이미지의 기술자 간 거리를 비교하여 대응점 찾기 |
| **비율 테스트** | 좋은 매칭만 남기는 Lowe의 기법 (0.75 기준) |

## 다음 섹션 미리보기

에지와 특징점은 이미지의 **구조적 정보**를 추출하는 방법이었습니다. 다음 섹션 **[형태학적 연산](./05-morphology.md)**에서는 이미지의 **형태(모양)**를 변형하는 침식, 팽창, 열기, 닫기 연산을 배웁니다. 이진 이미지 처리와 노이즈 제거에 필수적인 도구입니다.

## 참고 자료

- [OpenCV 공식 튜토리얼 - SIFT Introduction](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) - SIFT 알고리즘 공식 설명
- [OpenCV 공식 튜토리얼 - Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html) - BFMatcher, FLANN 매칭 가이드
- [Machine Learning Mastery - Image Feature Extraction in OpenCV](https://machinelearningmastery.com/opencv_sift_surf_orb_keypoints/) - SIFT/SURF/ORB 비교 가이드
- [Pysource - Feature Detection Tutorial](https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/) - Python 코드 예제와 시각화
