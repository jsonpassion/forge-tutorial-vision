# 시맨틱 세그멘테이션

> FCN, U-Net, DeepLab

## 개요

[객체 탐지](../07-object-detection/01-detection-basics.md)에서는 이미지 속 객체를 **사각형 박스**로 찾았습니다. 하지만 자율주행차가 도로의 정확한 경계를 알아야 하거나, 의료 영상에서 종양의 정확한 윤곽을 파악해야 한다면 어떨까요? 바운딩 박스만으로는 부족합니다. **시맨틱 세그멘테이션(Semantic Segmentation)**은 이미지의 **모든 픽셀**에 클래스 라벨을 부여하는 기술로, "이 픽셀은 도로, 저 픽셀은 자동차, 이 픽셀은 하늘"처럼 **픽셀 단위의 분류**를 수행합니다.

**선수 지식**: [CNN 아키텍처](../05-cnn-architectures/03-resnet.md), [객체 탐지 기초](../07-object-detection/01-detection-basics.md)
**학습 목표**:
- 시맨틱 세그멘테이션과 분류/탐지의 차이를 이해한다
- FCN, U-Net, DeepLab의 핵심 아이디어를 설명할 수 있다
- PyTorch로 간단한 세그멘테이션 모델을 구현할 수 있다

## 왜 알아야 할까?

분류는 "이 사진에 뭐가 있나?", 탐지는 "어디에 있나?", 그리고 세그멘테이션은 "정확히 어떤 모양인가?"에 답합니다. 세그멘테이션은 컴퓨터 비전에서 **가장 정밀한 이해**를 요구하는 태스크입니다.

| 태스크 | 출력 단위 | 예시 |
|--------|----------|------|
| **분류** | 이미지 1장 → 라벨 1개 | "고양이" |
| **객체 탐지** | 이미지 1장 → 박스 N개 | 고양이 박스, 강아지 박스 |
| **시맨틱 세그멘테이션** | 이미지 1장 → 픽셀 맵 | 모든 픽셀에 클래스 부여 |

실무에서 세그멘테이션이 핵심인 분야는 놀라울 정도로 많습니다:

- **자율주행**: 도로, 차선, 보행자, 차량의 정확한 경계 파악
- **의료 영상**: CT/MRI에서 종양, 장기, 혈관의 정밀 윤곽 추출
- **위성/항공 영상**: 건물, 도로, 농경지, 수역 분류
- **영상 편집**: 배경 제거, 인물 분리, 특수 효과
- **로봇 내비게이션**: 이동 가능 영역과 장애물 구분

## 핵심 개념

### 1. 시맨틱 세그멘테이션이란? — 색칠 공부의 AI 버전

> 💡 **비유**: 어린 시절 색칠 공부를 떠올려보세요. 그림의 각 영역에 정해진 색을 칠하죠 — 하늘은 파란색, 풀은 초록색, 집은 빨간색. 시맨틱 세그멘테이션은 AI가 하는 색칠 공부입니다. 이미지의 **모든 픽셀**을 보고, "이건 하늘, 이건 도로, 이건 자동차"라고 각각 다른 색으로 칠하는 거죠.

"시맨틱(Semantic)"은 **의미론적**이라는 뜻입니다. 단순히 경계선을 찾는 게 아니라, 각 픽셀이 **의미적으로 무엇인지**를 판단합니다. 핵심적인 특징은 **같은 클래스의 모든 객체를 하나로 취급**한다는 점입니다. 예를 들어 사진에 사람이 3명 있어도, 시맨틱 세그멘테이션은 모두 "사람" 색으로 칠합니다. 개별 인스턴스를 구분하지 않죠 — 그건 [인스턴스 세그멘테이션](./02-instance-segmentation.md)의 영역입니다.

입출력을 정리하면:

- **입력**: H × W × 3 컬러 이미지
- **출력**: H × W 크기의 **클래스 맵** (각 픽셀에 0~C-1 중 하나의 클래스 ID)

> ⚠️ **흔한 오해**: "세그멘테이션 = 에지 검출"이라고 생각하기 쉽지만, 완전히 다릅니다. [에지 검출](../02-classical-cv/03-edge-detection.md)은 밝기 변화가 급격한 경계선을 찾는 저수준 연산이고, 세그멘테이션은 각 픽셀의 **의미적 카테고리**를 예측하는 고수준 인식입니다. 에지 검출기는 나무와 하늘의 경계를 찾을 수 있지만, 어느 쪽이 나무이고 어느 쪽이 하늘인지는 모릅니다.

### 2. FCN — 분류 네트워크를 세그멘테이션에 쓰다

> 💡 **비유**: 돋보기로 책의 한 글자씩만 읽던 사람이, 갑자기 "페이지 전체를 한 번에 읽는 방법"을 발견한 것과 같습니다. 기존 CNN은 이미지를 하나의 라벨로 요약했지만, FCN은 **이미지의 모든 위치에 대해 동시에 분류**하는 방법을 제시했습니다.

2015년, UC Berkeley의 **Jonathan Long, Evan Shelhamer, Trevor Darrell**은 획기적인 논문을 발표합니다. 핵심 아이디어는 놀랍도록 간단했습니다: **분류 네트워크의 FC(Fully Connected) 레이어를 1×1 합성곱으로 바꾸면, 임의의 크기 입력에 대해 공간 정보를 보존한 출력을 얻을 수 있다!**

이것이 바로 **FCN(Fully Convolutional Network)**이고, 딥러닝 기반 세그멘테이션의 시작점입니다.

**FCN의 핵심 구조**:

1. **인코더(Encoder)**: VGG16 같은 분류 네트워크의 합성곱 부분을 그대로 사용 → 특징 맵 추출
2. **1×1 합성곱**: FC 레이어 대신 1×1 conv로 교체 → 클래스 수만큼의 채널을 가진 특징 맵 생성
3. **업샘플링**: 줄어든 특징 맵을 원본 크기로 복원 (전치 합성곱 사용)

**FCN의 3가지 변형**:

| 변형 | 업샘플링 방식 | 특징 |
|------|-------------|------|
| **FCN-32s** | 마지막 레이어에서 32배 업샘플 | 가장 빠르지만 거칠음 |
| **FCN-16s** | pool4 + 16배 업샘플 | 중간 수준의 디테일 |
| **FCN-8s** | pool3 + pool4 + 8배 업샘플 | 가장 세밀한 결과 |

FCN-8s가 여러 스케일의 특징을 합치는 **스킵 연결(Skip Connection)** 아이디어는 이후 등장하는 거의 모든 세그멘테이션 모델에 영향을 줍니다.

```python
import torch
import torch.nn as nn

class SimpleFCN(nn.Module):
    """FCN의 핵심 아이디어를 보여주는 간소화된 구현"""
    def __init__(self, num_classes=21):
        super().__init__()
        # 인코더: 합성곱으로 특징 추출 (해상도 1/8로 축소)
        self.encoder = nn.Sequential(
            # 블록 1: 3→64, 해상도 /2
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 블록 2: 64→128, 해상도 /4
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 블록 3: 128→256, 해상도 /8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # 1×1 합성곱: FC 레이어 대신 사용 → 공간 정보 보존
        self.classifier = nn.Conv2d(256, num_classes, 1)
        # 디코더: 전치 합성곱으로 원본 크기로 복원
        self.upsample = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4
        )

    def forward(self, x):
        x = self.encoder(x)           # [B, 256, H/8, W/8]
        x = self.classifier(x)        # [B, num_classes, H/8, W/8]
        x = self.upsample(x)          # [B, num_classes, H, W]
        return x

# 테스트
model = SimpleFCN(num_classes=21)
img = torch.randn(1, 3, 224, 224)  # 224×224 이미지
output = model(img)
print(f"입력: {img.shape}")          # [1, 3, 224, 224]
print(f"출력: {output.shape}")       # [1, 21, 224, 224] — 21개 클래스에 대한 픽셀별 예측
```

### 3. U-Net — 의료 영상의 전설, 그리고 그 이상

> 💡 **비유**: 시험 문제를 풀 때, 첫 번째 읽기에서 핵심 키워드를 메모해두고, 답을 쓸 때 그 메모를 다시 참고하면 훨씬 정확한 답을 쓸 수 있죠. U-Net의 **스킵 연결**이 바로 이 "메모" 역할을 합니다 — 인코더에서 추출한 세밀한 정보를 디코더에 직접 전달하여, 정밀한 경계를 복원합니다.

2015년 MICCAI 학회에서 독일 프라이부르크 대학의 **Olaf Ronneberger**가 발표한 U-Net은, 이름 그대로 **U자 모양**의 아키텍처를 가진 네트워크입니다.

**U-Net의 3가지 핵심 요소**:

1. **수축 경로(Contracting Path, 인코더)**: 합성곱 + 풀링으로 "무엇이 있는지" 파악
2. **확장 경로(Expansive Path, 디코더)**: 업샘플링으로 "어디에 있는지" 복원
3. **스킵 연결(Skip Connection)**: 인코더의 고해상도 특징을 디코더에 직접 연결 → **세밀한 경계 복원**

FCN도 스킵 연결을 사용했지만, U-Net은 이를 **체계적으로 모든 레벨에서** 적용했다는 점이 핵심적인 차이입니다. 인코더의 각 단계에서 추출한 특징 맵을 디코더의 대응 단계에 **concat(이어붙이기)**하여, 저수준 디테일(에지, 텍스처)과 고수준 의미 정보를 동시에 활용합니다.

```python
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    """U-Net의 기본 빌딩 블록: Conv → BN → ReLU 두 번"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class SimpleUNet(nn.Module):
    """U-Net 핵심 구조를 보여주는 간소화된 구현"""
    def __init__(self, num_classes=1):
        super().__init__()
        # === 인코더 (수축 경로) ===
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

        # === 바틀넥 (가장 아래) ===
        self.bottleneck = UNetBlock(256, 512)

        # === 디코더 (확장 경로) ===
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)    # 256(up) + 256(skip) = 512
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)    # 128(up) + 128(skip) = 256
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)     # 64(up) + 64(skip) = 128

        # 최종 1×1 합성곱으로 클래스 수만큼 출력
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # 인코더: 특징 추출 + 해상도 축소
        e1 = self.enc1(x)                  # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))      # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))      # [B, 256, H/4, W/4]

        # 바틀넥
        b = self.bottleneck(self.pool(e3))  # [B, 512, H/8, W/8]

        # 디코더: 업샘플 + 스킵 연결(concat) + 합성곱
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # 스킵 연결!
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)

# 테스트
model = SimpleUNet(num_classes=2)
img = torch.randn(1, 3, 256, 256)
output = model(img)
print(f"입력: {img.shape}")          # [1, 3, 256, 256]
print(f"출력: {output.shape}")       # [1, 2, 256, 256]
```

> 💡 **알고 계셨나요?**: U-Net 논문은 2025년 기준 **118,000회 이상 인용**된, 컴퓨터 비전 역사상 가장 많이 인용된 논문 중 하나입니다. 원래 의료 영상(세포 현미경 이미지)을 위해 설계되었지만, 지금은 Stable Diffusion, DALL-E 등 **이미지 생성 모델의 핵심 백본**으로도 사용되고 있습니다. 의료 AI에서 시작해 생성 AI의 심장이 되다니, 놀라운 여정이죠!

### 4. DeepLab — 넓은 시야를 확보하라

> 💡 **비유**: 숲에서 나무 한 그루만 보면 "소나무"인지 알 수 있지만, 그 나무가 숲의 어디쯤에 있는지, 주변이 호수인지 산인지는 **좀 더 넓은 시야**로 봐야 합니다. DeepLab의 핵심 아이디어인 **Atrous(Dilated) 합성곱**은, 파라미터를 늘리지 않으면서도 "시야(수용 영역)"를 넓히는 기법입니다.

Google의 **Liang-Chieh Chen** 팀이 개발한 DeepLab 시리즈는 시맨틱 세그멘테이션의 또 다른 이정표입니다. FCN과 U-Net이 인코더-디코더 구조에 집중했다면, DeepLab은 **수용 영역(Receptive Field)**을 효율적으로 넓히는 데 초점을 맞췄습니다.

**Atrous(Dilated) 합성곱**이란?

일반 3×3 합성곱은 바로 인접한 9개 픽셀만 봅니다. 하지만 Atrous 합성곱은 필터 사이에 **빈 공간(hole)**을 넣어, 같은 9개 파라미터로 **훨씬 넓은 영역**을 봅니다. 팽창률(dilation rate) $r$을 조절하면 수용 영역 크기를 자유롭게 조정할 수 있죠.

| 팽창률 (r) | 실제 커버 영역 | 파라미터 수 |
|-----------|--------------|-----------|
| r=1 | 3×3 (일반 합성곱) | 9 |
| r=2 | 5×5 영역 | 9 (동일!) |
| r=4 | 9×9 영역 | 9 (동일!) |
| r=8 | 17×17 영역 | 9 (동일!) |

**DeepLab의 진화**:

| 버전 | 연도 | 핵심 기여 |
|------|------|----------|
| **DeepLab v1** | 2015 | Atrous 합성곱 + CRF 후처리 도입 |
| **DeepLab v2** | 2017 | **ASPP(Atrous Spatial Pyramid Pooling)** — 여러 팽창률을 병렬 적용 |
| **DeepLab v3** | 2017 | ASPP 개선 + 배치 정규화 + 이미지 레벨 풀링 |
| **DeepLab v3+** | 2018 | 인코더-디코더 구조 결합 → 경계 복원 크게 개선 |

ASPP는 **서로 다른 팽창률**(예: r=6, 12, 18)의 Atrous 합성곱을 병렬로 적용하고 결과를 합치는 모듈입니다. 마치 카메라의 줌을 여러 단계로 바꿔가며 촬영한 뒤 합성하는 것처럼, **다양한 스케일의 문맥 정보**를 동시에 포착합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — DeepLab v3의 핵심 모듈"""
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        # 1×1 합성곱 (가까운 문맥)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU()
        )
        # 다양한 팽창률의 3×3 Atrous 합성곱
        self.atrous6 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6),  # r=6
            nn.BatchNorm2d(out_ch), nn.ReLU()
        )
        self.atrous12 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12),  # r=12
            nn.BatchNorm2d(out_ch), nn.ReLU()
        )
        self.atrous18 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18),  # r=18
            nn.BatchNorm2d(out_ch), nn.ReLU()
        )
        # 이미지 레벨 풀링 (전역 문맥)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU()
        )
        # 5개 브랜치를 합친 후 1×1 합성곱
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[2:]  # 원본 공간 크기 저장
        # 5가지 스케일의 특징 추출
        feat1 = self.conv1x1(x)
        feat2 = self.atrous6(x)
        feat3 = self.atrous12(x)
        feat4 = self.atrous18(x)
        feat5 = F.interpolate(
            self.global_pool(x), size=size, mode='bilinear', align_corners=False
        )
        # 모든 스케일의 특징을 합쳐서 통합
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)

# 테스트
aspp = ASPP(in_ch=512)
feature_map = torch.randn(1, 512, 32, 32)  # 인코더 출력 가정
output = aspp(feature_map)
print(f"입력: {feature_map.shape}")    # [1, 512, 32, 32]
print(f"ASPP 출력: {output.shape}")    # [1, 256, 32, 32]
```

### 5. 세그멘테이션의 손실 함수 — Cross-Entropy와 Dice Loss

세그멘테이션은 본질적으로 **픽셀 단위 분류**이므로, 기본적으로 Cross-Entropy Loss를 사용합니다. 하지만 실제로는 클래스 불균형 문제가 심각한 경우가 많습니다 — 예를 들어 의료 영상에서 종양은 전체 이미지의 1%도 안 될 수 있죠.

이런 문제를 해결하기 위해 **Dice Loss**가 널리 쓰입니다:

$$\text{Dice Loss} = 1 - \frac{2|P \cap G|}{|P| + |G|}$$

- $P$: 모델 예측(Prediction), $G$: 정답(Ground Truth)
- 분자: 예측과 정답이 겹치는 영역의 2배
- 분모: 예측 영역 + 정답 영역

직관적으로, Dice Loss는 **예측과 정답의 겹침 정도**를 직접 최적화합니다. 겹침이 완벽하면 0, 전혀 겹치지 않으면 1이 됩니다.

```python
import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1.0):
    """Dice Loss 구현 — 클래스 불균형에 강건한 손실 함수"""
    # pred: [B, C, H, W] 소프트맥스 적용된 예측
    # target: [B, C, H, W] 원핫 인코딩된 정답
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)    # [B, C, H*W]
    target_flat = target.view(target.size(0), target.size(1), -1)

    intersection = (pred_flat * target_flat).sum(dim=2)       # 겹치는 영역
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)     # 전체 영역

    dice = (2.0 * intersection + smooth) / (union + smooth)   # Dice 계수
    return 1.0 - dice.mean()  # 1에서 빼서 Loss로 변환

# 실무에서는 CE Loss + Dice Loss를 함께 사용하는 경우가 많습니다
class CombinedLoss(nn.Module):
    """Cross-Entropy + Dice Loss 조합"""
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        # Dice Loss를 위해 소프트맥스 + 원핫 변환
        pred_soft = torch.softmax(pred, dim=1)
        num_classes = pred.size(1)
        target_onehot = torch.zeros_like(pred_soft)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        d_loss = dice_loss(pred_soft, target_onehot)
        return self.ce_weight * ce_loss + self.dice_weight * d_loss
```

## 실습: torchvision으로 사전학습된 DeepLab v3 사용하기

실무에서 세그멘테이션 모델을 처음부터 학습하는 경우는 드물죠. torchvision이 제공하는 사전학습 모델을 바로 활용해봅시다.

```python
import torch
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

# 사전학습된 DeepLab v3+ 모델 로드
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.eval()

# 전처리 파이프라인 (학습 시 사용한 것과 동일해야 함)
preprocess = weights.transforms()

# 임의 이미지로 테스트 (실제로는 PIL Image를 사용)
dummy_input = torch.randn(1, 3, 520, 520)
with torch.no_grad():
    output = model(dummy_input)

# 출력 구조 확인
pred = output['out']           # 메인 출력: [B, 21, H, W]
print(f"출력 크기: {pred.shape}")
print(f"클래스 수: {pred.shape[1]}")  # 21 (Pascal VOC 클래스)

# 픽셀별 클래스 예측
seg_map = pred.argmax(dim=1)   # [B, H, W] — 각 픽셀의 예측 클래스
print(f"세그멘테이션 맵: {seg_map.shape}")
print(f"예측된 클래스들: {seg_map.unique().tolist()}")

# Pascal VOC 클래스 목록
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
```

## 더 깊이 알아보기

### FCN의 탄생 — "FC 레이어가 정말 필요한가?"

FCN 논문(2015)의 핵심 통찰은 단순하면서도 혁명적이었습니다. 기존 분류 네트워크(AlexNet, VGG 등)는 마지막에 Fully Connected 레이어를 사용해 공간 정보를 완전히 버렸습니다. Long과 동료들은 "FC 레이어를 1×1 합성곱으로 바꾸면 공간 정보가 살아 있는 **히트맵**을 얻을 수 있다"는 것을 보여줬죠. 이 간단한 아이디어가 **딥러닝 기반 세그멘테이션**이라는 분야 전체를 열었습니다.

### U-Net의 비하인드 스토리

U-Net을 만든 Ronneberger는 원래 의료 영상 분야의 연구자였습니다. 당시 의료 영상은 **데이터가 극도로 부족한** 분야였는데요 — 세포 현미경 이미지 30장으로 ISBI 2015의 세포 추적 챌린지에서 우승했을 정도입니다. U-Net의 스킵 연결 + 데이터 증강 전략이 적은 데이터에서도 뛰어난 성능을 낸 비결이었습니다. 흥미롭게도, U-Net 아키텍처는 10년 후인 2025년 현재도 **Stable Diffusion, DALL-E** 같은 이미지 생성 모델의 핵심 백본으로 사용되고 있습니다.

### 주요 평가 지표 — mIoU

세그멘테이션 성능을 측정하는 대표적인 지표는 **mIoU(mean Intersection over Union)**입니다:

$$\text{IoU} = \frac{\text{예측} \cap \text{정답}}{\text{예측} \cup \text{정답}}$$

각 클래스별로 IoU를 계산한 뒤 평균을 내면 mIoU입니다. 객체 탐지의 IoU와 같은 개념이지만, 여기서는 **픽셀 단위**로 계산한다는 차이가 있죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "시맨틱 세그멘테이션은 모든 객체를 구분할 수 있다" — 아닙니다! 시맨틱 세그멘테이션은 **클래스만 구분**합니다. 같은 클래스의 서로 다른 객체(예: 사람 3명)는 모두 같은 라벨로 처리됩니다. 개별 객체를 구분하려면 [인스턴스 세그멘테이션](./02-instance-segmentation.md)이 필요합니다.

> 🔥 **실무 팁**: 세그멘테이션 학습 시 **클래스 불균형**이 가장 큰 문제입니다. 배경이 90% 이상인 데이터셋에서는 Cross-Entropy만 쓰면 모델이 "전부 배경"으로 예측해도 90% 정확도를 달성합니다. **Dice Loss나 Focal Loss**를 함께 사용하고, 클래스별 가중치(`CrossEntropyLoss(weight=...)`)를 반드시 설정하세요.

> 💡 **알고 계셨나요?**: "Atrous"라는 이름은 프랑스어 "à trous"(구멍이 있는)에서 왔습니다. 필터 사이에 "구멍"을 넣어 수용 영역을 넓히는 아이디어를 직관적으로 표현한 거죠. 영어로는 Dilated Convolution이라고도 부르며, 두 용어는 완전히 동일한 연산을 가리킵니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 시맨틱 세그멘테이션 | 모든 픽셀에 클래스 라벨을 부여하는 태스크 |
| FCN | FC 레이어를 1×1 합성곱으로 바꿔, 분류 네트워크를 세그멘테이션에 활용 |
| U-Net | U자형 인코더-디코더 + 모든 레벨의 스킵 연결 → 세밀한 경계 복원 |
| DeepLab | Atrous(Dilated) 합성곱 + ASPP로 다양한 스케일의 문맥 정보 포착 |
| ASPP | 서로 다른 팽창률의 합성곱을 병렬 적용하는 다중 스케일 모듈 |
| Dice Loss | 예측과 정답의 겹침 정도를 직접 최적화 → 클래스 불균형에 강건 |
| mIoU | 세그멘테이션 대표 평가 지표 — 클래스별 IoU의 평균 |

## 다음 섹션 미리보기

시맨틱 세그멘테이션은 **같은 클래스의 객체를 하나로 묶어** 처리합니다. 하지만 자율주행에서는 앞에 있는 자동차 3대를 **각각 구분**해야 합니다. 다음 섹션 [인스턴스 세그멘테이션](./02-instance-segmentation.md)에서는 Mask R-CNN, YOLACT 등 **개별 객체를 구분하면서 동시에 픽셀 단위 마스크를 예측**하는 기술을 살펴봅니다.

## 참고 자료

- [Fully Convolutional Networks for Semantic Segmentation (Long et al., 2015)](https://arxiv.org/abs/1411.4038) - 딥러닝 기반 세그멘테이션의 시작점
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597) - 118,000+ 인용의 전설적 논문
- [DeepLab v3+: Encoder-Decoder with Atrous Separable Convolution (Chen et al., 2018)](https://arxiv.org/abs/1802.02611) - DeepLab 시리즈의 완성판
- [PyTorch Semantic Segmentation Tutorial](https://pytorch.org/vision/stable/models.html#semantic-segmentation) - torchvision 공식 문서
- [Semantic Segmentation 시각적 가이드 (Papers with Code)](https://paperswithcode.com/task/semantic-segmentation) - 최신 벤치마크와 모델 비교
