# 전이 학습

> 사전 학습 모델 활용법

## 개요

[CIFAR-10](./02-cifar10.md)에서 처음부터 CNN을 학습시켜 92% 정확도를 달성했습니다. 하지만 실무에서는 처음부터 학습하는 경우가 오히려 드뭅니다. **이미 학습된 모델의 지식을 빌려와서** 새로운 문제에 적용하는 것이 훨씬 효율적이기 때문이죠. 이것이 바로 **전이 학습(Transfer Learning)**입니다.

**선수 지식**: [CIFAR-10 분류](./02-cifar10.md), [ResNet](../05-cnn-architectures/03-resnet.md)
**학습 목표**:
- 전이 학습의 원리와 왜 효과적인지 이해한다
- 특징 추출기(Feature Extractor)와 파인 튜닝의 차이를 구분할 수 있다
- PyTorch에서 사전 학습 모델을 로드하고 새 작업에 적용할 수 있다

## 왜 알아야 할까?

현실의 대부분의 프로젝트에서는 데이터가 수천~수만 장 수준입니다. 이 정도 데이터로 ResNet-50 같은 대형 모델을 처음부터 학습하면, **과적합**이 발생하거나 아예 수렴하지 못합니다. ImageNet(120만 장, 1,000 클래스)에서 이미 학습된 모델은 "가장자리 → 질감 → 부분 → 전체"로 이어지는 **시각적 특징의 계층구조**를 이미 알고 있기 때문에, 이 지식을 빌려오면 적은 데이터로도 높은 성능을 낼 수 있습니다.

실제로 전이 학습은 실무에서 가장 많이 쓰이는 기법 중 하나입니다. Kaggle 대회의 상위권 솔루션 대부분이 사전 학습 모델에서 시작하고, 산업계에서도 "처음부터 학습"은 데이터와 GPU 예산이 충분한 극소수의 경우에만 해당됩니다.

## 핵심 개념

### 1. 전이 학습이란 — 지식의 재활용

> 💡 **비유**: 피아노를 10년 배운 사람이 기타를 배우면, 악보 읽기, 리듬 감각, 손가락 민첩성 같은 **음악적 기초**가 그대로 전이됩니다. 처음부터 배우는 사람보다 훨씬 빠르게 기타를 익힐 수 있죠. 전이 학습도 마찬가지입니다 — ImageNet에서 배운 "시각적 기초 체력"을 새로운 문제에 재활용하는 것입니다.

전이 학습의 핵심 아이디어는 간단합니다:

> **큰 데이터셋으로 학습된 모델의 가중치** → **새로운(작은) 데이터셋의 출발점으로 사용**

CNN이 학습하는 특징에는 재미있는 계층 구조가 있습니다:

| 레이어 위치 | 학습하는 특징 | 범용성 |
|-------------|--------------|--------|
| 초기 레이어 | 가장자리, 색상, 질감 | 매우 높음 (거의 모든 이미지에 공통) |
| 중간 레이어 | 패턴, 부분 형태 | 높음 |
| 후기 레이어 | 객체의 부분, 조합 | 중간 |
| 마지막 FC 레이어 | 클래스 분류 | 낮음 (원래 데이터셋에 특화) |

초기 레이어가 학습한 "가장자리 검출", "질감 인식" 같은 특징은 **어떤 이미지 문제에서든** 유용합니다. 고양이를 분류하든, 암세포를 검출하든, 위성 사진을 분석하든 — 결국 가장자리와 질감에서 시작하니까요.

### 2. 두 가지 접근법 — 특징 추출 vs 파인 튜닝

전이 학습에는 크게 두 가지 방식이 있습니다:

**방법 1: 특징 추출기(Feature Extractor)**

사전 학습 모델의 가중치를 **완전히 고정**하고, 마지막 분류 레이어만 교체합니다. 기존 모델을 "특징 추출 기계"로만 사용하는 거죠.

> 💡 **비유**: 전문 사진작가(사전 학습 모델)가 찍어준 사진을 받아서, 내가 분류 라벨만 붙이는 것과 같습니다. 사진작가의 촬영 실력(특징 추출 능력)은 건드리지 않고, 분류 기준만 내 것으로 바꾸는 거죠.

- 장점: 학습이 매우 빠르고, 적은 데이터에서도 안정적
- 단점: 성능의 상한선이 제한적
- 추천 상황: 데이터가 매우 적거나(수백 장), 빠른 프로토타이핑이 필요할 때

**방법 2: 파인 튜닝(Fine-Tuning)**

사전 학습 가중치로 **초기화한 후**, 전체 또는 일부 레이어를 새 데이터로 **다시 학습**합니다.

> 💡 **비유**: 유명 셰프(사전 학습 모델)의 레시피를 기본으로 하되, 한국인 입맛에 맞게 양념을 **미세 조정**하는 것과 같습니다. 셰프의 기본기는 유지하면서, 세부 맛을 조정하는 거죠.

- 장점: 더 높은 성능 가능
- 단점: 과적합 위험, 더 많은 데이터 필요
- 추천 상황: 데이터가 충분하고(수천 장 이상), 최고 성능이 필요할 때

파인 튜닝의 구체적인 전략은 [다음 섹션](./04-fine-tuning.md)에서 자세히 다룹니다.

### 3. 데이터셋 크기와 유사성에 따른 전략 선택

어떤 방법을 선택할지는 **데이터셋 크기**와 **원본 데이터셋과의 유사성**으로 결정합니다:

| | 원본과 유사한 데이터 | 원본과 다른 데이터 |
|------|---------------------|-------------------|
| **데이터 적음** | 특징 추출기 (후기 레이어만 활용) | 특징 추출기 (초기~중기 레이어 활용) |
| **데이터 많음** | 전체 파인 튜닝 | 많은 레이어 파인 튜닝 |

예를 들어, ImageNet으로 학습된 모델을 사용할 때:
- **강아지 품종 분류** (유사 + 적은 데이터) → 특징 추출기로 충분
- **의료 영상 분류** (다른 + 적은 데이터) → 초기 레이어 특징만 활용
- **대규모 자동차 분류** (유사 + 많은 데이터) → 전체 파인 튜닝

### 4. PyTorch에서 사전 학습 모델 사용하기

PyTorch의 `torchvision.models`는 ImageNet에서 학습된 다양한 모델을 제공합니다. 2024년 기준 새로운 API(`weights` 파라미터)를 사용하는 것이 권장됩니다.

```python
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# === 방법 1: 최신 API (권장) ===
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# === 방법 2: 문자열로 지정 ===
model = models.resnet18(weights="IMAGENET1K_V1")

# === 가중치에 포함된 전처리 정보 확인 ===
weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()  # 자동으로 적절한 전처리 반환
print(f"입력 크기: {weights.meta['min_size']}x{weights.meta['min_size']}")
print(f"카테고리 수: {len(weights.meta['categories'])}")
# 입력 크기: 224x224
# 카테고리 수: 1000
```

> ⚠️ **흔한 오해**: 옛날 코드에서 `pretrained=True`를 자주 보는데, 이 파라미터는 **deprecated**되었습니다. `weights=ResNet18_Weights.IMAGENET1K_V1` 방식을 사용하세요. 새 API는 가중치 버전 관리와 전처리 정보까지 함께 제공합니다.

## 실습: 전이 학습으로 CIFAR-10 분류하기

### Step 1: 데이터 준비 (ImageNet 전처리 기준)

사전 학습 모델을 사용할 때는 **원본 학습 데이터의 전처리**를 그대로 따라야 합니다. ImageNet 모델은 입력을 224×224로 리사이즈하고, 특정 평균/표준편차로 정규화합니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet 기준 전처리 (사전 학습 모델 사용 시 필수!)
# CIFAR-10은 32x32이므로 224x224로 리사이즈
train_transform = transforms.Compose([
    transforms.Resize(224),                          # 224x224로 리사이즈
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 평균
                         std=[0.229, 0.224, 0.225]),  # ImageNet 표준편차
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True,
                                  download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)
```

### Step 2: 특징 추출기 방식

```python
def create_feature_extractor(num_classes=10):
    """ResNet-18을 특징 추출기로 사용"""
    # 사전 학습된 ResNet-18 로드
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 모든 파라미터 고정 (학습 안 함)
    for param in model.parameters():
        param.requires_grad = False

    # 마지막 FC 레이어만 교체 (이것만 학습)
    # ResNet-18의 fc: Linear(512, 1000) → Linear(512, 10)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # 새로 만든 fc 레이어는 requires_grad=True가 기본값

    return model

model_fe = create_feature_extractor().to(DEVICE)

# 학습 가능한 파라미터 확인
trainable = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_fe.parameters())
print(f"전체 파라미터: {total:,}")
print(f"학습 가능 파라미터: {trainable:,} ({100*trainable/total:.1f}%)")
# 전체 파라미터: 11,181,642
# 학습 가능 파라미터: 5,130 (0.05%)  ← FC 레이어만!
```

전체 1,100만 개 파라미터 중 **5,130개(0.05%)**만 학습합니다. 나머지 99.95%는 ImageNet에서 배운 지식을 그대로 유지하는 거죠. 그럼에도 놀라운 성능을 보여줍니다.

### Step 3: 학습과 평가

```python
criterion = nn.CrossEntropyLoss()
# 특징 추출기: 학습 가능한 파라미터만 옵티마이저에 전달
optimizer = optim.Adam(model_fe.fc.parameters(), lr=1e-3)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100. * correct / total


# === 학습 실행 ===
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model_fe, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = evaluate(
        model_fe, test_loader, criterion, DEVICE)
    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# 예상 출력:
# Epoch  1/10 | Train Acc: 82.15% | Test Acc: 84.30%
# Epoch  5/10 | Train Acc: 89.42% | Test Acc: 89.85%
# Epoch 10/10 | Train Acc: 91.03% | Test Acc: 90.50%
```

결과를 보세요. **FC 레이어만 10 에포크 학습**했는데 90.5%입니다! [CIFAR-10](./02-cifar10.md)에서 3블록 CNN을 50 에포크 학습해서 92%를 달성한 것과 비교하면, **훨씬 적은 노력으로 비슷한 성능**을 낸 것이죠.

### Step 4: 파인 튜닝 방식 (비교용)

파인 튜닝은 사전 학습 가중치로 **시작**하되, 전체 모델을 학습합니다.

```python
def create_finetuned_model(num_classes=10):
    """ResNet-18 전체를 파인 튜닝"""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 마지막 FC 레이어 교체 (파인 튜닝도 이건 필수)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # requires_grad를 건드리지 않음 → 모든 레이어가 학습됨
    return model

model_ft = create_finetuned_model().to(DEVICE)

# 학습 가능 파라미터: 전체!
trainable = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
print(f"학습 가능 파라미터: {trainable:,} (전체)")
# 학습 가능 파라미터: 11,181,642 (전체)

# 파인 튜닝은 작은 학습률이 핵심!
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3,
                          momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model_ft, train_loader, criterion, optimizer_ft, DEVICE)
    test_loss, test_acc = evaluate(
        model_ft, test_loader, criterion, DEVICE)
    scheduler.step()
    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# 예상 출력:
# Epoch  1/10 | Train Acc: 90.23% | Test Acc: 91.85%
# Epoch  5/10 | Train Acc: 96.87% | Test Acc: 94.12%
# Epoch 10/10 | Train Acc: 98.51% | Test Acc: 95.30%
```

파인 튜닝은 **95.3%**로, 특징 추출기(90.5%)보다 약 5% 높습니다. 다만 모든 파라미터를 학습하므로 시간이 더 걸리고, 과적합 위험도 있습니다.

### 세 가지 방식 비교

| 방법 | 테스트 정확도 | 학습 파라미터 | 학습 시간 |
|------|-------------|--------------|----------|
| 처음부터 학습 (이전 섹션) | ~92% | 1.2M (100%) | 50 에포크 |
| 특징 추출기 | ~90.5% | 5.1K (0.05%) | 10 에포크 |
| 파인 튜닝 | ~95.3% | 11.2M (100%) | 10 에포크 |

파인 튜닝이 10 에포크만에 처음부터 학습한 50 에포크 결과를 뛰어넘었습니다. 이것이 전이 학습의 힘입니다.

## 더 깊이 알아보기

### 전이 학습의 역사 — "Feature Transfer"에서 "Foundation Model"까지

전이 학습 아이디어 자체는 1990년대부터 있었지만, 컴퓨터 비전에서 폭발적으로 성장한 계기는 **2012년 AlexNet**이었습니다. Krizhevsky 등이 ImageNet에서 학습한 AlexNet의 특징이 다른 데이터셋에도 유용하다는 것을 보여준 거죠.

2014년에는 제이슨 요신스키(Jason Yosinski) 등이 *"How transferable are features in deep neural networks?"*라는 논문에서 **CNN 각 레이어의 전이 가능성**을 체계적으로 분석했습니다. 이 논문에서 "초기 레이어는 범용적, 후기 레이어는 특화적"이라는 현재까지 통용되는 핵심 발견이 나왔습니다.

그리고 2020년대에 들어서면서 전이 학습은 **Foundation Model**(기반 모델)이라는 더 큰 개념으로 진화했습니다. CLIP, DINO, MAE 같은 대규모 사전 학습 모델이 등장하면서, 한 모델이 분류, 검출, 분할 등 다양한 작업에 전이되는 시대가 열렸죠.

> 💡 **알고 계셨나요?**: Stanford의 CS231n 강의(CNN for Visual Recognition)는 "실무에서 CNN을 처음부터 학습하는 사람은 거의 없다"고 강조합니다. ImageNet 사전 학습이 **사실상 표준(de facto standard)**이 된 것은 2014~2015년경부터이며, 이후 10년간 이 관행은 변하지 않았습니다.

### 사용 가능한 주요 사전 학습 모델

| 모델 | 파라미터 | ImageNet Top-1 | 특징 |
|------|---------|----------------|------|
| ResNet-18 | 11.7M | 69.8% | 가볍고 빠름, 입문용 |
| ResNet-50 | 25.6M | 76.1% | 가장 널리 사용되는 백본 |
| EfficientNet-B0 | 5.3M | 77.1% | 효율성 최강 |
| ConvNeXt-Tiny | 28.6M | 82.1% | 최신 순수 CNN |
| ViT-B/16 | 86.6M | 81.1% | Transformer 기반 |

어떤 모델을 선택할지는 **데이터 크기, GPU 메모리, 추론 속도 요구사항**에 따라 달라집니다. 처음에는 ResNet-18이나 EfficientNet-B0로 시작하고, 성능이 부족하면 더 큰 모델로 올리는 것이 일반적인 접근입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "전이 학습은 데이터가 적을 때만 유용하다" — 사실 데이터가 많아도 전이 학습이 유리합니다. ImageNet 사전 학습으로 시작하면 수렴이 빠르고, 최종 성능도 더 높은 경우가 많습니다. Google Brain의 연구에 따르면 300만 장 이상의 데이터에서도 사전 학습이 도움이 되었습니다.

> 🔥 **실무 팁**: 사전 학습 모델을 사용할 때 **반드시 해당 모델의 전처리를 따르세요**. ImageNet 모델은 `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`로 정규화해야 합니다. CIFAR-10의 자체 평균/표준편차를 쓰면 성능이 떨어집니다.

> 🔥 **실무 팁**: 특징 추출기 방식에서 학습 속도를 더 높이려면, 먼저 모든 이미지의 특징을 추출해서 저장한 뒤, 저장된 특징만으로 분류기를 학습하세요. 매 에포크마다 CNN을 통과시킬 필요가 없어 **5~10배 빠릅니다**.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 전이 학습 | 큰 데이터셋에서 학습된 모델을 새 작업에 재활용하는 기법 |
| 특징 추출기 | 사전 학습 가중치를 고정하고 분류 레이어만 학습. 빠르고 안정적 |
| 파인 튜닝 | 사전 학습 가중치로 시작하되 전체/일부를 재학습. 높은 성능 가능 |
| ImageNet 전처리 | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]로 정규화 필수 |
| CNN 특징 계층 | 초기 레이어(범용) → 후기 레이어(특화). 초기 레이어일수록 전이성이 높음 |
| 모델 선택 | ResNet-18(입문) → ResNet-50(범용) → EfficientNet(효율) 순으로 시도 |

## 다음 섹션 미리보기

전이 학습의 기본 개념을 이해했으니, 이제 **어떻게 파인 튜닝을 잘 할 것인가**라는 실전 문제를 다룹니다. [파인 튜닝 전략](./04-fine-tuning.md)에서는 레이어별 학습률 차등 적용, 점진적 해동(Gradual Unfreezing), 학습률 워밍업 등 파인 튜닝의 성능을 극대화하는 고급 테크닉을 배웁니다.

## 참고 자료

- [PyTorch 공식 전이 학습 튜토리얼](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) - ResNet을 활용한 전이 학습 공식 가이드
- [CS231n: Transfer Learning](https://cs231n.github.io/transfer-learning/) - Stanford CS231n의 전이 학습 설명 (데이터 크기/유사성 전략)
- [TorchVision 사전 학습 모델 목록](https://docs.pytorch.org/vision/stable/models.html) - 사용 가능한 모든 사전 학습 모델과 성능
- [How transferable are features in deep neural networks? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792) - CNN 레이어별 전이 가능성을 분석한 핵심 논문
- [A Practical Guide to Transfer Learning using PyTorch](https://www.kdnuggets.com/2023/06/practical-guide-transfer-learning-pytorch.html) - 실무 중심 전이 학습 가이드
