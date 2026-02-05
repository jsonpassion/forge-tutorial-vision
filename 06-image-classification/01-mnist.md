# MNIST 손글씨 분류

> 첫 번째 딥러닝 프로젝트

## 개요

지금까지 CNN의 이론과 아키텍처를 배웠습니다. 이제 드디어 **처음부터 끝까지 완전한 딥러닝 프로젝트**를 직접 수행할 시간입니다. 데이터 로드 → 모델 설계 → 학습 → 평가까지, MNIST 손글씨 숫자 분류를 통해 딥러닝 파이프라인의 전체 흐름을 손에 익힙니다.

**선수 지식**: [PyTorch 기초](../03-deep-learning-basics/05-pytorch-fundamentals.md), [CNN 핵심 개념](../04-cnn-fundamentals/01-convolution.md) 전체
**학습 목표**:
- PyTorch로 데이터셋을 로드하고 전처리할 수 있다
- CNN 모델을 설계하고 학습시킬 수 있다
- 학습/검증 곡선을 분석하고 결과를 평가할 수 있다

## 왜 알아야 할까?

MNIST는 딥러닝의 **"Hello, World!"**입니다. 데이터가 깔끔하고 모델이 빠르게 수렴하기 때문에, 파이프라인의 전체 흐름을 익히기에 최적이죠. 여기서 익히는 패턴 — DataLoader, 학습 루프, 검증, 모델 저장 — 은 이미지넷 분류든, 객체 탐지든, 생성 모델이든 **모든 딥러닝 프로젝트에서 동일하게** 반복됩니다.

## 핵심 개념

### 1. MNIST 데이터셋 — 딥러닝의 기본기

> 💡 **비유**: MNIST는 피아노를 배울 때 치는 **"나비야"** 같은 곡입니다. 너무 쉬워 보이지만, 이 곡을 통해 손가락 위치, 박자 맞추기, 악보 읽기 같은 기본기를 익힐 수 있죠.

| 항목 | 내용 |
|------|------|
| 이미지 수 | 학습 60,000장 + 테스트 10,000장 |
| 이미지 크기 | 28×28 픽셀 (흑백) |
| 클래스 | 0~9 숫자 (10개 클래스) |
| 특징 | 중앙 정렬, 크기 정규화 완료 |

MNIST는 1998년 **얀 르쿤(Yann LeCun)**이 미국 국립표준기술연구소(NIST)의 데이터를 재가공하여 만들었습니다. 원래 NIST 데이터는 학습셋이 인구조사국 직원의 글씨, 테스트셋이 고등학생 글씨여서 분포가 달랐는데, 르쿤이 이를 섞어서(**M**odified NIST) 학습과 평가에 적합하도록 만들었습니다.

### 2. 딥러닝 파이프라인 — 전체 흐름

모든 딥러닝 프로젝트는 이 5단계를 따릅니다:

> **1. 데이터 준비** → **2. 모델 정의** → **3. 학습** → **4. 평가** → **5. 저장/배포**

이 섹션에서 1~4단계를 모두 경험합니다. 각 단계를 하나씩 살펴보겠습니다.

### 3. Step 1: 데이터 준비

PyTorch는 `torchvision.datasets`로 MNIST를 자동 다운로드하고, `DataLoader`로 배치 단위 공급을 처리합니다.

**전처리가 중요한 이유**: 원본 픽셀 값(0~255)을 그대로 쓰면 학습이 불안정합니다. 텐서로 변환(0~1)하고, 평균/표준편차로 정규화하면 학습이 훨씬 빠르고 안정적입니다.

### 4. Step 2: 모델 설계

MNIST는 28×28 흑백 이미지로 매우 작기 때문에, 간단한 CNN으로도 99% 이상의 정확도를 달성할 수 있습니다. 여기서는 [Chapter 04](../04-cnn-fundamentals/01-convolution.md)에서 배운 Conv-BN-ReLU-Pool 패턴을 활용합니다.

### 5. Step 3~4: 학습과 평가

학습 루프의 핵심은 다음 4줄입니다:
1. **순전파**: 입력 → 모델 → 예측
2. **손실 계산**: 예측과 정답의 차이
3. **역전파**: 그래디언트 계산
4. **파라미터 업데이트**: 옵티마이저가 가중치 조정

이 4줄이 [역전파](../03-deep-learning-basics/03-backpropagation.md)와 [옵티마이저](../03-deep-learning-basics/04-loss-optimizer.md)에서 배운 이론의 실제 구현입니다.

## 실습: 완전한 MNIST 분류 프로젝트

### 데이터 준비

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# === 하이퍼파라미터 ===
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 데이터 전처리 ===
transform = transforms.Compose([
    transforms.ToTensor(),                 # [0,255] → [0,1] 텐서 변환
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균/표준편차로 정규화
])

# === 데이터셋 로드 (자동 다운로드) ===
train_dataset = datasets.MNIST(root='./data', train=True,
                                download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                               download=True, transform=transform)

# === DataLoader: 배치 단위로 데이터 공급 ===
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True)   # 학습: 매 에포크 순서 섞기
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False)   # 테스트: 순서 유지

print(f"학습 데이터: {len(train_dataset):,}장")
print(f"테스트 데이터: {len(test_dataset):,}장")
print(f"배치 수: {len(train_loader)}개 (배치 크기 {BATCH_SIZE})")
print(f"디바이스: {DEVICE}")
```

### 모델 정의

```python
import torch.nn as nn

class MNISTNet(nn.Module):
    """MNIST용 간단한 CNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 블록 1: 1채널 → 32채널
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [B, 32, 28, 28]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # [B, 32, 14, 14]

            # 블록 2: 32채널 → 64채널
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # [B, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # [B, 64, 7, 7]
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                        # [B, 64, 1, 1]
            nn.Flatten(),                                    # [B, 64]
            nn.Dropout(0.25),
            nn.Linear(64, 10),                              # [B, 10]
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = MNISTNet().to(DEVICE)
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
```

### 학습 루프

```python
import torch.optim as optim

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 핵심 4줄!
        optimizer.zero_grad()              # 1. 이전 그래디언트 초기화
        outputs = model(images)            # 2. 순전파
        loss = criterion(outputs, labels)  # 3. 손실 계산
        loss.backward()                    # 4. 역전파
        optimizer.step()                   # 5. 파라미터 업데이트

        # 통계 기록
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device):
    """모델 평가 (학습 X)"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 그래디언트 계산 비활성화 → 메모리 절약
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
        model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, DEVICE)

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

# 예상 출력:
# Epoch  1/10 | Train Loss: 0.1482 Acc: 95.63% | Test Loss: 0.0412 Acc: 98.71%
# Epoch  5/10 | Train Loss: 0.0298 Acc: 99.06% | Test Loss: 0.0287 Acc: 99.12%
# Epoch 10/10 | Train Loss: 0.0152 Acc: 99.53% | Test Loss: 0.0251 Acc: 99.25%
```

### 결과 분석과 모델 저장

```python
import torch

# === 최종 평가 ===
test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
print(f"\n최종 테스트 정확도: {test_acc:.2f}%")

# === 모델 저장 ===
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("모델이 'mnist_cnn.pth'로 저장되었습니다.")

# === 모델 불러오기 ===
loaded_model = MNISTNet().to(DEVICE)
loaded_model.load_state_dict(torch.load('mnist_cnn.pth'))
loaded_model.eval()

# === 단일 이미지 예측 ===
sample_image, true_label = test_dataset[0]
sample_image = sample_image.unsqueeze(0).to(DEVICE)  # 배치 차원 추가

with torch.no_grad():
    output = loaded_model(sample_image)
    pred_label = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1).max().item()

print(f"정답: {true_label}, 예측: {pred_label}, 확신도: {confidence:.4f}")
```

## 더 깊이 알아보기

### MNIST의 역사와 의의

MNIST는 1998년 르쿤이 만든 이래 20년 넘게 머신러닝의 **표준 벤치마크**로 사용되어 왔습니다. "딥러닝의 초파리(Drosophila)"라고 불릴 정도로, 새로운 알고리즘을 테스트하는 첫 번째 데이터셋이었죠.

원래 NIST(국립표준기술연구소) 데이터는 미국 인구조사국 직원과 고등학생의 손글씨를 모은 것인데, 학습/테스트 세트의 분포가 달라 기계학습 실험에 부적합했습니다. 르쿤이 두 그룹을 적절히 섞어 **M**odified NIST, 즉 MNIST를 만들었습니다.

하지만 최근에는 MNIST가 "너무 쉽다"는 비판도 있습니다. 간단한 모델로도 99%를 넘기기 때문이죠. 이에 패션 아이템 이미지로 만든 **Fashion-MNIST**(2017)가 드롭인(drop-in) 대체재로 제안되었고, 실무에서는 더 복잡한 데이터셋을 사용합니다.

> 💡 **알고 계셨나요?**: MNIST 홈페이지에는 수십 가지 모델의 에러율이 기록되어 있는데, 현재 최고 기록은 CNN 앙상블로 달성한 **0.17%** 에러율입니다. 10,000장 중 17장만 틀리는 수준이죠. 참고로 인간의 에러율도 약 0.2%로, 이미 기계가 인간을 넘어섰습니다.

### 흔히 하는 실수 체크리스트

학습이 잘 안 될 때 순서대로 확인하세요:

1. **`model.train()` / `model.eval()` 전환을 했는가?** — BatchNorm과 Dropout의 동작이 달라짐
2. **`optimizer.zero_grad()`를 매 스텝 호출했는가?** — 빠뜨리면 그래디언트가 누적됨
3. **데이터가 올바르게 정규화되었는가?** — Normalize 없이 학습하면 수렴이 느림
4. **`with torch.no_grad():`를 평가 시 사용했는가?** — 안 쓰면 메모리 낭비
5. **학습률(Learning Rate)이 적절한가?** — 너무 크면 발산, 너무 작으면 느림

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "MNIST에서 99%면 모델이 잘 된 것이다" — MNIST는 너무 쉬워서 99%는 사실 기본 수준입니다. 진짜 실력을 확인하려면 [CIFAR-10](./02-cifar10.md) 같은 더 어려운 데이터셋으로 시도해봐야 합니다.

> 🔥 **실무 팁**: 새 프로젝트를 시작할 때, 거대한 모델을 바로 학습시키지 마세요. 먼저 **작은 모델 + 작은 데이터 일부**로 파이프라인이 올바르게 동작하는지 확인한 뒤(sanity check), 본격적으로 규모를 키우세요. MNIST가 이 "sanity check" 역할을 할 수 있습니다.

> 🔥 **실무 팁**: `DataLoader`에서 `num_workers=4` (또는 CPU 코어 수)를 설정하면 데이터 로딩이 병렬화되어 학습 속도가 크게 향상됩니다. GPU가 데이터를 기다리는 시간을 줄이는 것이죠.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| MNIST | 28×28 흑백 손글씨 숫자 70,000장. 딥러닝의 "Hello, World" |
| 파이프라인 | 데이터 → 모델 → 학습 → 평가 → 저장. 모든 프로젝트의 공통 흐름 |
| DataLoader | 배치 단위로 데이터를 효율적으로 공급하는 PyTorch 유틸리티 |
| 학습 루프 핵심 | zero_grad → forward → loss → backward → step |
| model.eval() | 추론 시 반드시 호출. BatchNorm/Dropout 동작 변경 |
| torch.no_grad() | 평가 시 그래디언트 계산 비활성화 → 메모리 절약 |

## 다음 섹션 미리보기

MNIST는 28×28 흑백 이미지라 너무 쉽습니다. [CIFAR-10](./02-cifar10.md)에서는 32×32 **컬러** 이미지 10개 클래스를 분류하며, 실제 세계에 가까운 난이도의 문제를 풀어봅니다. 여기서부터 모델 설계와 하이퍼파라미터 튜닝의 진짜 고민이 시작됩니다.

## 참고 자료

- [MNIST 공식 페이지 (Yann LeCun)](https://yann.lecun.org/exdb/mnist/) - 원본 데이터셋과 역대 벤치마크 기록
- [PyTorch MNIST Tutorial - Nextjournal](https://nextjournal.com/gkoehler/pytorch-mnist) - 단계별 MNIST CNN 구현 튜토리얼
- [Building a CNN for MNIST with PyTorch - Medium](https://medium.com/@ponchanon.rone/building-a-cnn-for-handwritten-digit-recognition-with-pytorch-a-step-by-step-guide-9df1dcb1092d) - 2024년 최신 PyTorch MNIST 가이드
- [MNIST Database - Wikipedia](https://en.wikipedia.org/wiki/MNIST_database) - MNIST의 역사와 NIST 데이터셋 기원
