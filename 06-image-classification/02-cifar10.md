# CIFAR-10 분류

> 컬러 이미지 분류 도전

## 개요

[MNIST](./01-mnist.md)에서 28×28 흑백 이미지로 딥러닝 파이프라인을 익혔다면, 이제 **진짜 도전**을 시작할 차례입니다. CIFAR-10은 32×32 **컬러** 이미지 10개 클래스를 분류하는 데이터셋으로, MNIST보다 훨씬 복잡하고 실전에 가깝습니다. 여기서부터 모델 설계, 데이터 증강, 학습률 스케줄링 같은 **실전 테크닉**이 정확도를 좌우하기 시작합니다.

**선수 지식**: [MNIST 손글씨 분류](./01-mnist.md), [CNN 아키텍처](../05-cnn-architectures/01-lenet-alexnet.md)
**학습 목표**:
- CIFAR-10 데이터셋의 특성과 MNIST와의 차이를 이해한다
- 컬러 이미지에 적합한 CNN 모델을 설계할 수 있다
- 데이터 증강과 학습률 스케줄링으로 성능을 끌어올릴 수 있다

## 왜 알아야 할까?

MNIST에서 99% 정확도를 달성했다고 좋아하기엔 이릅니다. MNIST는 **너무 쉬운** 데이터셋이거든요. 간단한 모델로도 금방 99%를 넘기기 때문에, 모델 설계의 좋고 나쁨을 구분하기 어렵습니다.

CIFAR-10은 다릅니다. 단순한 CNN으로는 70~80%대에 머물고, 90%를 넘기려면 아키텍처 설계, 정규화, 데이터 증강, 학습률 전략 등 **실전 기법**을 총동원해야 합니다. 실무에서 만나는 이미지 분류 문제의 난이도가 CIFAR-10에 훨씬 가깝기 때문에, 여기서 익히는 테크닉이 실제 프로젝트에 곧바로 적용됩니다.

## 핵심 개념

### 1. CIFAR-10 데이터셋 — MNIST의 다음 단계

> 💡 **비유**: MNIST가 피아노 입문곡 "나비야"였다면, CIFAR-10은 "엘리제를 위하여"입니다. 곡 자체는 짧지만, 양손 협응, 페달 사용, 강약 조절 같은 **진짜 실력**이 드러나는 곡이죠.

CIFAR-10은 **Canadian Institute For Advanced Research**에서 이름을 따온 데이터셋입니다. 2009년 토론토 대학의 **알렉스 크리제프스키(Alex Krizhevsky)**, 비노드 나이어(Vinod Nair), **제프리 힌튼(Geoffrey Hinton)**이 만들었습니다.

| 항목 | MNIST | CIFAR-10 |
|------|-------|----------|
| 이미지 크기 | 28×28 | 32×32 |
| 채널 수 | 1 (흑백) | 3 (RGB 컬러) |
| 클래스 | 숫자 0~9 | 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭 |
| 학습 데이터 | 60,000장 | 50,000장 |
| 테스트 데이터 | 10,000장 | 10,000장 |
| 간단한 CNN 정확도 | ~99% | ~75% |
| 실전 난이도 | ⭐ | ⭐⭐⭐ |

같은 10-클래스 분류인데, 왜 CIFAR-10이 훨씬 어려울까요?

### 2. MNIST vs CIFAR-10 — 왜 이렇게 다를까?

세 가지 핵심적인 차이가 난이도를 결정합니다:

**첫째, 컬러의 복잡성입니다.** MNIST는 흑백이라 픽셀당 숫자 1개(밝기)만 있지만, CIFAR-10은 RGB 3채널이므로 픽셀당 숫자 3개입니다. 모델이 처리해야 할 정보량이 3배로 늘어나는 거죠.

**둘째, 클래스 내 다양성이 큽니다.** 숫자 "3"은 누가 써도 비슷하게 생겼지만, "고양이"는 색깔, 자세, 배경이 천차만별입니다. 흰 고양이, 검은 고양이, 앉은 고양이, 누운 고양이... 모델은 이 다양한 형태에서 "고양이다움"을 추출해야 합니다.

**셋째, 해상도가 매우 낮습니다.** 32×32는 사실 아주 작은 이미지입니다. 사람이 봐도 뭔지 헷갈리는 경우가 많죠. 모델은 이 적은 정보만으로 10개 클래스를 구분해야 합니다.

> ⚠️ **흔한 오해**: "CIFAR-10은 쉬운 데이터셋이다" — 사실 32×32 해상도에서 90% 이상 정확도를 달성하려면 상당한 노력이 필요합니다. 2011년 CIFAR-10의 최고 정확도(state-of-the-art)가 80.5%였을 정도로, 오랜 기간 도전적인 벤치마크였습니다.

### 3. 모델 설계 전략 — 더 깊고, 더 넓게

MNIST에서는 2블록 CNN으로 충분했지만, CIFAR-10에서는 **더 깊은 네트워크**가 필요합니다. 컬러 이미지의 복잡한 패턴을 포착하려면 합성곱 레이어를 더 쌓아야 하거든요.

설계할 때 기억할 핵심 원칙이 있습니다:

- **채널 수를 점진적으로 증가**: 32 → 64 → 128처럼 깊어질수록 더 많은 특징을 추출
- **공간 크기를 점진적으로 감소**: 풀링으로 32×32 → 16×16 → 8×8로 줄여가기
- **Conv-BN-ReLU 패턴 반복**: [배치 정규화](../04-cnn-fundamentals/03-batch-normalization.md)는 학습 안정화의 핵심
- **Dropout으로 과적합 방지**: [정규화 기법](../04-cnn-fundamentals/04-regularization.md)에서 배운 것처럼

> 💡 **비유**: 모델 설계는 망원경을 만드는 것과 비슷합니다. 렌즈(합성곱 레이어)를 하나만 쓰면 흐릿하게 보이지만, 여러 개를 잘 배치하면 먼 곳의 세밀한 디테일까지 잡아낼 수 있죠. CIFAR-10은 MNIST보다 더 많은 "렌즈"가 필요한 셈입니다.

### 4. 데이터 증강 — 적은 데이터로 더 많이 학습하기

50,000장으로 복잡한 컬러 이미지를 분류하기엔 데이터가 부족할 수 있습니다. **데이터 증강(Data Augmentation)**은 기존 이미지를 변형하여 모델이 더 다양한 경우를 학습하게 만드는 기법입니다.

CIFAR-10에서 효과적인 증강 기법:

| 기법 | 설명 | 효과 |
|------|------|------|
| 수평 뒤집기(Horizontal Flip) | 좌우 반전 | 대칭적 객체에 효과적 |
| 랜덤 크롭(Random Crop) | 패딩 후 랜덤 위치에서 잘라내기 | 위치 불변성 학습 |
| 색상 지터(Color Jitter) | 밝기, 대비, 채도 랜덤 변경 | 조명 변화에 강건 |
| 랜덤 회전(Random Rotation) | 소폭 회전 | 기울어진 객체 대응 |

데이터 증강은 **학습 데이터에만** 적용하고, 테스트 데이터는 원본 그대로 평가합니다. 이건 시험 공부할 때 다양한 문제를 풀지만, 실제 시험에서는 정해진 문제를 정확히 풀어야 하는 것과 같습니다.

### 5. 학습률 스케줄링 — 가속과 감속의 기술

처음에는 큰 학습률로 빠르게 탐색하고, 후반에는 작은 학습률로 세밀하게 수렴하는 전략이 효과적입니다. 이를 **학습률 스케줄링(Learning Rate Scheduling)**이라고 합니다.

> 💡 **비유**: 새로운 동네에서 맛집을 찾을 때를 생각해보세요. 처음에는 넓은 범위를 빠르게 돌아다니며(큰 학습률) 후보를 탐색하고, 유망한 곳을 찾으면 그 근방을 천천히 꼼꼼하게(작은 학습률) 살피는 거죠.

PyTorch에서 자주 쓰이는 스케줄러:

| 스케줄러 | 동작 방식 | 특징 |
|----------|-----------|------|
| StepLR | 정해진 주기마다 학습률 감소 | 간단하고 직관적 |
| CosineAnnealingLR | 코사인 함수를 따라 부드럽게 감소 | 안정적인 수렴 |
| OneCycleLR | 워밍업 후 감소하는 원사이클 정책 | 빠른 수렴, 높은 성능 |

## 실습: 완전한 CIFAR-10 분류 프로젝트

### Step 1: 데이터 준비 (증강 포함)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# === 하이퍼파라미터 ===
BATCH_SIZE = 128
LEARNING_RATE = 0.1
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 데이터 증강 + 정규화 ===
# 학습: 증강 적용 (랜덤 크롭, 수평 뒤집기)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       # 4픽셀 패딩 후 32x32 랜덤 크롭
    transforms.RandomHorizontalFlip(),           # 50% 확률로 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize(                        # CIFAR-10 평균/표준편차
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

# 테스트: 증강 없이 원본 그대로
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

# === 데이터셋 로드 ===
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                  download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)

# 클래스 이름 확인
classes = ('비행기', '자동차', '새', '고양이', '사슴',
           '개', '개구리', '말', '배', '트럭')

print(f"학습 데이터: {len(train_dataset):,}장")
print(f"테스트 데이터: {len(test_dataset):,}장")
print(f"클래스: {classes}")
print(f"이미지 크기: 3×32×32 (RGB)")
print(f"디바이스: {DEVICE}")
```

### Step 2: CNN 모델 설계

```python
class CIFAR10Net(nn.Module):
    """CIFAR-10용 CNN — VGG 스타일 블록 구조"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # === 블록 1: 3 → 64채널, 32x32 유지 ===
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                         # → [B, 64, 16, 16]
            nn.Dropout2d(0.1),

            # === 블록 2: 64 → 128채널 ===
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                         # → [B, 128, 8, 8]
            nn.Dropout2d(0.2),

            # === 블록 3: 128 → 256채널 ===
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                         # → [B, 256, 4, 4]
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # → [B, 256, 1, 1]
            nn.Flatten(),                             # → [B, 256]
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),              # → [B, 10]
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CIFAR10Net().to(DEVICE)

# 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"모델 파라미터 수: {total_params:,}")
# 출력: 약 1,200,000개 — MNIST 모델(약 20,000개)의 60배
```

MNIST 모델과 비교하면 블록이 3개로 늘었고, 각 블록에 합성곱이 2개씩 들어갑니다. [VGG](../05-cnn-architectures/02-vgg-googlenet.md)에서 배운 "작은 필터를 깊게 쌓는" 전략이죠. 채널 수도 64 → 128 → 256으로 점진적으로 증가하여 더 복잡한 특징을 추출합니다.

### Step 3: 학습 (스케줄러 포함)

```python
# === 손실 함수, 옵티마이저, 스케줄러 ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                      momentum=0.9, weight_decay=5e-4)

# 코사인 어닐링: 학습률을 부드럽게 0까지 감소
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

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
    """모델 평가"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

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
best_acc = 0
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, DEVICE)
    scheduler.step()  # 학습률 업데이트

    # 최고 성능 모델 저장
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'cifar10_best.pth')

    current_lr = optimizer.param_groups[0]['lr']
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"LR: {current_lr:.5f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

print(f"\n최고 테스트 정확도: {best_acc:.2f}%")

# 예상 출력 (GPU 기준):
# Epoch  1/50 | LR: 0.09990 | Train Loss: 1.4921 Acc: 44.83% | Test Loss: 1.0814 Acc: 61.32%
# Epoch 10/50 | LR: 0.09045 | Train Loss: 0.4835 Acc: 83.12% | Test Loss: 0.5124 Acc: 82.76%
# Epoch 20/50 | LR: 0.06545 | Train Loss: 0.2987 Acc: 89.62% | Test Loss: 0.3821 Acc: 87.90%
# Epoch 30/50 | LR: 0.03455 | Train Loss: 0.1876 Acc: 93.41% | Test Loss: 0.3185 Acc: 90.25%
# Epoch 40/50 | LR: 0.00955 | Train Loss: 0.1102 Acc: 96.22% | Test Loss: 0.2847 Acc: 91.58%
# Epoch 50/50 | LR: 0.00000 | Train Loss: 0.0812 Acc: 97.34% | Test Loss: 0.2753 Acc: 92.10%
# 최고 테스트 정확도: 92.10%
```

### Step 4: 클래스별 정확도 분석

```python
def class_accuracy(model, loader, classes, device):
    """클래스별 정확도 분석"""
    model.eval()
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    print("\n클래스별 정확도:")
    print("-" * 30)
    for i, cls_name in enumerate(classes):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"  {cls_name:6s}: {acc:5.1f}% ({class_correct[i]:4d}/{class_total[i]})")

class_accuracy(model, test_loader, classes, DEVICE)

# 예상 출력:
# 클래스별 정확도:
# ------------------------------
#   비행기: 93.2% ( 932/1000)
#   자동차: 96.1% ( 961/1000)
#   새    : 88.7% ( 887/1000)
#   고양이: 83.5% ( 835/1000)
#   사슴  : 91.8% ( 918/1000)
#   개    : 85.4% ( 854/1000)
#   개구리: 95.3% ( 953/1000)
#   말    : 93.7% ( 937/1000)
#   배    : 95.0% ( 950/1000)
#   트럭  : 94.8% ( 948/1000)
```

결과를 보면 **고양이**와 **개**의 정확도가 가장 낮습니다. 왜 그럴까요? 둘 다 네 발 달린 동물이라 32×32 해상도에서는 실루엣이 비슷하거든요. 이런 **혼동 패턴**을 파악하는 것이 모델 개선의 출발점입니다.

## 더 깊이 알아보기

### CIFAR-10의 탄생 이야기

CIFAR-10은 원래 **8천만 장의 작은 이미지 데이터셋(80 Million Tiny Images)**의 부분집합으로 만들어졌습니다. 토론토 대학의 알렉스 크리제프스키가 2009년 기술 보고서 *"Learning Multiple Layers of Features from Tiny Images"*에서 발표했죠.

흥미로운 점은, 크리제프스키가 바로 3년 뒤인 2012년에 **AlexNet**으로 ImageNet 대회를 석권한 장본인이라는 것입니다. CIFAR-10을 연구하며 쌓은 경험이 딥러닝 혁명의 씨앗이 된 셈이죠. 그의 지도교수는 딥러닝의 대부 **제프리 힌튼(Geoffrey Hinton)**이었고, 힌튼은 이 공로로 2024년 노벨 물리학상을 수상했습니다.

참고로, 원본 80 Million Tiny Images 데이터셋은 2020년에 인종차별적·공격적 라벨이 발견되어 공식 철회되었습니다. 하지만 CIFAR-10/100은 별도로 정제된 라벨을 사용하기 때문에 여전히 안전하게 사용 가능합니다.

> 💡 **알고 계셨나요?**: CIFAR-10의 "CIFAR"는 **Canadian Institute For Advanced Research**의 약자입니다. 이 연구소는 힌튼의 딥러닝 연구를 초기부터 지원한 곳으로, 딥러닝의 역사에서 매우 중요한 역할을 했습니다. 캐나다가 AI 강국이 된 배경에는 CIFAR의 선구적인 지원이 있었죠.

### MNIST에서 CIFAR-10으로 올 때 달라지는 것

| 항목 | MNIST | CIFAR-10 |
|------|-------|----------|
| 옵티마이저 | Adam(lr=1e-3) | SGD+Momentum(lr=0.1) |
| 에포크 | 10이면 충분 | 50~200 필요 |
| 데이터 증강 | 거의 불필요 | 필수 (정확도 5%↑) |
| 학습률 스케줄링 | 없어도 됨 | 필수 (정확도 2~3%↑) |
| 모델 깊이 | 2블록 | 3블록 이상 |
| Weight Decay | 없어도 됨 | 필수 (5e-4) |

여기서 주목할 점은 **옵티마이저 선택**입니다. MNIST에서는 Adam이 편리했지만, CIFAR-10에서는 SGD+Momentum이 더 좋은 일반화 성능을 보이는 경우가 많습니다. 이는 SGD가 더 "평평한" 최솟값을 찾는 경향이 있기 때문인데요, [손실 함수와 옵티마이저](../03-deep-learning-basics/04-loss-optimizer.md)에서 배운 개념의 실전적 연장입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "CIFAR-10에서 성능이 안 나오면 모델을 더 크게 만들어야 한다" — 사실 CIFAR-10은 이미지가 32×32로 매우 작아서, 너무 큰 모델은 오히려 과적합됩니다. **데이터 증강**과 **정규화**가 모델 크기를 키우는 것보다 효과적인 경우가 많습니다.

> 🔥 **실무 팁**: CIFAR-10의 평균(mean)과 표준편차(std) 값 `(0.4914, 0.4822, 0.4465)`, `(0.2470, 0.2435, 0.2616)`은 학습 데이터 전체에서 계산된 값입니다. 자신만의 데이터셋을 쓸 때는 반드시 해당 데이터의 평균/표준편차를 직접 계산해서 사용하세요.

> 🔥 **실무 팁**: 학습 정확도는 97%인데 테스트 정확도는 92%? 이 **5% 갭**은 과적합의 신호입니다. Dropout 비율을 높이거나, 데이터 증강을 더 강하게 적용하거나, Weight Decay를 키워보세요. 갭을 줄이는 것이 전체 성능을 높이는 지름길입니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| CIFAR-10 | 32×32 RGB 컬러 이미지 60,000장, 10클래스. MNIST보다 현실에 가까운 난이도 |
| 데이터 증강 | RandomCrop, HorizontalFlip 등으로 학습 데이터 다양성 확보 → 정확도 5%↑ |
| 학습률 스케줄링 | CosineAnnealing, OneCycleLR 등으로 학습률을 동적 조절 → 안정적 수렴 |
| SGD+Momentum | CIFAR-10에서는 Adam보다 SGD+Momentum이 더 좋은 일반화 성능 |
| 클래스별 분석 | 전체 정확도뿐 아니라 클래스별 정확도를 확인해야 약점을 파악 가능 |
| 과적합 갭 | Train-Test 정확도 차이를 모니터링하고 정규화로 줄이기 |

## 다음 섹션 미리보기

92%도 훌륭하지만, 더 높은 성능을 원한다면? 처음부터 학습할 필요 없이, ImageNet에서 이미 학습된 모델의 지식을 **빌려오는** 방법이 있습니다. [전이 학습](./03-transfer-learning.md)에서는 사전 학습 모델을 활용해 적은 데이터와 짧은 학습 시간으로도 높은 성능을 달성하는 방법을 배웁니다.

## 참고 자료

- [PyTorch 공식 CIFAR-10 튜토리얼](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) - PyTorch로 CIFAR-10 분류기를 만드는 공식 가이드
- [CIFAR-10 공식 페이지 (Alex Krizhevsky)](https://www.cs.toronto.edu/~kriz/cifar.html) - 데이터셋 원본과 기술 보고서
- [CIFAR-10 Wikipedia](https://en.wikipedia.org/wiki/CIFAR-10) - 데이터셋의 역사와 벤치마크 기록
- [94% on CIFAR-10 in 3.29 Seconds on a Single GPU (2024)](https://arxiv.org/html/2404.00498v2) - 초고속 CIFAR-10 학습 기법 연구
- [Deep Dive into Image Classification with PyTorch: CIFAR-10 Tutorial](https://medium.com/@saitejamummadi/deep-dive-into-image-classification-with-pytorch-a-cifar-10-tutorial-4151f9c4c7b1) - 단계별 CIFAR-10 CNN 구현 가이드
