# 역전파 알고리즘

> 경사하강법과 역전파의 수학적 이해

## 개요

신경망이 "학습"한다는 것은 무엇일까요? 정답과 예측의 차이(오차)를 줄이기 위해 가중치를 조금씩 조정하는 것입니다. 이때 "어떤 가중치를 얼마나 조정해야 하는가"를 알려주는 알고리즘이 **역전파(Backpropagation)**입니다. 딥러닝의 심장이라 할 수 있습니다.

**선수 지식**: [신경망의 구조](./01-neural-network.md) — 순전파, 가중치 / [활성화 함수](./02-activation-functions.md) — ReLU, 기울기
**학습 목표**:
- 경사하강법(Gradient Descent)의 직관을 이해한다
- 역전파가 연쇄 법칙(Chain Rule)으로 기울기를 계산하는 과정을 이해한다
- PyTorch의 자동미분(Autograd)을 사용할 수 있다

## 왜 알아야 할까?

모든 딥러닝 프레임워크(PyTorch, TensorFlow)가 역전파를 자동으로 처리해 주지만, **왜 학습이 안 되는지**, **왜 기울기가 폭발하는지** 등의 문제를 디버깅하려면 원리를 알아야 합니다. 역전파를 이해하면 학습률 설정, 기울기 클리핑, 배치 정규화 같은 기법이 왜 필요한지도 자연스럽게 이해됩니다.

## 핵심 개념

### 1. 학습의 전체 흐름

> 💡 **비유**: 역전파는 **시험 오답노트**를 만드는 과정입니다. 시험(순전파)을 보고 → 정답과 비교해 점수를 매기고(손실 계산) → 어떤 문제에서 얼마나 틀렸는지 분석하고(기울기 계산) → 취약한 부분을 집중 보완합니다(가중치 업데이트).

전체 학습 사이클:

1. **순전파 (Forward Pass)**: 입력 → 레이어들 → 예측값 출력
2. **손실 계산 (Loss)**: 예측값과 정답의 차이를 숫자로 측정
3. **역전파 (Backward Pass)**: 손실을 줄이려면 각 가중치를 어떻게 바꿔야 하는지 계산
4. **가중치 업데이트**: 계산된 방향으로 가중치를 조금씩 조정
5. **반복**: 1~4를 수천~수만 번 반복

### 2. 경사하강법(Gradient Descent) — 가장 낮은 곳 찾기

> 💡 **비유**: 안개 낀 산에서 **가장 낮은 골짜기**를 찾아야 한다고 상상하세요. 앞이 안 보이니 **발밑의 경사(기울기)**만 느낄 수 있습니다. "가장 가파르게 내려가는 방향"으로 한 걸음씩 내딛으면, 결국 골짜기(최소 손실)에 도착합니다. 이것이 경사하강법입니다.

**가중치 업데이트 공식:**

> **w_new = w_old − 학습률 × 기울기**

| 요소 | 의미 | 비유 |
|------|------|------|
| **기울기 (Gradient)** | 현재 위치에서 손실이 증가하는 방향 | 발밑 경사의 방향 |
| **학습률 (Learning Rate)** | 한 번에 얼마나 움직일지 | 걸음 크기 |
| **− (마이너스)** | 기울기의 반대 방향으로 이동 | 오르막 반대 = 내리막 |

> **학습률이 너무 크면?** 골짜기를 지나쳐 왔다 갔다 발산합니다.
> **학습률이 너무 작으면?** 도착까지 시간이 너무 오래 걸립니다.

```python
import torch

# 경사하강법 직접 구현 (1D 예시)
# 목표: y = (x - 3)² 에서 최소점 x = 3 찾기

x = torch.tensor(10.0, requires_grad=True)  # 시작점: x = 10
learning_rate = 0.1

for step in range(20):
    y = (x - 3) ** 2   # 손실 함수
    y.backward()        # 기울기 계산 (dy/dx)

    with torch.no_grad():
        x -= learning_rate * x.grad  # 가중치 업데이트
        x.grad.zero_()               # 기울기 초기화

    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, 손실 = {(x.item()-3)**2:.4f}")
```

### 3. 연쇄 법칙(Chain Rule) — 역전파의 수학적 핵심

> 💡 **비유**: 도미노가 줄지어 쓰러질 때, 마지막 도미노가 쓰러진 원인을 추적하려면 **하나씩 역방향으로** 거슬러 올라가야 합니다. 각 도미노가 다음 도미노에 미친 영향을 곱해가면 **첫 번째 도미노의 영향**을 알 수 있습니다.

합성 함수 f(g(x))의 미분:

> **df/dx = (df/dg) × (dg/dx)**

신경망에서는 이것이 **레이어를 거슬러 올라가며** 적용됩니다:

> 출력층의 기울기 → 은닉층2의 기울기 → 은닉층1의 기울기 → 입력층

```python
import torch

# 연쇄 법칙 확인
x = torch.tensor(2.0, requires_grad=True)

# y = 3x + 1, z = y², loss = z  (합성 함수)
y = 3 * x + 1      # y = 7
z = y ** 2          # z = 49

z.backward()

# dz/dx = dz/dy × dy/dx = 2y × 3 = 2(7) × 3 = 42
print(f"x = {x.item()}")
print(f"y = 3x + 1 = {y.item()}")
print(f"z = y² = {z.item()}")
print(f"dz/dx (자동 계산) = {x.grad.item()}")
print(f"dz/dx (수동 계산) = {2 * y.item() * 3}")  # 42.0
```

### 4. PyTorch의 자동미분(Autograd)

PyTorch는 연산을 기록하는 **계산 그래프(Computational Graph)**를 자동으로 만들고, `.backward()`를 호출하면 연쇄 법칙을 자동 적용하여 모든 파라미터의 기울기를 계산합니다.

```python
import torch
import torch.nn as nn

# 간단한 모델
model = nn.Linear(3, 1)  # 입력 3, 출력 1

# 입력과 정답
x = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([[5.0]])

# 순전파
prediction = model(x)
loss = (prediction - target) ** 2  # MSE 손실

print(f"예측: {prediction.item():.4f}")
print(f"정답: {target.item():.4f}")
print(f"손실: {loss.item():.4f}")

# 역전파 — 기울기 자동 계산!
loss.backward()

# 각 파라미터의 기울기 확인
print(f"\n가중치 기울기: {model.weight.grad}")
print(f"편향 기울기: {model.bias.grad}")
```

### 5. 실제 학습 루프

```python
import torch
import torch.nn as nn

# 모델 정의
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 옵티마이저: 경사하강법 자동화
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 학습 데이터: y = 2x + 1
x_train = torch.linspace(-5, 5, 100).unsqueeze(1)
y_train = 2 * x_train + 1 + torch.randn_like(x_train) * 0.3

# 학습 루프
for epoch in range(100):
    # 1. 순전파
    prediction = model(x_train)
    loss = criterion(prediction, y_train)

    # 2. 역전파
    optimizer.zero_grad()  # 기울기 초기화 (중요!)
    loss.backward()        # 기울기 계산

    # 3. 가중치 업데이트
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: 손실 = {loss.item():.4f}")
```

> ⚠️ **`optimizer.zero_grad()` 를 빼먹으면?** PyTorch는 기울기를 누적합니다. 초기화 안 하면 이전 배치의 기울기가 쌓여서 학습이 제대로 안 됩니다. 가장 흔한 PyTorch 실수 중 하나입니다.

### 6. 기울기 문제들

| 문제 | 원인 | 증상 | 해결책 |
|------|------|------|--------|
| **기울기 소실** | Sigmoid/Tanh, 너무 깊은 망 | 앞쪽 레이어 학습 안 됨 | ReLU, 배치 정규화, Skip Connection |
| **기울기 폭발** | 기울기가 곱해지며 급증 | 손실이 NaN/Inf로 발산 | 기울기 클리핑, 학습률 감소 |

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **경사하강법** | 기울기의 반대 방향으로 가중치를 조금씩 이동하여 손실 최소화 |
| **학습률** | 한 번의 업데이트에서 얼마나 움직일지 결정. 너무 크면 발산, 너무 작으면 느림 |
| **연쇄 법칙** | 합성 함수의 미분을 단계별 곱셈으로 분해. 역전파의 수학적 기반 |
| **역전파** | 출력→입력 방향으로 기울기를 전파하여 모든 파라미터의 기울기 계산 |
| **Autograd** | PyTorch가 연산을 기록하고 `.backward()`로 자동 미분하는 시스템 |
| **zero_grad()** | 매 반복마다 기울기를 0으로 초기화. 빼먹으면 기울기 누적 |

## 다음 섹션 미리보기

역전파에서 "손실"을 계산한다고 했는데, 손실 함수에도 여러 종류가 있습니다. 그리고 경사하강법(SGD) 외에도 Adam 같은 더 똑똑한 옵티마이저가 있습니다. 다음 섹션 **[손실 함수와 옵티마이저](./04-loss-optimizer.md)**에서 자세히 알아봅니다.

## 참고 자료

- [Stanford CS231n - Backpropagation](https://cs231n.github.io/optimization-2/) - 기울기 계산의 직관적 설명과 시각화
- [IBM - What is Backpropagation?](https://www.ibm.com/think/topics/backpropagation) - 역전파 개념을 비즈니스 관점에서도 설명
- [GeeksforGeeks - Backpropagation in Neural Network](https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/) - 단계별 수식 설명과 Python 코드
- [Neural Networks and Deep Learning - Chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html) - Michael Nielsen의 역전파 완전 가이드
