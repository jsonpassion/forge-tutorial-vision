# 신경망의 구조

> 뉴런, 레이어, 가중치의 이해

## 개요

드디어 **딥러닝**의 세계에 들어왔습니다. 앞서 배운 필터와 특징점 검출은 사람이 규칙을 설계했지만, 딥러닝은 **컴퓨터가 스스로 규칙을 학습합니다.** 그 핵심이 바로 신경망(Neural Network)입니다. 이 섹션에서는 신경망을 이루는 가장 기본 단위인 뉴런, 레이어, 가중치를 하나씩 살펴봅니다.

**선수 지식**: [형태학적 연산](../02-classical-cv/05-morphology.md)까지의 기초 CV 지식, Python 기본 문법
**학습 목표**:
- 인공 뉴런이 무엇을 계산하는지 설명할 수 있다
- 레이어의 종류(입력, 은닉, 출력)를 구분할 수 있다
- 가중치와 편향의 역할을 이해한다
- PyTorch로 간단한 신경망을 구성할 수 있다

## 왜 알아야 할까?

CNN, Transformer, Diffusion 모델 등 이후 배울 모든 딥러닝 모델은 **신경망이라는 공통 골격** 위에 세워져 있습니다. 신경망의 기본을 확실히 이해하면, 이후 아무리 복잡한 모델을 만나도 "결국 뉴런과 레이어의 조합"이라는 사실을 기억할 수 있습니다.

## 핵심 개념

### 1. 인공 뉴런 — 신경망의 최소 단위

> 💡 **비유**: 인공 뉴런은 **투표 집계소**와 같습니다. 여러 사람(입력)이 각각 의견(값)을 내놓으면, 집계소는 각 의견에 **중요도(가중치)**를 곱해 합산한 뒤, 기준을 넘으면 "통과"(활성화), 못 넘으면 "탈락"을 선언합니다.

하나의 뉴런이 하는 일을 수식으로 쓰면:

> **출력 = 활성화 함수( w₁x₁ + w₂x₂ + ... + wₙxₙ + b )**

각 기호의 의미:

| 기호 | 이름 | 의미 |
|------|------|------|
| x₁, x₂, ... | **입력** | 뉴런에 들어오는 데이터 |
| w₁, w₂, ... | **가중치 (Weight)** | 각 입력의 중요도 |
| b | **편향 (Bias)** | 기준점을 조절하는 값 |
| Σ | **가중합** | 입력 × 가중치의 총합 |
| f() | **활성화 함수** | 결과를 비선형으로 변환 |

```python
import numpy as np

# 하나의 뉴런 직접 구현
def neuron(inputs, weights, bias):
    """입력에 가중치를 곱하고 편향을 더한 뒤 활성화(ReLU)"""
    weighted_sum = np.dot(inputs, weights) + bias
    output = max(0, weighted_sum)  # ReLU 활성화
    return output

# 예시: 3개의 입력을 받는 뉴런
inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([0.4, 0.5, -0.2])
bias = 0.1

result = neuron(inputs, weights, bias)
print(f"입력: {inputs}")
print(f"가중치: {weights}")
print(f"가중합: {np.dot(inputs, weights) + bias:.1f}")
print(f"출력 (ReLU): {result:.1f}")
```

### 2. 가중치(Weight)와 편향(Bias)

> 💡 **비유**: 가중치는 **볼륨 조절 노브**입니다. 입력 신호가 들어올 때 어떤 입력을 크게 키우고(중요), 어떤 입력을 줄이거나 뒤집을지(중요하지 않음) 결정합니다. 편향은 **전체 음량의 기본값**으로, 볼륨을 모두 0으로 놔도 최소한 이만큼은 소리가 나게(또는 안 나게) 만듭니다.

| 개념 | 역할 | 학습 과정 |
|------|------|----------|
| **가중치** | 각 입력의 중요도 조절 | 훈련 중 자동 업데이트 |
| **편향** | 활성화 기준점 조절 | 훈련 중 자동 업데이트 |

> **핵심**: 딥러닝의 "학습"이란, 이 가중치와 편향을 데이터를 보며 **최적의 값으로 조정하는 과정**입니다.

### 3. 레이어 — 뉴런의 그룹

> 💡 **비유**: 공장의 **조립 라인**을 생각하세요. 원자재(입력)가 들어오면 1번 라인(입력 레이어)에서 기본 처리를 하고, 2번·3번 라인(은닉 레이어)에서 점점 정교하게 가공하고, 마지막 라인(출력 레이어)에서 완성품(결과)이 나옵니다.

신경망은 뉴런을 **층(Layer)**으로 묶어 순서대로 연결합니다.

| 레이어 | 위치 | 역할 | 예시 |
|--------|------|------|------|
| **입력 레이어** | 맨 앞 | 원본 데이터를 받아들임 | 28×28 이미지 = 784개 뉴런 |
| **은닉 레이어** | 중간 | 데이터에서 패턴을 추출 | 128개, 64개 뉴런 등 |
| **출력 레이어** | 맨 뒤 | 최종 결과 생성 | 10개 뉴런 (숫자 0~9 분류) |

> "**딥**러닝"의 "딥"은 **은닉 레이어가 여러 개**라는 뜻입니다. 레이어가 깊을수록 더 복잡한 패턴을 학습할 수 있습니다.

### 4. 순전파(Forward Pass) — 데이터가 흐르는 방향

데이터가 입력 레이어에서 출력 레이어까지 **앞으로만 흐르는** 과정을 순전파라고 합니다.

> 입력 데이터 → 레이어 1 (가중합 + 활성화) → 레이어 2 (가중합 + 활성화) → ... → 출력

```python
import numpy as np

def simple_network(x):
    """2개의 은닉 레이어를 가진 간단한 신경망"""
    # 레이어 1: 입력 3개 → 뉴런 4개
    W1 = np.random.randn(3, 4) * 0.5
    b1 = np.zeros(4)
    h1 = np.maximum(0, x @ W1 + b1)  # ReLU 활성화

    # 레이어 2: 뉴런 4개 → 뉴런 2개
    W2 = np.random.randn(4, 2) * 0.5
    b2 = np.zeros(2)
    h2 = np.maximum(0, h1 @ W2 + b2)  # ReLU 활성화

    # 출력 레이어: 뉴런 2개 → 출력 1개
    W3 = np.random.randn(2, 1) * 0.5
    b3 = np.zeros(1)
    output = h1 @ W2 @ W3  # 마지막은 활성화 없이

    return output

x = np.array([1.0, 2.0, 3.0])
result = simple_network(x)
print(f"입력: {x}")
print(f"출력: {result}")
```

### 5. PyTorch로 신경망 만들기

PyTorch에서는 `nn.Module`을 상속받아 신경망을 정의합니다.

```python
import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 레이어 정의
        self.layer1 = nn.Linear(784, 128)  # 입력 784 → 은닉 128
        self.layer2 = nn.Linear(128, 64)   # 은닉 128 → 은닉 64
        self.layer3 = nn.Linear(64, 10)    # 은닉 64 → 출력 10
        self.relu = nn.ReLU()              # 활성화 함수

    def forward(self, x):
        """순전파: 데이터가 레이어를 통과하는 과정"""
        x = self.relu(self.layer1(x))  # 레이어1 + 활성화
        x = self.relu(self.layer2(x))  # 레이어2 + 활성화
        x = self.layer3(x)             # 출력 (활성화 없음)
        return x

# 모델 생성
model = SimpleNetwork()
print(model)

# 파라미터(가중치+편향) 수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"\n총 파라미터 수: {total_params:,}")
```

```python
import torch

# 모델에 데이터 통과시키기
model = SimpleNetwork()

# 가짜 입력: 배치 크기 1, 특성 784개
x = torch.randn(1, 784)

# 순전파
output = model(x)
print(f"입력 shape: {x.shape}")    # [1, 784]
print(f"출력 shape: {output.shape}")  # [1, 10]
print(f"출력 값: {output.data}")

# 각 레이어의 가중치 shape 확인
for name, param in model.named_parameters():
    print(f"{name:20s} | shape: {str(param.shape):15s} | 파라미터 수: {param.numel():,}")
```

### 6. 파라미터 수 계산하기

`nn.Linear(입력, 출력)`의 파라미터 수 = (입력 × 출력) + 출력(편향)

| 레이어 | 구조 | 가중치 | 편향 | 합계 |
|--------|------|--------|------|------|
| layer1 | 784 → 128 | 784×128 = 100,352 | 128 | 100,480 |
| layer2 | 128 → 64 | 128×64 = 8,192 | 64 | 8,256 |
| layer3 | 64 → 10 | 64×10 = 640 | 10 | 650 |
| **합계** | | | | **109,386** |

> 이 간단한 3개 레이어 네트워크에도 **약 11만 개의 파라미터**가 있습니다. 현대 딥러닝 모델은 수십억 개의 파라미터를 가집니다.

## 실습: 직접 해보기

### nn.Linear의 내부 들여다보기

```python
import torch
import torch.nn as nn

# 하나의 Linear 레이어 생성
linear = nn.Linear(in_features=3, out_features=2)

# 가중치와 편향 확인
print(f"가중치 shape: {linear.weight.shape}")  # [2, 3]
print(f"가중치 값:\n{linear.weight.data}")
print(f"\n편향 shape: {linear.bias.shape}")      # [2]
print(f"편향 값: {linear.bias.data}")

# 직접 계산해보기
x = torch.tensor([1.0, 2.0, 3.0])
manual_output = x @ linear.weight.T + linear.bias
auto_output = linear(x)

print(f"\n수동 계산: {manual_output.data}")
print(f"Linear 결과: {auto_output.data}")
print(f"동일한가? {torch.allclose(manual_output, auto_output)}")
```

## 더 깊이 알아보기

### 신경망의 파란만장한 역사 — AI 겨울과 부활

> 💡 **알고 계셨나요?**: 신경망의 역사는 드라마보다 극적입니다.

**1958년 — 퍼셉트론의 탄생과 과대 광고**

프랭크 로젠블랫(Frank Rosenblatt)이 **퍼셉트론(Perceptron)**을 발명했을 때, 뉴욕 타임즈는 *"New Navy Device Learns By Doing"*이라는 헤드라인으로 대서특필했습니다. 미 해군 연구소에서 만든 이 기계가 스스로 학습한다니, 세상이 흥분했죠. "곧 인간처럼 생각하는 기계가 나올 것"이라는 기대감이 하늘을 찔렀습니다.

**1969년 — 마빈 민스키의 냉수**

MIT의 마빈 민스키(Marvin Minsky)와 시모어 페퍼트(Seymour Papert)가 *Perceptrons*라는 책을 출판하면서 분위기가 급반전됩니다. 이 책은 단층 퍼셉트론이 **XOR 문제조차 풀 수 없다**는 것을 수학적으로 증명했거든요. "직선 하나로는 XOR을 분리할 수 없다" — 이 간단한 사실이 신경망 연구의 자금줄을 끊어버렸습니다. 이후 약 15년간 **AI 겨울(AI Winter)**이 찾아왔습니다.

**1986년 — 역전파의 부활**

데이비드 루멜하트(David Rumelhart), 제프리 힌튼(Geoffrey Hinton), 로널드 윌리엄스(Ronald Williams)가 **역전파(Backpropagation)**를 Nature에 발표하면서, 다층 신경망을 효과적으로 학습시킬 수 있다는 것을 보여주었습니다. 민스키가 지적한 XOR 문제? 2층 네트워크면 간단히 해결된다는 것이죠. 신경망 연구가 다시 불을 붙기 시작했습니다.

**안타까운 뒷이야기**: 퍼셉트론을 발명한 로젠블랫은 1971년 43세의 나이에 보트 사고로 세상을 떠났습니다. 자신의 발명이 결국 옳았다는 것을 증명하는 순간을 보지 못한 것입니다. 때로는 시대를 앞서간 아이디어가 인정받기까지 수십 년이 걸리기도 합니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "신경망은 뇌를 모방한 것이다"
>
> 생물학적 뉴런에서 **영감을 받은** 것은 맞지만, 실제 동작 방식은 매우 다릅니다. 생물학적 뉴런은 전기-화학적 신호를 사용하고, 시간에 따라 동적으로 변하며, 구조도 훨씬 복잡합니다. 인공 뉴런은 그저 행렬 곱셈과 비선형 함수의 조합일 뿐이에요. "비행기가 새에서 영감을 받았지만 날개를 퍼덕이지는 않는 것"과 비슷하다고 생각하시면 됩니다.

> ⚠️ **흔한 오해**: "레이어를 깊게 쌓으면 항상 좋다"
>
> 레이어가 깊어지면 더 복잡한 패턴을 학습할 수 있는 건 맞지만, 무조건 좋은 것은 아닙니다. **기울기 소실(Vanishing Gradient)**로 앞쪽 레이어가 학습되지 않거나, **과적합(Overfitting)**으로 훈련 데이터만 외우는 문제가 생길 수 있거든요. 적절한 깊이를 찾는 것이 중요하며, 이 문제를 해결하기 위해 이후에 배울 ResNet의 Skip Connection 같은 기법이 등장했습니다.

> 💡 **알고 계셨나요?**: XOR 문제가 왜 그렇게 중요할까요?
>
> XOR은 "둘 중 하나만 참일 때 참"인 논리 연산입니다. 2D 평면에 점을 찍으면, XOR의 결과를 **하나의 직선으로는 절대 분리할 수 없습니다.** 하지만 직선 **2개** (즉, 2층 네트워크)를 사용하면 깔끔하게 분리됩니다. 이것이 바로 "깊이가 필요한 이유"의 가장 간단하고 명쾌한 예시입니다. 민스키는 "1층으로는 안 된다"고 말했을 뿐인데, 많은 사람이 "신경망 자체가 안 된다"로 오해한 것이 AI 겨울의 비극이었습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **인공 뉴런** | 입력에 가중치를 곱하고, 편향을 더하고, 활성화 함수를 통과 |
| **가중치 (Weight)** | 각 입력의 중요도를 결정하는 학습 가능한 파라미터 |
| **편향 (Bias)** | 활성화 기준점을 조절하는 학습 가능한 파라미터 |
| **레이어** | 뉴런의 묶음. 입력→은닉→출력 순서로 연결 |
| **순전파** | 데이터가 입력에서 출력까지 레이어를 통과하는 과정 |
| **nn.Linear** | PyTorch에서 완전연결 레이어를 만드는 클래스 |

## 다음 섹션 미리보기

뉴런의 출력에서 "활성화 함수"를 언급했는데, 아직 자세히 다루지 않았습니다. 다음 섹션 **[활성화 함수](./02-activation-functions.md)**에서는 ReLU, Sigmoid, GELU 등 왜 활성화 함수가 필요한지, 각각 어떤 특성을 가지는지 깊이 파봅니다.

## 참고 자료

- [Google ML Crash Course - Neural Network Nodes and Layers](https://developers.google.com/machine-learning/crash-course/neural-networks/nodes-hidden-layers) - 뉴런과 레이어의 직관적 설명
- [PyTorch 공식 튜토리얼 - Neural Networks](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) - PyTorch로 신경망 구축하기
- [Victor Zhou - Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/) - 수학부터 코드까지 단계별 설명
- [GeeksforGeeks - Weights and Bias in Neural Networks](https://www.geeksforgeeks.org/deep-learning/the-role-of-weights-and-bias-in-neural-networks/) - 가중치와 편향의 역할 상세 설명
