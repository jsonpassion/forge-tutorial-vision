# 활성화 함수

> ReLU, Sigmoid, Tanh, GELU 비교

## 개요

앞 섹션에서 뉴런이 "가중합 → 활성화 함수"를 거친다고 배웠습니다. 그런데 왜 활성화 함수가 필요할까요? 활성화 함수가 없으면 아무리 레이어를 쌓아도 결국 **하나의 선형 변환**과 같아져서, 복잡한 패턴을 절대 학습할 수 없습니다. 활성화 함수는 신경망에 **비선형성**을 부여하는 핵심 부품입니다.

**선수 지식**: [신경망의 구조](./01-neural-network.md) — 뉴런, 가중합, 레이어
**학습 목표**:
- 활성화 함수가 왜 필요한지 직관적으로 이해한다
- Sigmoid, Tanh, ReLU, GELU의 특성과 차이를 구분한다
- 상황에 맞는 활성화 함수를 선택할 수 있다

## 왜 알아야 할까?

활성화 함수 선택은 모델의 학습 속도와 성능에 직접 영향을 줍니다. CNN에서는 ReLU가, Transformer에서는 GELU가 표준으로 쓰입니다. 잘못된 활성화 함수를 쓰면 **기울기 소실(Vanishing Gradient)** 문제로 모델이 전혀 학습하지 못할 수도 있습니다.

## 핵심 개념

### 1. 왜 비선형이 필요한가?

> 💡 **비유**: 활성화 함수 없는 신경망은 **직선 자만 가진 제도사**와 같습니다. 직선만으로는 곡선이나 복잡한 도형을 그릴 수 없죠. 활성화 함수는 **곡선 자(곡자)**를 추가해 주는 것입니다. 이것이 있어야 비로소 복잡한 패턴을 표현할 수 있습니다.

활성화 함수가 없다면:

> 레이어1: y = W₁x + b₁
> 레이어2: z = W₂y + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)

결국 하나의 선형 변환 `z = Ax + c`와 동일합니다. 100개 레이어를 쌓아도 1개와 같은 효과입니다.

### 2. Sigmoid — 0과 1 사이로 압축

> 💡 **비유**: 시험 점수를 **합격률(0~100%)**로 바꾸는 것과 같습니다. 점수가 아무리 높아도 100%를 넘지 않고, 아무리 낮아도 0% 아래로 내려가지 않습니다.

**수식**: σ(x) = 1 / (1 + e⁻ˣ)

| 특성 | 값 |
|------|---|
| 출력 범위 | (0, 1) |
| x = 0일 때 | 0.5 |
| 큰 양수 | → 1에 수렴 |
| 큰 음수 | → 0에 수렴 |

```python
import torch
import torch.nn as nn

sigmoid = nn.Sigmoid()

x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
y = sigmoid(x)
print(f"입력: {x.tolist()}")
print(f"출력: {[f'{v:.4f}' for v in y.tolist()]}")
# 출력: [0.0067, 0.2689, 0.5000, 0.7311, 0.9933]
```

**장점**: 출력이 확률처럼 해석 가능
**단점**: 큰 값/작은 값에서 기울기가 거의 0 → **기울기 소실 문제**

> ⚠️ 기울기 소실(Vanishing Gradient): 역전파 시 기울기가 0에 가까워져 앞쪽 레이어가 거의 학습되지 않는 현상. 깊은 네트워크에서 치명적입니다.

### 3. Tanh — -1과 1 사이로 압축

> 💡 **비유**: Sigmoid와 비슷하지만, 결과를 **-100점 ~ +100점** 범위로 매기는 것입니다. **0을 기준으로 대칭**이라 양수/음수 모두 표현합니다.

**수식**: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

```python
import torch
import torch.nn as nn

tanh = nn.Tanh()

x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
y = tanh(x)
print(f"입력: {x.tolist()}")
print(f"출력: {[f'{v:.4f}' for v in y.tolist()]}")
# 출력: [-0.9999, -0.7616, 0.0000, 0.7616, 0.9999]
```

**Sigmoid 대비 장점**: 출력이 0 중심 → 다음 레이어 학습에 유리
**단점**: 여전히 큰 값에서 기울기 소실 발생

### 4. ReLU — 단순하지만 강력한

> 💡 **비유**: **음수는 0으로, 양수는 그대로** 통과시키는 문지기입니다. "부정적인 건 차단하고, 긍정적인 건 그대로 보내라!" 규칙이 매우 단순해서 빠르고, 놀랍도록 잘 작동합니다.

**수식**: ReLU(x) = max(0, x)

```python
import torch
import torch.nn as nn

relu = nn.ReLU()

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
y = relu(x)
print(f"입력: {x.tolist()}")
print(f"출력: {y.tolist()}")
# 출력: [0.0, 0.0, 0.0, 1.0, 3.0]
```

**왜 ReLU가 대세가 되었나:**
- 계산이 매우 빠름 (`max` 연산 하나)
- 양수 영역에서 기울기가 항상 1 → **기울기 소실 없음**
- Sigmoid/Tanh보다 학습이 **6배 빠르다**는 연구 결과 (AlexNet 논문)

**ReLU의 단점 — Dead Neuron 문제:**

> 학습 중 가중치가 업데이트되어 뉴런의 입력이 항상 음수가 되면, 출력이 영원히 0이 됩니다. 이런 뉴런은 "죽은 뉴런"이라 하며, 더 이상 학습에 참여하지 못합니다.

**해결책 — Leaky ReLU:**

```python
import torch
import torch.nn as nn

# Leaky ReLU: 음수도 아주 작은 기울기(0.01)를 줌
leaky_relu = nn.LeakyReLU(0.01)

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
y = leaky_relu(x)
print(f"입력:      {x.tolist()}")
print(f"LeakyReLU: {y.tolist()}")
# 출력: [-0.03, -0.01, 0.0, 1.0, 3.0]  ← 음수도 완전 0이 아님
```

### 5. GELU — Transformer 시대의 표준

> 💡 **비유**: ReLU가 "통과/차단"을 **딱 잘라서** 결정하는 경비원이라면, GELU는 **확률적으로** 결정하는 부드러운 경비원입니다. "이 정도면 70% 확률로 통과시키겠습니다" 식으로 부드러운 경계를 만듭니다.

**수식**: GELU(x) = x · Φ(x) (Φ는 표준 정규분포의 누적분포함수)

```python
import torch
import torch.nn as nn

gelu = nn.GELU()

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
y = gelu(x)
print(f"입력: {x.tolist()}")
print(f"GELU: {[f'{v:.4f}' for v in y.tolist()]}")
# 출력: [-0.0036, -0.1587, 0.0000, 0.8413, 2.9964]
```

**GELU를 쓰는 모델들**: BERT, GPT, ViT, Swin Transformer 등 Transformer 기반 모델 대부분

### 6. 한눈에 비교

| 활성화 함수 | 출력 범위 | 기울기 소실 | 속도 | 주요 용도 |
|------------|----------|-----------|------|----------|
| **Sigmoid** | (0, 1) | 심함 | 보통 | 이진 분류 출력층 |
| **Tanh** | (-1, 1) | 있음 | 보통 | RNN 은닉 상태 |
| **ReLU** | [0, ∞) | 없음 (양수) | 매우 빠름 | **CNN 기본값** |
| **Leaky ReLU** | (-∞, ∞) | 없음 | 빠름 | Dead Neuron 방지 |
| **GELU** | ≈(-0.17, ∞) | 없음 | 보통 | **Transformer 기본값** |

> **실무 가이드**: CNN → **ReLU**, Transformer → **GELU**, 출력층 확률 → **Sigmoid**(이진) 또는 **Softmax**(다중)

## 실습: 직접 해보기

### 활성화 함수별 출력 비교

```python
import torch
import torch.nn as nn

# 활성화 함수 모음
activations = {
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(0.01),
    "GELU": nn.GELU(),
}

x = torch.linspace(-5, 5, 11)  # -5부터 5까지 11개 점

print(f"{'입력':>8s}", end="")
for name in activations:
    print(f" | {name:>10s}", end="")
print()
print("-" * 75)

for val in x:
    print(f"{val.item():>8.1f}", end="")
    for name, fn in activations.items():
        out = fn(val.unsqueeze(0)).item()
        print(f" | {out:>10.4f}", end="")
    print()
```

## 더 깊이 알아보기

### ReLU의 놀라운 역사 — 무시당한 70년

> 💡 **알고 계셨나요?**: ReLU의 개념은 사실 **1941년**까지 거슬러 올라갑니다.

수학자 알스턴 하우스홀더(Alston Householder)가 1941년에 이미 `max(0, x)` 형태의 함수를 제안했습니다. 하지만 수십 년 동안 아무도 이것을 신경망에 쓸 생각을 하지 않았어요. 왜냐하면 Sigmoid가 생물학적 뉴런의 "발화" 패턴과 더 비슷해 보였거든요. "0 아니면 그대로 통과"라는 ReLU의 단순함은 너무 단순해서 오히려 무시당한 셈이죠.

**2010~2011년의 전환점**: 글로로(Glorot) 등의 연구진이 ReLU가 깊은 네트워크에서 Sigmoid보다 **극적으로 뛰어난 성능**을 보인다는 것을 증명했습니다. 그리고 2012년, **AlexNet**이 ReLU를 사용하여 ImageNet 대회에서 압도적 승리를 거두면서 세상이 바뀌었죠. AlexNet 논문에서 크리체프스키(Krizhevsky)는 ReLU 덕분에 학습 속도가 Sigmoid 대비 **6배 빨라졌다**고 보고했습니다.

**Dying ReLU의 현실**: 실무에서 ReLU를 사용하면, 학습 중 뉴런의 최대 **40%가 죽을 수 있다**는 연구 결과가 있습니다. 한번 죽은 뉴런은 절대 되살아나지 않거든요. 이것이 Leaky ReLU, PReLU, ELU 같은 변형들이 계속 등장하는 이유입니다. 그럼에도 불구하고 기본 ReLU가 여전히 가장 널리 쓰이는 것은, 대부분의 경우 충분히 잘 작동하기 때문입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "ReLU는 x=0에서 미분 불가능하니 문제가 있다"
>
> 수학적으로는 맞는 말이지만, 실무에서는 전혀 문제가 되지 않습니다. 연속적인 실수 값으로 이루어진 입력이 **정확히 0.000...0**일 확률은 사실상 0이거든요. PyTorch를 포함한 대부분의 프레임워크는 x=0에서의 기울기를 그냥 0으로 정의해 버리며, 이것으로 아무 문제 없이 학습이 잘 됩니다.

> ⚠️ **흔한 오해**: "생물학적으로 정확한 함수가 더 좋은 활성화 함수다"
>
> Sigmoid가 생물학적 뉴런의 발화 패턴과 비슷하다는 이유로 오랫동안 선호되었지만, 실제로 성능과 생물학적 유사성은 **전혀 상관이 없습니다.** 단순한 ReLU가 "생물학적으로 정확한" Sigmoid보다 훨씬 잘 작동한다는 것이 수많은 실험으로 증명되었습니다. 딥러닝은 뇌 시뮬레이션이 아니라 **최적화 문제**라는 점을 기억하세요.

> 🔥 **실무 팁**: 학습이 갑자기 멈추고 출력이 항상 같은 값이라면?
>
> **Dying ReLU**를 의심해 보세요! 뉴런이 대량으로 죽으면 모델이 더 이상 학습하지 못합니다. 해결 방법은 세 가지입니다: (1) **학습률을 낮추기** — 너무 큰 학습률이 가중치를 급격히 바꿔 뉴런을 죽이는 경우가 많습니다. (2) **He 초기화** 사용 — ReLU에 최적화된 가중치 초기화 방법입니다. (3) **Leaky ReLU로 교체** — 음수 영역에서도 작은 기울기를 유지하므로 뉴런이 죽지 않습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **비선형성** | 복잡한 패턴을 학습하기 위해 필수. 없으면 깊은 망 = 얕은 망 |
| **기울기 소실** | Sigmoid/Tanh의 문제. 깊은 네트워크에서 앞쪽 레이어 학습 불능 |
| **ReLU** | max(0, x). CNN의 기본 활성화 함수. 빠르고 기울기 소실 없음 |
| **GELU** | 부드러운 ReLU. Transformer 계열의 표준 |
| **Dead Neuron** | ReLU에서 항상 0 출력하는 뉴런. Leaky ReLU로 해결 |

## 다음 섹션 미리보기

뉴런이 순전파로 출력을 만드는 법을 배웠습니다. 하지만 "학습"은 어떻게 일어날까요? 다음 섹션 **[역전파 알고리즘](./03-backpropagation.md)**에서는 신경망이 **실수를 교정하며 스스로 개선**하는 핵심 메커니즘을 배웁니다.

## 참고 자료

- [Interactive Guide to Activation Functions](https://mbrenndoerfer.com/writing/activation-functions-neural-networks-complete-guide) - Sigmoid부터 GELU까지 인터랙티브 시각화
- [IABAC - ReLU Activation Function Complete Guide 2025](https://iabac.org/blog/relu-activation-function) - ReLU의 역사와 변형 종합 가이드
- [GeeksforGeeks - Activation Functions in Neural Networks](https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/) - 모든 활성화 함수 비교 정리
- [Prodia Blog - GELU vs ReLU](https://blog.prodia.com/post/compare-4-key-differences-gelu-vs-re-lu-in-neural-networks) - GELU와 ReLU의 4가지 핵심 차이
