# Embodied AI

> 로봇 비전과 액션

## 개요

이 섹션에서는 AI가 **물리적 세계에서 행동하는** Embodied AI를 다룹니다. 카메라로 보고, 언어로 이해하고, 로봇 팔로 행동하는 **Vision-Language-Action(VLA)** 모델의 세계를 탐험합니다. RT-2, PaLM-E, OpenVLA 같은 최신 연구부터, "로봇에게 언어로 명령하면 알아서 수행"하는 미래까지 살펴봅니다.

**선수 지식**:
- [Vision-Language 모델](../10-vision-language/01-multimodal-learning.md)의 기본 개념
- [World Models](./02-world-models.md)의 시뮬레이션 개념

**학습 목표**:
- Embodied AI와 VLA 모델의 개념 이해하기
- RT-2, PaLM-E의 아키텍처와 동작 방식 파악하기
- 로봇 학습의 현재 상태와 도전 과제 알아보기

## 왜 알아야 할까?

지금까지 배운 AI 모델들은 대부분 **"인식"** 에 집중했습니다:

| 기존 CV/VLM | Embodied AI |
|-------------|-------------|
| 이미지 → 텍스트 | 이미지 + 텍스트 → 행동 |
| "이게 뭐야?" | "이걸 저기로 옮겨" |
| 화면 안에서 동작 | 물리적 세계에서 동작 |
| 결과: 답변 | 결과: 로봇 움직임 |

Embodied AI는 **AI의 최종 목표** 중 하나입니다. 단순히 "보고 이해하는" 것을 넘어 **실제로 세상에 영향을 미치는** AI를 만드는 것이죠.

## 핵심 개념

### 개념 1: Embodied AI란?

> 💡 **비유**: 지금까지의 AI가 "TV 해설위원"이라면, Embodied AI는 "실제 경기에 뛰는 선수"입니다. 경기를 분석하고 설명하는 것과, 직접 공을 차고 골을 넣는 것은 완전히 다른 능력이 필요하죠.

**Embodied AI의 정의:**

물리적 **몸(body)**을 가지고 환경과 **상호작용**하는 AI 시스템:
- **Perception**: 센서로 환경 인식 (카메라, LiDAR 등)
- **Cognition**: 상황 이해와 계획 수립
- **Action**: 액추에이터로 행동 실행 (로봇 팔, 바퀴 등)

**핵심 루프:**

```
환경 관찰 → 상황 이해 → 목표 설정 → 행동 계획 → 행동 실행 → 피드백 수신 → ...
    ↑                                                        |
    └────────────────────────────────────────────────────────┘
```

**기존 로봇과의 차이:**

| 전통적 로봇 | Embodied AI |
|------------|-------------|
| 사전 프로그래밍된 동작 | 학습된 일반화 능력 |
| 정해진 환경에서만 동작 | 새로운 환경에 적응 |
| 정확한 명령 필요 | 자연어 명령 이해 |
| 단일 태스크 | 다중 태스크 수행 |

### 개념 2: Vision-Language-Action (VLA) 모델

> 💡 **비유**: VLA 모델은 **동시통역사 겸 배우**와 같습니다. 상황을 보고(Vision), 언어로 명령을 듣고(Language), 직접 연기하듯 행동합니다(Action).

**VLA의 구조:**

```
       ┌─────────────┐
       │ 카메라 영상  │
       └──────┬──────┘
              ▼
       ┌─────────────┐      ┌─────────────┐
       │ 비전 인코더  │      │ 언어 명령   │
       └──────┬──────┘      └──────┬──────┘
              │                    │
              └────────┬───────────┘
                       ▼
              ┌─────────────┐
              │ 멀티모달 융합 │
              │  (LLM 기반)  │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │ 행동 디코더  │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │ 로봇 제어   │
              │ (x, y, z,   │
              │  θ, gripper)│
              └─────────────┘
```

**행동 표현 방식:**

로봇 행동을 "토큰"으로 표현하여 LLM이 다룰 수 있게 합니다:
- **End-effector pose**: (x, y, z, roll, pitch, yaw)
- **그리퍼 상태**: 열기/닫기
- **이산화(Discretization)**: 연속 값을 토큰으로 변환

### 개념 3: RT-2 - Robotics Transformer 2

**RT-2 (2023, Google DeepMind)**는 대규모 VLM을 로봇 제어에 적용한 선구적 연구입니다.

**핵심 아이디어:**

> "인터넷에서 학습한 상식을 로봇 행동으로 전이"

**아키텍처:**

1. **베이스 모델**: PaLI-X 또는 PaLM-E (거대 VLM)
2. **입력**: 이미지 + 텍스트 명령
3. **출력**: 텍스트로 표현된 로봇 행동

**행동의 토큰화:**

```
행동: "앞으로 0.1m 이동, 그리퍼 닫기"
    ↓
토큰: [<action> 1 128 64 32 255 1 </action>]
      (x=0.1, y=0, z=0, rx=0, ry=0, rz=0, gripper=closed)
```

**놀라운 일반화:**

RT-2는 학습 데이터에 없는 새로운 상황에서도 동작합니다:
- "컵을 Taylor Swift 그림 위로 옮겨" → 유명인 지식 활용
- "빨간색이 아닌 것을 골라" → 부정 추론
- "재활용품 분류" → 상식적 개념 적용

### 개념 4: PaLM-E - 거대 Embodied LLM

**PaLM-E (2023, Google)**는 **5620억 파라미터**의 거대 멀티모달 언어 모델입니다.

**핵심 특징:**

1. **다중 로봇 플랫폼**: 단일 모델로 여러 로봇 제어
2. **장기 계획**: 복잡한 태스크를 단계별로 분해
3. **상식 추론**: 텍스트 데이터에서 학습한 지식 활용

**구조:**

```
┌───────────────────────────────────────────────┐
│                   PaLM-E                       │
│  ┌─────────────────────────────────────────┐  │
│  │            PaLM (540B LLM)              │  │
│  └─────────────────────────────────────────┘  │
│        ▲           ▲           ▲              │
│  ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐       │
│  │ 비전 토큰 │ │텍스트 │ │ 상태 토큰 │       │
│  │ (ViT-22B)│ │       │ │ (센서)   │       │
│  └───────────┘ └───────┘ └───────────┘       │
└───────────────────────────────────────────────┘
```

**태스크 분해 예시:**

명령: "서랍에서 칩을 꺼내 테이블로 가져와"

PaLM-E 출력:
```
1. 서랍으로 이동
2. 서랍 손잡이 잡기
3. 서랍 열기
4. 칩 봉지 찾기
5. 칩 봉지 집기
6. 테이블로 이동
7. 칩 봉지 내려놓기
```

### 개념 5: Open X-Embodiment와 OpenVLA

**데이터의 문제:**

로봇 학습의 가장 큰 병목은 **데이터 부족**입니다:
- 인터넷에 로봇 행동 데이터 거의 없음
- 각 연구실마다 다른 로봇, 다른 형식
- 소량의 데모만으로 일반화 어려움

**Open X-Embodiment (2023):**

Google DeepMind 주도로 **22개 기관**이 협력:
- **150만+ 에피소드** 로봇 데이터 통합
- **22종의 로봇** 플랫폼 포함
- RT-X: 이 데이터로 학습한 범용 정책

**OpenVLA (2024):**

오픈소스 VLA 모델:
- **7B 파라미터** (PaLM-E보다 훨씬 작음)
- Llama 2 기반 파인튜닝
- 상용 로봇에서 실행 가능

### 개념 6: 로봇 학습의 패러다임

**1. Imitation Learning (모방 학습)**

사람의 시연을 따라 배움:
- 전문가 데모 수집
- 상태-행동 쌍 학습
- 장점: 보상 설계 불필요
- 단점: 분포 이탈(distribution shift) 문제

**2. Reinforcement Learning (강화 학습)**

시행착오로 배움:
- 환경에서 탐색
- 보상 최대화 학습
- 장점: 최적 정책 발견 가능
- 단점: 샘플 비효율, 실제 로봇에서 위험

**3. Foundation Model 기반**

대규모 사전학습 후 적응:
- VLM으로 상식 습득
- 소량의 로봇 데이터로 파인튜닝
- 장점: 뛰어난 일반화
- 단점: 계산 비용 높음

**현재 트렌드: 1 + 3 조합**

소량의 시연 + Foundation Model의 일반화 능력

## 실습: VLA 모델 개념 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class SimpleVLA(nn.Module):
    """
    간단한 Vision-Language-Action 모델

    이미지와 언어 명령을 받아 로봇 행동 출력
    """
    def __init__(
        self,
        vision_dim: int = 768,     # 비전 인코더 출력 차원
        action_dim: int = 7,        # 행동 차원 (6DoF + gripper)
        hidden_dim: int = 512,
        num_action_bins: int = 256  # 행동 이산화 빈 수
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_action_bins = num_action_bins

        # 비전 인코더 (사전학습된 ViT 사용 가정)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # 언어 임베딩 프로젝션
        self.text_proj = nn.Linear(768, hidden_dim)  # BERT 차원 가정

        # 멀티모달 융합
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=4
        )

        # 행동 예측 헤드 (이산화된 행동)
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_action_bins)
            for _ in range(action_dim)
        ])

        # 행동 값 범위 (정규화용)
        self.action_min = torch.tensor([-0.5, -0.5, -0.5, -3.14, -3.14, -3.14, 0.0])
        self.action_max = torch.tensor([0.5, 0.5, 0.5, 3.14, 3.14, 3.14, 1.0])

    def forward(
        self,
        vision_features: torch.Tensor,   # (B, num_patches, vision_dim)
        text_features: torch.Tensor,     # (B, seq_len, 768)
    ) -> dict:
        """
        VLA forward pass

        Args:
            vision_features: 비전 인코더 출력
            text_features: 텍스트 인코더 출력

        Returns:
            action_logits: 각 행동 차원별 로짓
            actions: 디코딩된 연속 행동 값
        """
        B = vision_features.shape[0]

        # 프로젝션
        vision_emb = self.vision_proj(vision_features)  # (B, num_patches, hidden)
        text_emb = self.text_proj(text_features)        # (B, seq_len, hidden)

        # 시퀀스 결합 [CLS, vision_tokens, text_tokens]
        cls_token = torch.zeros(B, 1, vision_emb.shape[-1], device=vision_emb.device)
        sequence = torch.cat([cls_token, vision_emb, text_emb], dim=1)

        # Transformer 처리
        fused = self.fusion(sequence)

        # CLS 토큰에서 행동 예측
        cls_output = fused[:, 0]  # (B, hidden)

        # 각 행동 차원별 분류
        action_logits = [head(cls_output) for head in self.action_heads]
        action_probs = [F.softmax(logits, dim=-1) for logits in action_logits]

        # 가장 높은 확률의 빈 선택
        action_bins = [probs.argmax(dim=-1) for probs in action_probs]
        action_bins = torch.stack(action_bins, dim=-1)  # (B, action_dim)

        # 빈을 연속 값으로 변환
        actions = self.bins_to_continuous(action_bins)

        return {
            'action_logits': action_logits,
            'action_bins': action_bins,
            'actions': actions
        }

    def bins_to_continuous(self, bins: torch.Tensor) -> torch.Tensor:
        """이산화된 빈을 연속 행동 값으로 변환"""
        # [0, num_bins-1] → [0, 1] → [min, max]
        normalized = bins.float() / (self.num_action_bins - 1)
        range_vals = self.action_max - self.action_min
        actions = self.action_min + normalized * range_vals
        return actions

    def continuous_to_bins(self, actions: torch.Tensor) -> torch.Tensor:
        """연속 행동 값을 빈으로 변환 (학습 타깃용)"""
        normalized = (actions - self.action_min) / (self.action_max - self.action_min)
        normalized = torch.clamp(normalized, 0, 1)
        bins = (normalized * (self.num_action_bins - 1)).long()
        return bins


class VLATrainer:
    """VLA 모델 학습"""

    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        target_actions: torch.Tensor
    ):
        """
        한 스텝 학습 (모방 학습)

        Args:
            target_actions: 전문가의 실제 행동
        """
        self.optimizer.zero_grad()

        outputs = self.model(vision_features, text_features)

        # 행동을 빈으로 변환
        target_bins = self.model.continuous_to_bins(target_actions)

        # Cross-entropy 손실
        total_loss = 0
        for d in range(self.model.action_dim):
            loss = F.cross_entropy(
                outputs['action_logits'][d],
                target_bins[:, d]
            )
            total_loss += loss

        total_loss.backward()
        self.optimizer.step()

        return {'loss': total_loss.item() / self.model.action_dim}


# 사용 예시
if __name__ == "__main__":
    model = SimpleVLA(action_dim=7)
    print(f"VLA 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 더미 입력
    batch_size = 4
    vision_feat = torch.randn(batch_size, 196, 768)  # ViT 패치
    text_feat = torch.randn(batch_size, 32, 768)      # 텍스트 토큰

    # Forward
    outputs = model(vision_feat, text_feat)
    print(f"예측 행동: {outputs['actions'].shape}")  # (4, 7)
    print(f"행동 값: {outputs['actions'][0]}")  # 첫 샘플의 7DoF 행동
```

```python
# OpenVLA 스타일 모델 (실제 사용 예시)
"""
OpenVLA는 Hugging Face에서 사용 가능합니다.
실제 로봇에서 실행하려면 해당 로봇의 제어 인터페이스가 필요합니다.
"""

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

def load_openvla():
    """OpenVLA 모델 로드"""
    # 7B 모델 (GPU 메모리 24GB+ 권장)
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda")

    return model, processor


def predict_action(model, processor, image, instruction):
    """
    이미지와 명령으로 로봇 행동 예측

    Args:
        image: PIL Image 또는 이미지 경로
        instruction: 자연어 명령 문자열

    Returns:
        actions: numpy array (7,) - [x, y, z, rx, ry, rz, gripper]
    """
    if isinstance(image, str):
        image = Image.open(image)

    # 입력 처리
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

    # 행동 생성
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig")

    return action  # numpy array (7,)


# 사용 예시
"""
model, processor = load_openvla()

# 카메라에서 이미지 획득
image = camera.get_frame()

# 자연어 명령
instruction = "pick up the red cup"

# 행동 예측
action = predict_action(model, processor, image, instruction)
# action = [0.02, -0.01, 0.15, 0.0, 0.0, 0.1, 0.8]
#           (x,    y,     z,   rx,  ry,  rz,  gripper)

# 로봇 제어 (로봇 SDK 사용)
robot.execute_action(action)
"""
```

```python
# 강화학습 + Foundation Model 결합 예시
import torch
import torch.nn as nn
import numpy as np

class VLAWithRL(nn.Module):
    """
    VLA + 강화학습 파인튜닝

    Foundation Model의 일반화 + RL의 최적화
    """
    def __init__(self, base_vla, action_dim=7):
        super().__init__()
        self.base_vla = base_vla  # 사전학습된 VLA

        # 가치 함수 헤드 (Actor-Critic용)
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 정책 분포 파라미터 (연속 행동용)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, vision_features, text_features):
        # Base VLA로 행동 예측
        base_outputs = self.base_vla(vision_features, text_features)

        # 행동 평균 (base VLA 출력)
        action_mean = base_outputs['actions']

        # 행동 분산 (학습 가능)
        action_std = torch.exp(self.log_std)

        # 가치 추정 (Critic)
        # 실제로는 fusion 출력을 사용해야 함
        value = self.value_head(torch.zeros_like(action_mean[:, :1].expand(-1, 512)))

        return {
            'action_mean': action_mean,
            'action_std': action_std,
            'value': value.squeeze(-1)
        }

    def get_action(self, vision_features, text_features, deterministic=False):
        """행동 샘플링"""
        outputs = self.forward(vision_features, text_features)

        if deterministic:
            return outputs['action_mean']
        else:
            # 가우시안에서 샘플링
            noise = torch.randn_like(outputs['action_mean']) * outputs['action_std']
            return outputs['action_mean'] + noise


# PPO 스타일 업데이트 (개념)
def ppo_update(model, optimizer, batch, clip_ratio=0.2):
    """
    PPO (Proximal Policy Optimization) 업데이트

    실제 환경에서 수집한 데이터로 정책 개선
    """
    vision_feat = batch['vision_features']
    text_feat = batch['text_features']
    actions = batch['actions']
    returns = batch['returns']
    old_log_probs = batch['log_probs']

    outputs = model(vision_feat, text_feat)

    # 새 정책의 로그 확률
    dist = torch.distributions.Normal(
        outputs['action_mean'],
        outputs['action_std']
    )
    new_log_probs = dist.log_prob(actions).sum(-1)

    # PPO Clipped Objective
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    advantages = returns - outputs['value'].detach()
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # Value function loss
    value_loss = F.mse_loss(outputs['value'], returns)

    # Entropy bonus (탐색 장려)
    entropy = dist.entropy().mean()

    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item()
    }


print("VLA + RL 예시 로드 완료")
```

## 더 깊이 알아보기

### RT-2의 혁신성

RT-2의 가장 놀라운 점은 **인터넷에서 학습한 지식이 로봇 행동으로 전이**된다는 것입니다.

> 💡 **알고 계셨나요?**: RT-2 학습 데이터에는 "Taylor Swift" 사진이 없었습니다. 하지만 "Taylor Swift 그림 위로 물건을 옮겨"라는 명령을 수행할 수 있었습니다. 웹에서 학습한 유명인 지식이 로봇 행동에 자연스럽게 연결된 거죠!

### Sim-to-Real Gap

시뮬레이션에서 학습한 정책이 실제 로봇에서 잘 동작하지 않는 문제:

**원인:**
- 물리 시뮬레이션의 한계 (마찰, 접촉 등)
- 센서 노이즈 차이
- 시각적 도메인 차이

**해결책:**
1. **Domain Randomization**: 시뮬레이션 파라미터 무작위화
2. **Real-to-Sim-to-Real**: 실제 데이터로 시뮬레이터 개선
3. **Foundation Model**: 다양한 실제 데이터에서 사전학습

> 🔥 **실무 팁**: Sim-to-Real 전이를 개선하려면 **시각적 도메인 랜덤화**가 효과적입니다. 조명, 텍스처, 카메라 위치 등을 무작위로 바꿔가며 학습하면 실제 환경 변화에 강건해집니다.

### 인간-로봇 협업의 미래

**현재 (2024-2025):**
- 단일 태스크 수행 가능
- 제한된 환경에서 동작
- 높은 실패율

**단기 미래 (2026-2028):**
- 가정용 보조 로봇
- 창고 물류 자동화
- 요식업 서비스 로봇

**장기 미래:**
- 범용 가사 로봇
- 건설/제조 협업 로봇
- 의료 보조 로봇

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "거대 모델이면 로봇도 잘 제어한다" — PaLM-E처럼 큰 모델도 **물리적 접촉, 힘 조절** 같은 저수준 제어는 여전히 어렵습니다. 언어적 계획과 물리적 실행은 다른 문제입니다.

> 💡 **알고 계셨나요?**: 현재 가장 성능 좋은 로봇 학습 시스템도 **새로운 물체 집기 성공률이 70-80%** 수준입니다. 인간은 99%+를 가볍게 달성하죠. 아직 갈 길이 멉니다.

> 🔥 **실무 팁**: 로봇 학습 프로젝트를 시작한다면 **Open X-Embodiment 데이터셋**을 활용하세요. 다양한 로봇의 150만+ 에피소드가 공개되어 있어 사전학습에 유용합니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| Embodied AI | 물리적 세계에서 행동하는 AI 시스템 |
| VLA | Vision-Language-Action, 멀티모달 로봇 제어 모델 |
| RT-2 | VLM→로봇 행동 전이, 상식 기반 일반화 |
| PaLM-E | 540B 거대 모델, 장기 계획 및 다중 로봇 제어 |
| OpenVLA | 7B 오픈소스 VLA, 실용적 로봇 적용 가능 |

## 다음 섹션 미리보기

Embodied AI가 현재 로봇 기술의 최전선이라면, 이 모든 것은 어디로 향하고 있을까요? [미래 연구 방향](./04-future-directions.md)에서는 **AGI를 향한 비전 기술**의 로드맵, 현재의 한계와 극복 방향, 그리고 CV 분야의 미래 전망을 살펴봅니다.

## 참고 자료

- [VLA Survey (arXiv 2024)](https://arxiv.org/abs/2405.14093) - Vision-Language-Action 모델 종합 서베이
- [RT-2 논문](https://robotics-transformer2.github.io/) - Google DeepMind
- [PaLM-E 논문](https://palm-e.github.io/) - 거대 Embodied 멀티모달 모델
- [OpenVLA GitHub](https://github.com/openvla/openvla) - 오픈소스 VLA 구현
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) - 대규모 로봇 데이터셋
- [Embodied AI Survey (2025)](https://arxiv.org/abs/2505.20503) - 최신 동향 정리
