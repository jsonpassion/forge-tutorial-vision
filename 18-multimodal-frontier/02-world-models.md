# World Models

> 물리 세계 이해와 시뮬레이션

## 개요

이 섹션에서는 AI가 **물리 세계를 이해하고 시뮬레이션**하는 World Models의 개념을 다룹니다. OpenAI가 Sora를 "세계 시뮬레이터"라고 부른 이유, NVIDIA Cosmos와 DeepMind Genie 같은 최신 연구, 그리고 World Models가 자율주행, 로보틱스, 게임 AI에 미치는 영향까지 살펴봅니다.

**선수 지식**:
- [비디오 Diffusion](../15-video-generation/01-video-diffusion.md)의 시간적 모델링
- [Sora와 DiT](../15-video-generation/04-sora.md)의 아키텍처

**학습 목표**:
- World Model의 정의와 목적 이해하기
- 비디오 생성 모델과 World Model의 관계 파악하기
- 물리 법칙 학습의 가능성과 한계 알아보기

## 왜 알아야 할까?

[비디오 생성](../15-video-generation/04-sora.md)에서 Sora의 놀라운 영상 생성 능력을 배웠습니다. 하지만 OpenAI는 Sora를 단순한 "비디오 생성기"가 아닌 **"세계 시뮬레이터"**라고 정의했습니다. 왜일까요?

| 비디오 생성 모델 | World Model |
|-----------------|-------------|
| "그럴듯한" 영상 생성 | 물리 법칙에 맞는 예측 |
| 패턴 학습 | 인과 관계 이해 |
| 정해진 프롬프트 실행 | 상호작용 가능한 시뮬레이션 |
| 엔터테인먼트 목적 | 의사결정, 계획 수립 |

World Models는 **자율주행 시뮬레이션**, **로봇 훈련**, **게임 AI** 등 실제 세계와 상호작용해야 하는 AI의 핵심 기술입니다.

## 핵심 개념

### 개념 1: World Model이란 무엇인가?

> 💡 **비유**: 체스 마스터가 다음 수를 두기 전에 **머릿속으로 여러 수를 시뮬레이션**하는 것을 생각해보세요. "내가 이렇게 두면 상대는 저렇게 둘 것이고..." 이 "머릿속 시뮬레이션"이 바로 World Model입니다. AI가 실제로 행동하지 않고도 결과를 예측하는 내부 모델이죠.

**World Model의 정의:**

세계 모델(World Model)은 환경의 **역학(dynamics)**을 학습한 내부 표현입니다:

$$s_{t+1} = f(s_t, a_t)$$

- $s_t$: 현재 상태 (현재 보이는 장면)
- $a_t$: 취한 행동 (카메라 이동, 물체 조작 등)
- $s_{t+1}$: 예측된 다음 상태
- $f$: World Model (학습된 전이 함수)

**핵심 기능:**

1. **예측 (Prediction)**: 미래 상태 예측
2. **시뮬레이션 (Simulation)**: 가상 환경에서 실험
3. **계획 (Planning)**: 최적 행동 시퀀스 탐색
4. **이해 (Understanding)**: 인과 관계 파악

### 개념 2: Sora as a World Simulator

> 💡 **비유**: Sora가 물 위에 떠 있는 배를 보여줄 때, 단순히 "배 이미지를 여러 프레임으로 복사"하는 게 아닙니다. 파도가 치면 배가 흔들리고, 물이 튀고, 햇빛 반사가 변합니다. 마치 **물리 엔진을 내장한 것처럼** 행동합니다.

**Sora의 World Model 특성:**

OpenAI는 2024년 2월 Sora 기술 보고서에서 다음을 강조했습니다:

1. **3D 일관성**: 카메라 이동에 따른 올바른 시점 변화
2. **객체 영속성**: 물체가 가려져도 존재 유지
3. **물리 시뮬레이션**: 페인트가 튀거나, 유리가 깨지는 효과
4. **상호작용 이해**: 물체 간 충돌, 중력 효과

**Sora 1 → Sora 2 발전:**

| Sora 1 (2024.02) | Sora 2 (2024.12) |
|------------------|------------------|
| "GPT-1 모먼트" | "GPT-3.5 모먼트" |
| 기본 물리 이해 | 고급 물리 시뮬레이션 |
| 단일 장면 | 멀티씬 제어 |
| 읽기만 가능 | 편집/리믹스 가능 |

### 개념 3: 주요 World Model 시스템들

**2024-2025년 주요 연구:**

| 모델 | 개발사 | 특징 |
|------|--------|------|
| Sora 2 | OpenAI | 비디오 기반 범용 World Model |
| Genie 2 | DeepMind | 상호작용 가능한 3D 환경 생성 |
| Cosmos | NVIDIA | 자율주행/로보틱스 특화 |
| GAIA-2 | Wayve | 자율주행 시뮬레이션 |
| V-JEPA 2 | Meta | 예측 기반 표현 학습 |

**DeepMind Genie 2 (2024.12):**

단일 이미지에서 **상호작용 가능한 3D 환경** 생성:
- 사용자 입력에 반응하는 가상 세계
- 게임 같은 탐험 가능
- 1분 길이의 일관된 환경 유지

**NVIDIA Cosmos (2025.01):**

물리 기반 World Model 플랫폼:
- 실제 물리 법칙 시뮬레이션
- 자율주행 합성 데이터 생성
- 로봇 훈련 환경 제공

### 개념 4: World Model의 아키텍처

**공통 구조:**

1. **인코더**: 관찰(이미지/비디오)을 잠재 표현으로
2. **동역학 모델**: 잠재 공간에서 미래 예측
3. **디코더**: 잠재 표현을 다시 관찰로

**주요 아키텍처 패러다임:**

**1. 재구성 기반 (Reconstruction-based)**
```
관찰 → 인코더 → 잠재 상태 → 동역학 → 미래 잠재 → 디코더 → 예측 영상
                    ↑
                  행동
```
- 예: Dreamer, DreamerV3
- 장점: 직관적, 시각화 가능
- 단점: 재구성 병목

**2. 예측 기반 (Prediction-based)**
```
관찰 → 인코더 → 잠재 상태 → 예측기 → 미래 잠재
                    ↑           ↓
                  행동      (재구성 없이 직접 사용)
```
- 예: V-JEPA, BYOL
- 장점: 더 풍부한 표현 학습
- 단점: 해석 어려움

**3. Diffusion 기반**
```
관찰 + 행동 + 노이즈 → Diffusion 모델 → 미래 관찰
```
- 예: Sora, UniSim
- 장점: 고품질 생성
- 단점: 샘플링 느림

### 개념 5: 물리 법칙을 정말 학습할까?

> 💡 **비유**: 아이가 공을 떨어뜨리는 것을 수천 번 보면 "물체는 아래로 떨어진다"는 것을 배웁니다. 하지만 그게 중력 법칙 $F = ma$를 이해하는 것과 같을까요?

**찬성 관점:**

- Sora가 복잡한 물리 시뮬레이션을 보여줌
- 학습 데이터에 없는 상황도 예측
- "창발적 물리 이해"의 증거

**반대 관점:**

최근 연구(2024-2025)는 비판적 시각을 제시합니다:

**"How Far is Video Generation from World Model"** 논문:
- 비디오 모델은 **통계적 패턴**을 학습
- 진정한 물리 법칙 이해가 아닌 **근사**
- 새로운 상황에서 실패하는 경우 다수

**실패 사례:**

| 상황 | 기대 | 실제 |
|------|------|------|
| 공이 벽에 부딪힘 | 반사 | 통과하거나 사라짐 |
| 물이 컵에서 쏟아짐 | 아래로 흐름 | 방향 불일치 |
| 그림자 | 광원 방향과 일치 | 불일치하는 경우 |

**현실적 결론:**

현재 World Models는 **근사적 물리 이해**를 보여주지만, 뉴턴 역학 같은 **명시적 물리 법칙**을 내재화한 것은 아닙니다. 하지만 실용적 목적으로는 충분히 유용합니다.

### 개념 6: World Models의 응용

**1. 자율주행 시뮬레이션:**

```
실제 주행 데이터 → World Model 학습 → 가상 시나리오 생성
                                           ↓
                                    자율주행 AI 훈련
```

- **장점**: 희귀 상황(사고 직전) 무한 생성
- **예**: NVIDIA Cosmos, Wayve GAIA-2, Tesla FSD 시뮬레이터

**2. 로봇 훈련:**

실제 로봇 훈련의 한계:
- 비용: 로봇 파손 위험
- 시간: 실시간으로만 진행
- 다양성: 제한된 환경

World Model 기반 훈련:
- **안전**: 가상에서 실패해도 OK
- **속도**: 병렬 시뮬레이션
- **다양성**: 무한한 환경 변형

**3. 게임/메타버스:**

Genie 2 같은 모델로:
- 단일 이미지에서 탐험 가능한 세계 생성
- 무한한 레벨 생성
- 사용자 맞춤형 환경

## 실습: World Model 개념 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleWorldModel(nn.Module):
    """
    간단한 World Model 구현

    관찰(이미지)과 행동을 받아 다음 관찰을 예측
    """
    def __init__(
        self,
        obs_dim: int = 64,      # 관찰 잠재 차원
        action_dim: int = 4,    # 행동 차원
        hidden_dim: int = 256,
        latent_dim: int = 128
    ):
        super().__init__()

        # 인코더: 이미지 → 잠재 상태
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)
        )

        # 동역학 모델: (잠재 상태, 행동) → 다음 잠재 상태
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 디코더: 잠재 상태 → 이미지
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # 보상 예측 (선택적)
        self.reward_head = nn.Linear(latent_dim, 1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """관찰을 잠재 상태로 인코딩"""
        return self.encoder(obs)

    def predict_next(
        self,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """다음 잠재 상태 예측"""
        combined = torch.cat([latent, action], dim=-1)
        return self.dynamics(combined)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """잠재 상태를 이미지로 디코딩"""
        return self.decoder(latent)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> dict:
        """
        한 스텝 예측

        Args:
            obs: (B, 3, 64, 64) 현재 관찰
            action: (B, action_dim) 취한 행동

        Returns:
            next_obs_pred: (B, 3, 64, 64) 예측된 다음 관찰
            next_latent: (B, latent_dim) 다음 잠재 상태
            reward_pred: (B, 1) 예측된 보상
        """
        latent = self.encode(obs)
        next_latent = self.predict_next(latent, action)
        next_obs_pred = self.decode(next_latent)
        reward_pred = self.reward_head(next_latent)

        return {
            'next_obs': next_obs_pred,
            'next_latent': next_latent,
            'reward': reward_pred,
            'current_latent': latent
        }

    def imagine(
        self,
        initial_obs: torch.Tensor,
        action_sequence: torch.Tensor,
        horizon: int = None
    ) -> list:
        """
        상상(imagination): 행동 시퀀스에 따른 미래 예측

        Args:
            initial_obs: (B, 3, 64, 64) 초기 관찰
            action_sequence: (B, T, action_dim) 행동 시퀀스
            horizon: 예측 길이 (None이면 action_sequence 길이)

        Returns:
            imagined_obs: 예측된 관찰 시퀀스
        """
        if horizon is None:
            horizon = action_sequence.shape[1]

        latent = self.encode(initial_obs)
        imagined = [self.decode(latent)]

        for t in range(horizon):
            action = action_sequence[:, t]
            latent = self.predict_next(latent, action)
            imagined.append(self.decode(latent))

        return torch.stack(imagined, dim=1)  # (B, T+1, 3, 64, 64)


class WorldModelTrainer:
    """World Model 학습 도우미"""

    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor = None
    ):
        """
        한 스텝 학습

        Args:
            obs: 현재 관찰
            action: 취한 행동
            next_obs: 실제 다음 관찰
            reward: 실제 보상 (선택적)
        """
        self.optimizer.zero_grad()

        # 예측
        outputs = self.model(obs, action)

        # 재구성 손실
        recon_loss = F.mse_loss(outputs['next_obs'], next_obs)

        # 보상 예측 손실 (있는 경우)
        total_loss = recon_loss
        if reward is not None:
            reward_loss = F.mse_loss(outputs['reward'].squeeze(), reward)
            total_loss = total_loss + 0.1 * reward_loss

        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item()
        }


# 테스트
if __name__ == "__main__":
    # 모델 생성
    model = SimpleWorldModel(action_dim=4, latent_dim=128)
    print(f"World Model 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 더미 입력
    batch_size = 8
    obs = torch.randn(batch_size, 3, 64, 64)
    action = torch.randn(batch_size, 4)
    action_seq = torch.randn(batch_size, 10, 4)  # 10스텝 행동

    # 단일 스텝 예측
    outputs = model(obs, action)
    print(f"다음 관찰 예측: {outputs['next_obs'].shape}")

    # 상상 (10스텝 미래)
    imagined = model.imagine(obs, action_seq)
    print(f"상상된 시퀀스: {imagined.shape}")  # (8, 11, 3, 64, 64)
```

```python
# Dreamer 스타일 World Model (RSSM 기반)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RSSM(nn.Module):
    """
    Recurrent State-Space Model (Dreamer의 핵심)

    확률적 상태 + 결정적 상태를 분리하여
    다양한 미래 시나리오 생성 가능
    """
    def __init__(
        self,
        stoch_dim: int = 32,     # 확률적 상태 차원
        deter_dim: int = 256,    # 결정적 상태 차원
        hidden_dim: int = 256,
        action_dim: int = 4,
        embed_dim: int = 256     # 관찰 임베딩 차원
    ):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # 결정적 상태 업데이트 (GRU)
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)

        # Prior: 이전 상태에서 다음 확률적 상태 예측
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2)  # mean, logvar
        )

        # Posterior: 관찰까지 고려한 실제 상태 추론
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )

    def initial_state(self, batch_size: int, device):
        """초기 상태 생성"""
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim, device=device)
        }

    def prior(self, state: dict, action: torch.Tensor) -> dict:
        """
        Prior 예측: 행동만으로 다음 상태 예측 (상상에 사용)
        """
        # 결정적 상태 업데이트
        gru_input = torch.cat([state['stoch'], action], dim=-1)
        deter = self.gru(gru_input, state['deter'])

        # 확률적 상태 예측
        prior_params = self.prior_net(deter)
        mean, logvar = prior_params.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1

        # 샘플링 (reparameterization trick)
        stoch = mean + std * torch.randn_like(std)

        return {
            'deter': deter,
            'stoch': stoch,
            'mean': mean,
            'std': std
        }

    def posterior(
        self,
        state: dict,
        action: torch.Tensor,
        obs_embed: torch.Tensor
    ) -> dict:
        """
        Posterior 추론: 실제 관찰을 보고 상태 추론 (학습에 사용)
        """
        # 결정적 상태 업데이트
        gru_input = torch.cat([state['stoch'], action], dim=-1)
        deter = self.gru(gru_input, state['deter'])

        # 관찰 정보를 포함한 확률적 상태 추론
        posterior_input = torch.cat([deter, obs_embed], dim=-1)
        posterior_params = self.posterior_net(posterior_input)
        mean, logvar = posterior_params.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1

        stoch = mean + std * torch.randn_like(std)

        return {
            'deter': deter,
            'stoch': stoch,
            'mean': mean,
            'std': std
        }

    def imagine_trajectory(
        self,
        initial_state: dict,
        actions: torch.Tensor,
        horizon: int
    ) -> list:
        """
        상상 궤적 생성: prior만 사용하여 미래 상태 시퀀스 예측
        """
        states = [initial_state]
        state = initial_state

        for t in range(horizon):
            state = self.prior(state, actions[:, t])
            states.append(state)

        return states


def compute_kl_loss(prior_state: dict, posterior_state: dict) -> torch.Tensor:
    """Prior와 Posterior 사이의 KL Divergence"""
    prior_dist = Normal(prior_state['mean'], prior_state['std'])
    posterior_dist = Normal(posterior_state['mean'], posterior_state['std'])

    kl = torch.distributions.kl_divergence(posterior_dist, prior_dist)
    return kl.sum(dim=-1).mean()


# 사용 예시
if __name__ == "__main__":
    rssm = RSSM(stoch_dim=32, deter_dim=256, action_dim=4, embed_dim=256)

    batch_size = 16
    device = 'cpu'

    # 초기 상태
    state = rssm.initial_state(batch_size, device)

    # 관찰 임베딩 (인코더 출력 가정)
    obs_embed = torch.randn(batch_size, 256)
    action = torch.randn(batch_size, 4)

    # Posterior (학습 시)
    posterior = rssm.posterior(state, action, obs_embed)
    print(f"Posterior 상태: deter={posterior['deter'].shape}, stoch={posterior['stoch'].shape}")

    # Prior (상상 시)
    prior = rssm.prior(state, action)
    print(f"Prior 상태: deter={prior['deter'].shape}, stoch={prior['stoch'].shape}")

    # KL Loss
    kl_loss = compute_kl_loss(prior, posterior)
    print(f"KL Loss: {kl_loss.item():.4f}")
```

## 더 깊이 알아보기

### World Model의 역사

World Model이라는 개념은 1989년 Jürgen Schmidhuber의 연구까지 거슬러 올라갑니다. 하지만 현대적 형태로 주목받은 것은 2018년 David Ha와 Schmidhuber의 **"World Models"** 논문입니다.

> 💡 **알고 계셨나요?**: 2018년 "World Models" 논문에서는 VAE + RNN으로 간단한 레이싱 게임 환경을 학습했습니다. 놀랍게도 에이전트가 **완전히 꿈속에서(World Model 내에서만)** 학습해도 실제 게임에서 좋은 성능을 보였습니다!

### Sora는 진정한 World Model인가?

이 질문은 2024-2025년 AI 커뮤니티의 가장 뜨거운 논쟁 주제 중 하나입니다.

**"Yes" 진영:**
- 물체 영속성, 3D 일관성 등 물리 이해의 증거
- 학습 데이터에 없는 새로운 상황도 합리적으로 생성

**"No" 진영:**
- 물리 법칙 위반 사례 다수 존재
- **상관관계 학습 vs 인과관계 이해**의 차이
- 상호작용 불가 (액션을 받아 시뮬레이션 X)

> 🔥 **실무 팁**: 현재 비디오 생성 모델을 World Model로 사용할 때는 **제한된 도메인**에서 시작하세요. 자율주행이나 로보틱스처럼 **물리 법칙이 명확한 영역**에서 더 안정적으로 동작합니다.

### World Model의 미래

**단기 전망 (2025-2027):**
- 자율주행 시뮬레이터 상용화 (NVIDIA Cosmos)
- 게임 레벨 자동 생성 (Genie 2 후속)
- 로봇 사전 훈련 필수 도구화

**장기 전망:**
- **범용 World Model**: 모든 도메인 통합
- **물리 엔진과 융합**: 학습 + 명시적 물리
- **AGI의 핵심 컴포넌트**: 계획과 추론의 기반

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Sora가 물리를 완벽히 이해한다" — Sora는 많은 경우 놀라운 물리 시뮬레이션을 보여주지만, 여전히 물리 법칙을 위반하는 경우가 많습니다. "통계적 물리 근사"에 가깝습니다.

> 💡 **알고 계셨나요?**: DeepMind의 Genie 2는 **단일 이미지**에서 1분 길이의 탐험 가능한 3D 환경을 생성합니다. 별도의 게임 엔진 없이 AI만으로 상호작용 가능한 세계를 만드는 거죠!

> 🔥 **실무 팁**: World Model을 실제 응용에 사용할 때는 **불확실성 추정**이 중요합니다. RSSM처럼 확률적 상태를 사용하면 예측의 신뢰도를 함께 얻을 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| World Model | 환경의 역학을 학습한 내부 시뮬레이터 |
| Sora as Simulator | 비디오 생성을 통한 세계 시뮬레이션 |
| RSSM | 확률적+결정적 상태로 다양한 미래 예측 |
| Genie 2 | 이미지에서 상호작용 가능한 환경 생성 |
| Cosmos | NVIDIA의 물리 기반 World Model 플랫폼 |

## 다음 섹션 미리보기

World Model이 가상 세계를 시뮬레이션한다면, 그 다음은 **실제 세계에서 행동하는 AI**입니다. [Embodied AI](./03-embodied-ai.md)에서는 RT-2, PaLM-E 같은 Vision-Language-Action 모델이 어떻게 로봇을 제어하는지 알아봅니다.

## 참고 자료

- [Sora 기술 보고서](https://openai.com/index/video-generation-models-as-world-simulators/) - OpenAI 공식
- [Sora 2 발표](https://openai.com/index/sora-2/) - 2024년 12월 업데이트
- [World Model Survey (ACM 2025)](https://github.com/tsinghua-fib-lab/World-Model) - 포괄적 서베이
- [Genie 2 프로젝트](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) - DeepMind
- [How Far is Video Generation from World Model](https://phyworld.github.io/) - 물리 법칙 관점 분석
- [Dreamer 시리즈 논문](https://danijar.com/project/dreamerv3/) - RSSM 기반 강화학습
