# NeRF 변형들

> Instant-NGP, Mip-NeRF, Nerfacto

## 개요

이 섹션에서는 원본 NeRF의 한계를 극복한 주요 변형 모델들을 살펴봅니다. **1000배 빠른 학습**을 달성한 Instant-NGP, **멀티스케일 렌더링 품질**을 개선한 Mip-NeRF, 그리고 실무에서 바로 쓸 수 있는 **Nerfacto**까지 각각의 핵심 아이디어와 구현 방법을 배웁니다.

**선수 지식**:
- [NeRF 기초](./01-nerf-basics.md)의 볼륨 렌더링과 MLP 구조
- 해시 테이블의 기본 개념 (선택)

**학습 목표**:
- Instant-NGP의 해시 인코딩 원리 이해하기
- Mip-NeRF의 원뿔 트레이싱과 IPE 개념 파악하기
- Nerfacto의 실용적 조합과 Nerfstudio 활용법 익히기

## 왜 알아야 할까?

[NeRF 기초](./01-nerf-basics.md)에서 배웠듯이, 원본 NeRF는 혁신적이지만 실용성에 한계가 있었습니다:

| 문제 | 원본 NeRF | 해결한 변형 |
|------|-----------|------------|
| 느린 학습 (1~2일) | MLP 전체 학습 | **Instant-NGP**: 해시 인코딩으로 수 초 만에 |
| 멀티스케일 앨리어싱 | 점 샘플링 | **Mip-NeRF**: 원뿔 기반 적분 |
| 무한 장면 한계 | bounded scene | **Mip-NeRF 360**: unbounded 처리 |
| 복잡한 설정 | 연구용 코드 | **Nerfacto**: 모듈화된 프레임워크 |

이 변형들을 이해하면 실제 프로젝트에서 상황에 맞는 최적의 방법을 선택할 수 있습니다.

## 핵심 개념

### 개념 1: Instant-NGP - 해시 인코딩의 마법

> 💡 **비유**: 도서관에서 책을 찾는다고 생각해보세요. 원본 NeRF는 책을 찾으려면 전체 서가를 돌아다녀야 하는 것처럼 MLP 전체를 통과합니다. Instant-NGP는 도서 색인표(해시 테이블)를 만들어서 원하는 정보에 바로 접근합니다.

NVIDIA의 **Instant Neural Graphics Primitives(2022)**는 NeRF 학습 속도를 **1000배** 향상시켰습니다. 핵심은 **Multiresolution Hash Encoding**입니다.

**구조:**

1. **다중 해상도 격자**: L개의 서로 다른 해상도 레벨
2. **해시 테이블**: 각 레벨마다 T개 엔트리를 가진 해시 테이블
3. **특징 벡터**: 각 엔트리에 F차원 학습 가능한 벡터 저장
4. **작은 MLP**: 해시 특징을 받아 색상/밀도 출력

**해시 인코딩 과정:**

입력 위치 $\mathbf{x}$가 주어지면:
1. 각 해상도 레벨 $l$에서 $\mathbf{x}$가 속한 복셀의 8개 꼭짓점 찾기
2. 꼭짓점 좌표를 해시 함수로 테이블 인덱스 계산
3. 해당 인덱스에서 특징 벡터 조회
4. 8개 특징을 trilinear interpolation으로 보간
5. 모든 레벨의 특징을 concatenate

**해시 함수:**

$$h(\mathbf{x}) = \left(\bigoplus_{i=1}^{d} x_i \cdot \pi_i \right) \mod T$$

여기서 $\bigoplus$는 XOR 연산, $\pi_i$는 큰 소수입니다.

**왜 빠른가?**
- MLP 크기 대폭 축소 (256→64 유닛)
- 해시 테이블이 공간 정보를 직접 저장 → MLP는 보간만
- GPU의 병렬 해시 조회 최적화

### 개념 2: Mip-NeRF - 앨리어싱 없는 렌더링

> 💡 **비유**: 카메라로 멀리 있는 체크무늬를 찍으면 이상한 무늬(모아레)가 생깁니다. 이건 점 하나로 샘플링해서 생기는 문제예요. Mip-NeRF는 점 대신 "원뿔"로 샘플링해서 영역 전체의 평균을 구합니다.

**Mip-NeRF(2021)**는 멀티스케일 렌더링의 **앨리어싱 문제**를 해결했습니다.

**문제 상황:**
- 가까운 물체: 한 픽셀이 작은 영역 커버 → 디테일 필요
- 먼 물체: 한 픽셀이 넓은 영역 커버 → 평균 필요
- 원본 NeRF: 점 샘플링으로 스케일 무시 → 앨리어싱

**해결책: Cone Tracing**

광선(ray) 대신 **원뿔(cone)**을 쏴서 각 샘플이 차지하는 영역을 고려합니다:

1. 픽셀 크기에 따라 원뿔 각도 결정
2. 원뿔 내 영역을 **3D 가우시안**으로 근사
3. **Integrated Positional Encoding(IPE)**: 가우시안 영역의 포지셔널 인코딩 적분

**IPE의 핵심:**

일반 PE:
$$\gamma(\mathbf{x}) = [\sin(2^l \pi \mathbf{x}), \cos(2^l \pi \mathbf{x})]_{l=0}^{L-1}$$

IPE (가우시안 $\mathcal{N}(\mu, \Sigma)$에 대해):
$$\gamma^{IPE}(\mu, \Sigma) = \mathbb{E}_{\mathbf{x} \sim \mathcal{N}}[\gamma(\mathbf{x})]$$

고주파 성분($2^l$이 클 때)은 가우시안 평균에서 자연스럽게 감쇠됩니다.

### 개념 3: Mip-NeRF 360 - 무한 장면 처리

**Mip-NeRF 360(2022)**은 360도 촬영된 **무한 장면(unbounded scene)**을 처리합니다.

**문제:**
- 실외 장면은 배경이 무한대까지 뻗어 있음
- 원본 NeRF의 near-far 바운딩 적용 불가

**해결책:**

1. **Scene Contraction**: 무한 공간을 유한 공간으로 수축
   - 단위 구 내부: 그대로 유지
   - 단위 구 외부: 반지름에 반비례하게 압축

$$\text{contract}(\mathbf{x}) = \begin{cases}
\mathbf{x} & \|\mathbf{x}\| \leq 1 \\
\left(2 - \frac{1}{\|\mathbf{x}\|}\right) \frac{\mathbf{x}}{\|\mathbf{x}\|} & \|\mathbf{x}\| > 1
\end{cases}$$

2. **Proposal Network**: 밀도 분포를 예측하는 가벼운 네트워크로 효율적 샘플링

3. **Distortion Loss**: 가중치가 한 곳에 집중되도록 정규화

### 개념 4: Nerfacto - 실전을 위한 통합 솔루션

> 💡 **비유**: 요리에 비유하면, Instant-NGP는 빠른 전자레인지, Mip-NeRF는 고품질 오븐입니다. Nerfacto는 이 둘의 장점을 조합한 **만능 조리기구**예요.

**Nerfacto**는 Nerfstudio 팀이 만든 실용적 NeRF 모델로, 여러 기법의 장점을 조합했습니다:

**핵심 구성요소:**

| 출처 | 기법 | 역할 |
|------|------|------|
| Instant-NGP | Hash Encoding | 빠른 학습 |
| Mip-NeRF 360 | Proposal Sampling | 효율적 샘플링 |
| Ref-NeRF | Appearance Embedding | 조명 변화 처리 |
| Mip-NeRF | Integrated PE | 앨리어싱 감소 |

**Nerfstudio 프레임워크:**

```
Nerfstudio
├── Data Parsing (COLMAP, Record3D, ...)
├── Models
│   ├── nerfacto (기본)
│   ├── instant-ngp
│   ├── mipnerf
│   └── ...
├── Viewer (실시간 웹 뷰어)
└── Export (mesh, point cloud, video)
```

### 개념 5: 기타 주요 변형들

**TensoRF (2022)**: 4D 텐서 분해로 컴팩트한 표현

- 밀도와 특징을 VM(Vector-Matrix) 분해로 표현
- 메모리 효율적이면서 품질 유지

**Plenoxels (2022)**: 신경망 없이 복셀 격자만 사용

- Spherical Harmonics로 뷰 의존적 색상 표현
- 수 분 만에 학습, but 메모리 사용량 큼

**3D Gaussian Splatting (2023)**: NeRF의 대안 (다음 섹션에서 자세히)

- 명시적 가우시안으로 장면 표현
- 실시간 렌더링 가능

## 실습: Nerfstudio로 NeRF 훈련하기

```python
# Nerfstudio 설치 및 기본 사용법
# 터미널에서 실행

# 1. 설치 (CUDA 11.8 기준)
"""
pip install nerfstudio

# 또는 conda 환경
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio
"""
```

```bash
# 2. 데이터 준비 (COLMAP 형식)
# 직접 촬영한 영상이나 이미지 폴더 사용
ns-process-data images --data ./my_images --output-dir ./processed_data

# 또는 예제 데이터 다운로드
ns-download-data nerfstudio --capture-name=poster
```

```bash
# 3. 학습 시작
# Nerfacto (기본, 균형 잡힌 품질/속도)
ns-train nerfacto --data ./processed_data

# Instant-NGP (빠른 학습)
ns-train instant-ngp --data ./processed_data

# 커스텀 설정
ns-train nerfacto \
    --data ./processed_data \
    --max-num-iterations 30000 \
    --pipeline.model.near-plane 0.05 \
    --pipeline.model.far-plane 1000.0
```

```bash
# 4. 실시간 뷰어 접속
# 학습 중 자동으로 뷰어 URL 출력됨
# 브라우저에서 http://localhost:7007 접속

# 5. 결과 내보내기
# 렌더링된 비디오
ns-render camera-path \
    --load-config outputs/poster/nerfacto/config.yml \
    --camera-path-filename camera_path.json \
    --output-path renders/video.mp4

# 포인트 클라우드 추출
ns-export pointcloud \
    --load-config outputs/poster/nerfacto/config.yml \
    --output-dir exports/pointcloud
```

```python
# Instant-NGP 해시 인코딩 직접 구현 (간략화)
import torch
import torch.nn as nn
import numpy as np

class HashEncoding(nn.Module):
    """
    Multiresolution Hash Encoding (Instant-NGP 스타일)
    """
    def __init__(
        self,
        n_levels=16,           # 해상도 레벨 수
        n_features=2,          # 각 엔트리의 특징 차원
        log2_hashmap_size=19,  # 해시 테이블 크기 = 2^19
        base_resolution=16,    # 최소 해상도
        max_resolution=2048    # 최대 해상도
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.hashmap_size = 2 ** log2_hashmap_size

        # 해상도 스케일 계산 (geometric progression)
        b = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1))
        self.resolutions = [int(base_resolution * (b ** l)) for l in range(n_levels)]

        # 각 레벨의 해시 테이블 (학습 가능한 파라미터)
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, n_features) * 0.01)
            for _ in range(n_levels)
        ])

        # 해시 프라임 (XOR 해싱용)
        self.primes = [1, 2654435761, 805459861]

    def hash_function(self, coords):
        """
        공간 좌표를 해시 테이블 인덱스로 변환

        Args:
            coords: (batch, 3) 정수 격자 좌표
        Returns:
            (batch,) 해시 인덱스
        """
        # XOR 해시
        result = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        for i in range(3):
            result ^= coords[:, i] * self.primes[i]
        return result % self.hashmap_size

    def forward(self, x):
        """
        Args:
            x: (batch, 3) 정규화된 좌표 [0, 1]^3
        Returns:
            (batch, n_levels * n_features) 인코딩된 특징
        """
        batch_size = x.shape[0]
        outputs = []

        for level, resolution in enumerate(self.resolutions):
            # 해당 해상도로 스케일링
            scaled = x * resolution

            # 복셀의 코너 좌표 (floor, ceil)
            corner_0 = torch.floor(scaled).long()
            corner_1 = corner_0 + 1

            # Trilinear interpolation 가중치
            weights = scaled - corner_0.float()

            # 8개 코너에서 특징 조회
            features = torch.zeros(batch_size, self.n_features, device=x.device)

            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # 코너 좌표
                        corner = torch.stack([
                            corner_0[:, 0] if i == 0 else corner_1[:, 0],
                            corner_0[:, 1] if j == 0 else corner_1[:, 1],
                            corner_0[:, 2] if k == 0 else corner_1[:, 2],
                        ], dim=-1)

                        # 해시 인덱스
                        idx = self.hash_function(corner)

                        # 특징 조회
                        f = self.hash_tables[level][idx]

                        # Trilinear 가중치
                        w = (weights[:, 0] if i == 1 else 1 - weights[:, 0]) * \
                            (weights[:, 1] if j == 1 else 1 - weights[:, 1]) * \
                            (weights[:, 2] if k == 1 else 1 - weights[:, 2])

                        features += w.unsqueeze(-1) * f

            outputs.append(features)

        return torch.cat(outputs, dim=-1)


class InstantNGP(nn.Module):
    """간략화된 Instant-NGP 모델"""
    def __init__(self):
        super().__init__()
        self.hash_encoding = HashEncoding()

        # 작은 MLP (해시 인코딩 덕분에 큰 MLP 불필요)
        encoded_dim = 16 * 2  # n_levels * n_features
        self.mlp = nn.Sequential(
            nn.Linear(encoded_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.sigma_head = nn.Linear(64, 1)
        self.rgb_head = nn.Sequential(
            nn.Linear(64 + 3, 32),  # +3 for view direction
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

    def forward(self, positions, directions):
        # 위치를 [0, 1] 범위로 정규화 (scene bounds에 따라)
        normalized = (positions + 1) / 2  # 예: [-1, 1] → [0, 1]

        # 해시 인코딩
        encoded = self.hash_encoding(normalized)

        # MLP
        features = self.mlp(encoded)

        # 밀도 (위치만 의존)
        sigma = torch.relu(self.sigma_head(features))

        # 색상 (방향도 의존)
        rgb_input = torch.cat([features, directions], dim=-1)
        rgb = self.rgb_head(rgb_input)

        return rgb, sigma


# 테스트
if __name__ == "__main__":
    model = InstantNGP()
    print(f"Instant-NGP 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 원본 NeRF 대비 파라미터 수 비교
    # 원본 NeRF: ~1.2M, Instant-NGP: ~12M (대부분 해시 테이블)
    # 하지만 MLP는 훨씬 작아서 forward가 빠름
```

```python
# Mip-NeRF의 Integrated Positional Encoding 구현
import torch
import torch.nn as nn
import numpy as np

def integrated_positional_encoding(mean, var, num_freqs=10):
    """
    가우시안 영역에 대한 Integrated Positional Encoding

    Args:
        mean: (batch, 3) 가우시안 평균
        var: (batch, 3) 가우시안 분산 (대각 공분산 가정)
        num_freqs: 주파수 수

    Returns:
        (batch, 3 * 2 * num_freqs) IPE 특징
    """
    # 주파수 밴드
    freqs = 2.0 ** torch.arange(num_freqs, device=mean.device)  # [1, 2, 4, ..., 2^(L-1)]

    # mean과 var를 주파수로 스케일링
    # (batch, 3, num_freqs)
    scaled_mean = mean.unsqueeze(-1) * freqs * np.pi
    scaled_var = var.unsqueeze(-1) * (freqs * np.pi) ** 2

    # 가우시안의 sin/cos 기댓값
    # E[sin(x)] where x ~ N(μ, σ²) = sin(μ) * exp(-σ²/2)
    # E[cos(x)] where x ~ N(μ, σ²) = cos(μ) * exp(-σ²/2)
    decay = torch.exp(-0.5 * scaled_var)

    sin_features = torch.sin(scaled_mean) * decay
    cos_features = torch.cos(scaled_mean) * decay

    # 결합 및 reshape
    features = torch.cat([sin_features, cos_features], dim=-1)
    features = features.reshape(mean.shape[0], -1)

    return features


def cast_rays_to_gaussians(rays_o, rays_d, t_vals, radii):
    """
    광선을 원뿔로 확장하고 각 샘플을 3D 가우시안으로 근사

    Args:
        rays_o: (batch, 3) 광선 원점
        rays_d: (batch, 3) 광선 방향
        t_vals: (batch, num_samples) 샘플 깊이 값
        radii: (batch,) 픽셀의 원뿔 반경

    Returns:
        means: (batch, num_samples, 3) 가우시안 평균
        covs: (batch, num_samples, 3, 3) 가우시안 공분산
    """
    num_samples = t_vals.shape[1]

    # 각 샘플의 중심점 (가우시안 평균)
    # t 값의 중점 사용
    t_mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
    t_mids = torch.cat([t_mids, t_vals[:, -1:]], dim=-1)

    means = rays_o.unsqueeze(1) + t_mids.unsqueeze(-1) * rays_d.unsqueeze(1)

    # 원뿔의 가우시안 근사 (Mip-NeRF 논문의 conical frustum)
    # 간략화: 등방성(isotropic) 가우시안 가정

    # 원뿔 반경은 깊이에 비례
    sample_radii = radii.unsqueeze(1) * t_mids

    # 분산 (대각 공분산으로 근사)
    # 실제로는 광선 방향에 따라 비등방성이지만 여기선 간략화
    covs_diag = sample_radii.unsqueeze(-1) ** 2
    covs_diag = covs_diag.expand(-1, -1, 3)

    return means, covs_diag


# 테스트
if __name__ == "__main__":
    batch_size = 32
    num_samples = 64

    # 테스트 입력
    mean = torch.randn(batch_size, 3)
    var = torch.abs(torch.randn(batch_size, 3)) * 0.1

    # IPE 계산
    ipe_features = integrated_positional_encoding(mean, var, num_freqs=10)
    print(f"IPE 출력 차원: {ipe_features.shape}")  # (32, 60)

    # 원뿔 가우시안 테스트
    rays_o = torch.zeros(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    t_vals = torch.linspace(0.1, 5.0, num_samples).unsqueeze(0).expand(batch_size, -1)
    radii = torch.ones(batch_size) * 0.001  # 픽셀 크기에 따른 반경

    means, covs = cast_rays_to_gaussians(rays_o, rays_d, t_vals, radii)
    print(f"가우시안 평균: {means.shape}")  # (32, 64, 3)
    print(f"가우시안 분산: {covs.shape}")    # (32, 64, 3)
```

## 더 깊이 알아보기

### Instant-NGP의 해시 충돌 문제

해시 테이블에서 서로 다른 위치가 같은 인덱스에 매핑되면 **해시 충돌**이 발생합니다. Instant-NGP에서는 이게 오히려 장점이 될 수 있습니다:

1. **저주파 공유**: 비슷한 색상/밀도를 가진 먼 영역이 같은 엔트리 공유 → 메모리 절약
2. **다중 해상도 보완**: 한 레벨에서 충돌해도 다른 레벨에서 구분 가능
3. **학습 시 자연 해결**: 그래디언트가 충돌 위치들의 평균을 학습

> 💡 **알고 계셨나요?**: Instant-NGP의 저자 Thomas Müller는 NVIDIA 연구원으로, 논문 발표 당시 학습 시간을 "수 초"라고 했지만, 실제로 RTX 3090에서 5-15초 정도 걸립니다. 그래도 원본 NeRF의 1-2일에 비하면 1000배 이상 빠른 건 사실이죠!

### Mip-NeRF의 탄생 배경

Mip-NeRF의 "Mip"은 라틴어 "multum in parvo"(작은 공간에 많은 것)의 약자로, 텍스처 매핑의 **Mipmap**에서 따왔습니다. 1983년 Lance Williams가 발명한 mipmapping은 미리 여러 해상도의 텍스처를 저장해두는 기법인데, Mip-NeRF는 이 아이디어를 신경망 렌더링에 적용한 것입니다.

> 🔥 **실무 팁**: Mip-NeRF 360은 드론 촬영이나 자동차 대시캠처럼 360도 주변을 촬영한 영상에 특히 효과적입니다. 반면 실내나 작은 물체는 원본 NeRF 계열이 더 나을 수 있어요.

### NeRF 변형 선택 가이드

| 상황 | 추천 모델 | 이유 |
|------|----------|------|
| 빠른 프로토타이핑 | Instant-NGP | 수 초~분 내 결과 확인 |
| 고품질 결과물 | Mip-NeRF 360 | 앨리어싱 없는 깨끗한 렌더링 |
| 처음 시작 | Nerfacto | 쉬운 설정, 좋은 기본 품질 |
| 메모리 제한 | TensoRF | 텐서 분해로 컴팩트 |
| 연구/커스터마이징 | Nerfstudio | 모듈화된 코드베이스 |

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Instant-NGP가 무조건 최고다" — 속도는 빠르지만 품질은 Mip-NeRF 360이 더 나은 경우가 많습니다. 용도에 따라 선택하세요.

> 💡 **알고 계셨나요?**: Nerfstudio의 Nerfacto는 논문으로 발표된 모델이 아니라, 여러 논문의 기법을 엔지니어링적으로 조합한 것입니다. "논문에 없으니 못 쓴다"는 생각은 버리세요!

> 🔥 **실무 팁**: COLMAP 포즈 추정이 NeRF 품질의 80%를 결정합니다. 데이터 전처리에 시간을 투자하세요. 특히 촬영 시 50% 이상 오버랩되도록 사진을 찍는 게 중요합니다.

## 핵심 정리

| 모델 | 핵심 기법 | 장점 | 단점 |
|------|----------|------|------|
| Instant-NGP | Hash Encoding | 초고속 학습 (5-15초) | 해시 충돌로 인한 아티팩트 가능 |
| Mip-NeRF | Cone Tracing + IPE | 앨리어싱 제거, 고품질 | 학습 시간 오래 걸림 |
| Mip-NeRF 360 | Scene Contraction | 무한 장면 처리 | 복잡한 설정 |
| Nerfacto | 여러 기법 조합 | 균형 잡힌 품질/속도 | 특정 상황 최적화 아님 |
| TensoRF | 텐서 분해 | 메모리 효율 | 큰 장면에서 품질 저하 |

## 다음 섹션 미리보기

NeRF의 다양한 변형들을 살펴보았습니다. 하지만 2023년, NeRF의 대안으로 떠오른 새로운 패러다임이 있습니다. [3D Gaussian Splatting 기초](./03-3dgs-basics.md)에서는 신경망 없이 **명시적 가우시안**으로 장면을 표현하여 **실시간 렌더링(100+ FPS)**을 달성하는 혁신적인 방법을 알아봅니다.

## 참고 자료

- [Instant-NGP 공식 저장소](https://github.com/NVlabs/instant-ngp) - NVIDIA의 공식 구현
- [Instant-NGP 논문 페이지](https://nvlabs.github.io/instant-ngp/) - 인터랙티브 데모
- [Mip-NeRF 프로젝트](https://jonbarron.info/mipnerf/) - Jon Barron의 프로젝트 페이지
- [Mip-NeRF 360 프로젝트](https://jonbarron.info/mipnerf360/) - 무한 장면 처리
- [Nerfstudio 문서](https://docs.nerf.studio/) - 설치부터 학습까지 상세 가이드
- [NeRF at CVPR 2024](https://arxiv.org/abs/2501.13104) - 최신 NeRF 연구 동향 서베이
