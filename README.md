# 컴퓨터 비전 마스터 가이드

> 픽셀의 이해부터 멀티모달 AI까지 완전 정복

19챕터 93섹션으로 구성된 A-to-Z 컴퓨터 비전 튜토리얼입니다.

---

## Part 1: 기초 (입문~초급)

### Ch1. 이미지의 이해
- 01\. 이미지란 무엇인가 — 픽셀, 해상도, 비트 깊이의 이해
- 02\. 색상 공간의 이해 — RGB, HSV, LAB 등 다양한 색상 표현 방식
- 03\. 이미지 형식과 압축 — JPEG, PNG, WebP 등 형식별 특성과 압축 원리

### Ch2. 전통적 컴퓨터 비전
- 01\. OpenCV 시작하기 — OpenCV 설치와 기본 이미지 입출력
- 02\. 필터와 커널 — 블러, 샤프닝, 에지 검출 필터
- 03\. 에지 검출 — Sobel, Canny, Laplacian 에지 검출
- 04\. 특징점 검출 — SIFT, SURF, ORB 특징 추출기
- 05\. 형태학적 연산 — 침식, 팽창, 열기, 닫기 연산

### Ch3. 딥러닝 기초
- 01\. 신경망의 구조 — 뉴런, 레이어, 가중치의 이해
- 02\. 활성화 함수 — ReLU, Sigmoid, Tanh, GELU 비교
- 03\. 역전파 알고리즘 — 경사하강법과 역전파의 수학적 이해
- 04\. 손실 함수와 옵티마이저 — Cross-Entropy, MSE, Adam, SGD
- 05\. PyTorch 기초 — 텐서, 자동미분, 모델 구축

### Ch4. CNN 핵심 개념
- 01\. 합성곱 연산 — 커널, 스트라이드, 패딩의 이해
- 02\. 풀링 레이어 — Max Pooling, Average Pooling, Global Pooling
- 03\. 배치 정규화 — BatchNorm, LayerNorm, GroupNorm
- 04\. 정규화 기법 — Dropout, Weight Decay, Data Augmentation

---

## Part 2: 핵심 (중급)

### Ch5. CNN 아키텍처의 진화
- 01\. LeNet과 AlexNet — CNN의 탄생과 딥러닝 혁명의 시작
- 02\. VGG와 GoogLeNet — 깊이의 힘과 Inception 모듈
- 03\. ResNet과 Skip Connection — 잔차 학습으로 깊은 네트워크 훈련
- 04\. DenseNet과 SENet — 밀집 연결과 채널 어텐션
- 05\. EfficientNet — 복합 스케일링으로 효율성 극대화
- 06\. ConvNeXt — Transformer 시대의 순수 CNN

### Ch6. 이미지 분류 실전
- 01\. MNIST 손글씨 분류 — 첫 번째 딥러닝 프로젝트
- 02\. CIFAR-10 분류 — 컬러 이미지 분류 도전
- 03\. 전이 학습 — 사전 학습 모델 활용법
- 04\. 파인 튜닝 전략 — 효과적인 모델 미세 조정
- 05\. 데이터 증강 — Albumentations, RandAugment, MixUp

### Ch7. 객체 탐지
- 01\. 객체 탐지 기초 — 바운딩 박스, IoU, NMS 이해
- 02\. R-CNN 계열 — R-CNN, Fast R-CNN, Faster R-CNN
- 03\. YOLO 시리즈 — YOLOv5부터 YOLOv11까지
- 04\. Anchor-Free 탐지기 — FCOS, CenterNet, CornerNet
- 05\. DETR과 Transformer 기반 탐지 — End-to-End 객체 탐지

### Ch8. 이미지 분할
- 01\. 시맨틱 세그멘테이션 — FCN, U-Net, DeepLab
- 02\. 인스턴스 세그멘테이션 — Mask R-CNN, YOLACT
- 03\. 파놉틱 세그멘테이션 — 시맨틱과 인스턴스의 통합
- 04\. Segment Anything Model — 프롬프트 기반 범용 분할 모델

### Ch9. Vision Transformer
- 01\. 어텐션 메커니즘 — Self-Attention과 Multi-Head Attention
- 02\. Transformer 아키텍처 — Encoder-Decoder 구조의 이해
- 03\. Vision Transformer (ViT) — 이미지를 패치로 분할하여 처리
- 04\. Swin Transformer — 계층적 윈도우 기반 어텐션
- 05\. 하이브리드 모델들 — CNN과 Transformer의 결합

---

## Part 3: 고급

### Ch10. Vision-Language 모델
- 01\. 멀티모달 학습 개론 — 시각-언어 통합의 기초
- 02\. CLIP — 대조 학습으로 이미지-텍스트 연결
- 03\. BLIP과 BLIP-2 — 부트스트래핑 기반 사전학습
- 04\. LLaVA — 시각적 지시 튜닝
- 05\. GPT-4V와 Gemini Vision — 상용 멀티모달 LLM 활용

### Ch11. 생성 모델 기초
- 01\. 생성 모델 개론 — 판별 vs 생성 모델
- 02\. Variational Autoencoder — 잠재 공간과 재구성
- 03\. GAN 기초 — 생성자와 판별자의 적대적 학습
- 04\. GAN 변형들 — DCGAN, StyleGAN, CycleGAN
- 05\. GAN 응용 — 이미지 편집, 스타일 변환

### Ch12. Diffusion 모델
- 01\. Diffusion 이론 — 전방/역방향 프로세스의 수학
- 02\. DDPM — Denoising Diffusion Probabilistic Models
- 03\. DDIM과 샘플링 가속 — 빠른 샘플링 기법들
- 04\. U-Net 아키텍처 — 노이즈 예측 네트워크 구조
- 05\. Classifier-Free Guidance — 조건부 생성과 가이던스 스케일
- 06\. Latent Diffusion — 잠재 공간에서의 확산

### Ch13. Stable Diffusion 심화
- 01\. Stable Diffusion 아키텍처 — VAE, U-Net, CLIP 텍스트 인코더
- 02\. SD 1.5 vs SDXL — 모델 버전별 차이점
- 03\. 프롬프트 엔지니어링 — 효과적인 프롬프트 작성법
- 04\. 샘플러 가이드 — Euler, DPM++, UniPC 비교
- 05\. FLUX 모델 — 차세대 Diffusion Transformer
- 06\. SD3와 미래 방향 — MMDiT 아키텍처와 최신 동향

### Ch14. 생성 AI 실전
- 01\. LoRA 학습 — 효율적인 모델 미세조정
- 02\. DreamBooth — 개인화된 이미지 생성
- 03\. ControlNet — 포즈, 에지, 깊이 기반 제어
- 04\. IP-Adapter — 이미지 프롬프트 활용
- 05\. ComfyUI 워크플로우 — 노드 기반 이미지 생성
- 06\. 인페인팅과 아웃페인팅 — 이미지 부분 수정과 확장

---

## Part 4: 전문가/박사급

### Ch15. 비디오 생성
- 01\. 비디오 Diffusion 기초 — 시간 축으로의 확장
- 02\. AnimateDiff — 이미지 모델의 비디오 확장
- 03\. Stable Video Diffusion — 이미지-투-비디오 생성
- 04\. Sora와 대규모 비디오 모델 — Diffusion Transformer 기반 비디오

### Ch16. 3D 컴퓨터 비전
- 01\. 깊이 추정 — 단안/스테레오 깊이 추정
- 02\. 포인트 클라우드 — PointNet, PointNet++ 이해
- 03\. 카메라 기하학 — 내부/외부 파라미터, 에피폴라 기하
- 04\. SLAM 기초 — 동시적 위치추정 및 지도작성
- 05\. 3D 복원 — Structure from Motion, MVS

### Ch17. Neural Rendering
- 01\. NeRF 기초 — Neural Radiance Fields의 원리
- 02\. NeRF 변형들 — Instant-NGP, Mip-NeRF, Nerfacto
- 03\. 3D Gaussian Splatting 기초 — 가우시안 기반 실시간 렌더링
- 04\. 3DGS 심화 — 동적 장면, 아바타 생성
- 05\. Text-to-3D — DreamFusion, Zero-1-to-3

### Ch18. 멀티모달 AI 최전선
- 01\. 통합 멀티모달 모델 — 이미지-텍스트-오디오 통합
- 02\. World Models — 물리 세계 이해와 시뮬레이션
- 03\. Embodied AI — 로봇 비전과 액션
- 04\. 미래 연구 방향 — AGI를 향한 비전 기술

---

## Part 5: 실무

### Ch19. 배포와 최적화
- 01\. 모델 최적화 — 양자화, 프루닝, 지식 증류
- 02\. ONNX와 TensorRT — 추론 가속화
- 03\. 엣지 배포 — Jetson, 라즈베리파이, 모바일
- 04\. CV MLOps — 학습 파이프라인과 모니터링
- 05\. 모델 서빙 — Triton, TorchServe, FastAPI
