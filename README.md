# 컴퓨터 비전 마스터 가이드

> 픽셀의 이해부터 멀티모달 AI까지 완전 정복

**19챕터 93섹션**으로 구성된 A-to-Z 컴퓨터 비전 튜토리얼입니다.

## 튜토리얼 현황

| 항목 | 수량 | 상태 |
|------|------|------|
| 총 챕터 | 19개 | ✅ 완료 |
| 총 섹션 | 93개 | ✅ 완료 |
| Part 1: 기초 | 17개 섹션 | ✅ 완료 |
| Part 2: 핵심 | 25개 섹션 | ✅ 완료 |
| Part 3: 고급 | 28개 섹션 | ✅ 완료 |
| Part 4: 전문가 | 18개 섹션 | ✅ 완료 |
| Part 5: 실무 | 5개 섹션 | ✅ 완료 |

---

## Part 1: 기초 (입문~초급)

### Ch1. 이미지의 이해
- [01. 이미지란 무엇인가](01-foundations/01-what-is-image.md) — 픽셀, 해상도, 비트 깊이의 이해
- [02. 색상 공간의 이해](01-foundations/02-color-spaces.md) — RGB, HSV, LAB 등 다양한 색상 표현 방식
- [03. 이미지 형식과 압축](01-foundations/03-image-formats.md) — JPEG, PNG, WebP 등 형식별 특성과 압축 원리

### Ch2. 전통적 컴퓨터 비전
- [01. OpenCV 시작하기](02-classical-cv/01-opencv-basics.md) — OpenCV 설치와 기본 이미지 입출력
- [02. 필터와 커널](02-classical-cv/02-filters-kernels.md) — 블러, 샤프닝, 에지 검출 필터
- [03. 에지 검출](02-classical-cv/03-edge-detection.md) — Sobel, Canny, Laplacian 에지 검출
- [04. 특징점 검출](02-classical-cv/04-feature-detection.md) — SIFT, SURF, ORB 특징 추출기
- [05. 형태학적 연산](02-classical-cv/05-morphology.md) — 침식, 팽창, 열기, 닫기 연산

### Ch3. 딥러닝 기초
- [01. 신경망의 구조](03-deep-learning-basics/01-neural-network.md) — 뉴런, 레이어, 가중치의 이해
- [02. 활성화 함수](03-deep-learning-basics/02-activation-functions.md) — ReLU, Sigmoid, Tanh, GELU 비교
- [03. 역전파 알고리즘](03-deep-learning-basics/03-backpropagation.md) — 경사하강법과 역전파의 수학적 이해
- [04. 손실 함수와 옵티마이저](03-deep-learning-basics/04-loss-optimizer.md) — Cross-Entropy, MSE, Adam, SGD
- [05. PyTorch 기초](03-deep-learning-basics/05-pytorch-fundamentals.md) — 텐서, 자동미분, 모델 구축

### Ch4. CNN 핵심 개념
- [01. 합성곱 연산](04-cnn-fundamentals/01-convolution.md) — 커널, 스트라이드, 패딩의 이해
- [02. 풀링 레이어](04-cnn-fundamentals/02-pooling.md) — Max Pooling, Average Pooling, Global Pooling
- [03. 배치 정규화](04-cnn-fundamentals/03-batch-normalization.md) — BatchNorm, LayerNorm, GroupNorm
- [04. 정규화 기법](04-cnn-fundamentals/04-regularization.md) — Dropout, Weight Decay, Data Augmentation

---

## Part 2: 핵심 (중급)

### Ch5. CNN 아키텍처의 진화
- [01. LeNet과 AlexNet](05-cnn-architectures/01-lenet-alexnet.md) — CNN의 탄생과 딥러닝 혁명의 시작
- [02. VGG와 GoogLeNet](05-cnn-architectures/02-vgg-googlenet.md) — 깊이의 힘과 Inception 모듈
- [03. ResNet과 Skip Connection](05-cnn-architectures/03-resnet.md) — 잔차 학습으로 깊은 네트워크 훈련
- [04. DenseNet과 SENet](05-cnn-architectures/04-densenet-senet.md) — 밀집 연결과 채널 어텐션
- [05. EfficientNet](05-cnn-architectures/05-efficientnet.md) — 복합 스케일링으로 효율성 극대화
- [06. ConvNeXt](05-cnn-architectures/06-convnext.md) — Transformer 시대의 순수 CNN

### Ch6. 이미지 분류 실전
- [01. MNIST 손글씨 분류](06-image-classification/01-mnist.md) — 첫 번째 딥러닝 프로젝트
- [02. CIFAR-10 분류](06-image-classification/02-cifar10.md) — 컬러 이미지 분류 도전
- [03. 전이 학습](06-image-classification/03-transfer-learning.md) — 사전 학습 모델 활용법
- [04. 파인 튜닝 전략](06-image-classification/04-fine-tuning.md) — 효과적인 모델 미세 조정
- [05. 데이터 증강](06-image-classification/05-data-augmentation.md) — Albumentations, RandAugment, MixUp

### Ch7. 객체 탐지
- [01. 객체 탐지 기초](07-object-detection/01-detection-basics.md) — 바운딩 박스, IoU, NMS 이해
- [02. R-CNN 계열](07-object-detection/02-rcnn-family.md) — R-CNN, Fast R-CNN, Faster R-CNN
- [03. YOLO 시리즈](07-object-detection/03-yolo.md) — YOLOv5부터 YOLOv11까지
- [04. Anchor-Free 탐지기](07-object-detection/04-anchor-free.md) — FCOS, CenterNet, CornerNet
- [05. DETR과 Transformer 기반 탐지](07-object-detection/05-detr.md) — End-to-End 객체 탐지

### Ch8. 이미지 분할
- [01. 시맨틱 세그멘테이션](08-segmentation/01-semantic-segmentation.md) — FCN, U-Net, DeepLab
- [02. 인스턴스 세그멘테이션](08-segmentation/02-instance-segmentation.md) — Mask R-CNN, YOLACT
- [03. 파놉틱 세그멘테이션](08-segmentation/03-panoptic-segmentation.md) — 시맨틱과 인스턴스의 통합
- [04. Segment Anything Model](08-segmentation/04-sam.md) — 프롬프트 기반 범용 분할 모델

### Ch9. Vision Transformer
- [01. 어텐션 메커니즘](09-vision-transformer/01-attention-mechanism.md) — Self-Attention과 Multi-Head Attention
- [02. Transformer 아키텍처](09-vision-transformer/02-transformer-basics.md) — Encoder-Decoder 구조의 이해
- [03. Vision Transformer (ViT)](09-vision-transformer/03-vit.md) — 이미지를 패치로 분할하여 처리
- [04. Swin Transformer](09-vision-transformer/04-swin-transformer.md) — 계층적 윈도우 기반 어텐션
- [05. 하이브리드 모델들](09-vision-transformer/05-hybrid-models.md) — CNN과 Transformer의 결합

---

## Part 3: 고급

### Ch10. Vision-Language 모델
- [01. 멀티모달 학습 개론](10-vision-language/01-multimodal-learning.md) — 시각-언어 통합의 기초
- [02. CLIP](10-vision-language/02-clip.md) — 대조 학습으로 이미지-텍스트 연결
- [03. BLIP과 BLIP-2](10-vision-language/03-blip.md) — 부트스트래핑 기반 사전학습
- [04. LLaVA](10-vision-language/04-llava.md) — 시각적 지시 튜닝
- [05. GPT-4V와 Gemini Vision](10-vision-language/05-gpt4v-gemini.md) — 상용 멀티모달 LLM 활용

### Ch11. 생성 모델 기초
- [01. 생성 모델 개론](11-generative-basics/01-generative-intro.md) — 판별 vs 생성 모델
- [02. Variational Autoencoder](11-generative-basics/02-vae.md) — 잠재 공간과 재구성
- [03. GAN 기초](11-generative-basics/03-gan-basics.md) — 생성자와 판별자의 적대적 학습
- [04. GAN 변형들](11-generative-basics/04-gan-variants.md) — DCGAN, StyleGAN, CycleGAN
- [05. GAN 응용](11-generative-basics/05-gan-applications.md) — 이미지 편집, 스타일 변환

### Ch12. Diffusion 모델
- [01. Diffusion 이론](12-diffusion-models/01-diffusion-theory.md) — 전방/역방향 프로세스의 수학
- [02. DDPM](12-diffusion-models/02-ddpm.md) — Denoising Diffusion Probabilistic Models
- [03. DDIM과 샘플링 가속](12-diffusion-models/03-ddim.md) — 빠른 샘플링 기법들
- [04. U-Net 아키텍처](12-diffusion-models/04-unet-architecture.md) — 노이즈 예측 네트워크 구조
- [05. Classifier-Free Guidance](12-diffusion-models/05-cfg.md) — 조건부 생성과 가이던스 스케일
- [06. Latent Diffusion](12-diffusion-models/06-latent-diffusion.md) — 잠재 공간에서의 확산

### Ch13. Stable Diffusion 심화
- [01. Stable Diffusion 아키텍처](13-stable-diffusion/01-sd-architecture.md) — VAE, U-Net, CLIP 텍스트 인코더
- [02. SD 1.5 vs SDXL](13-stable-diffusion/02-sd15-vs-sdxl.md) — 모델 버전별 차이점
- [03. 프롬프트 엔지니어링](13-stable-diffusion/03-prompting.md) — 효과적인 프롬프트 작성법
- [04. 샘플러 가이드](13-stable-diffusion/04-samplers.md) — Euler, DPM++, UniPC 비교
- [05. FLUX 모델](13-stable-diffusion/05-flux.md) — 차세대 Diffusion Transformer
- [06. SD3와 미래 방향](13-stable-diffusion/06-sd3-future.md) — MMDiT 아키텍처와 최신 동향

### Ch14. 생성 AI 실전
- [01. LoRA 학습](14-generative-practice/01-lora.md) — 효율적인 모델 미세조정
- [02. DreamBooth](14-generative-practice/02-dreambooth.md) — 개인화된 이미지 생성
- [03. ControlNet](14-generative-practice/03-controlnet.md) — 포즈, 에지, 깊이 기반 제어
- [04. IP-Adapter](14-generative-practice/04-ip-adapter.md) — 이미지 프롬프트 활용
- [05. ComfyUI 워크플로우](14-generative-practice/05-comfyui.md) — 노드 기반 이미지 생성
- [06. 인페인팅과 아웃페인팅](14-generative-practice/06-inpainting-outpainting.md) — 이미지 부분 수정과 확장

---

## Part 4: 전문가/박사급

### Ch15. 비디오 생성
- [01. 비디오 Diffusion 기초](15-video-generation/01-video-diffusion.md) — 시간 축으로의 확장
- [02. AnimateDiff](15-video-generation/02-animatediff.md) — 이미지 모델의 비디오 확장
- [03. Stable Video Diffusion](15-video-generation/03-svd.md) — 이미지-투-비디오 생성
- [04. Sora와 대규모 비디오 모델](15-video-generation/04-sora.md) — Diffusion Transformer 기반 비디오

### Ch16. 3D 컴퓨터 비전
- [01. 깊이 추정](16-3d-vision/01-depth-estimation.md) — 단안/스테레오 깊이 추정
- [02. 포인트 클라우드](16-3d-vision/02-point-clouds.md) — PointNet, PointNet++ 이해
- [03. 카메라 기하학](16-3d-vision/03-camera-geometry.md) — 내부/외부 파라미터, 에피폴라 기하
- [04. SLAM 기초](16-3d-vision/04-slam.md) — 동시적 위치추정 및 지도작성
- [05. 3D 복원](16-3d-vision/05-3d-reconstruction.md) — Structure from Motion, MVS

### Ch17. Neural Rendering
- [01. NeRF 기초](17-neural-rendering/01-nerf-basics.md) — Neural Radiance Fields의 원리
- [02. NeRF 변형들](17-neural-rendering/02-nerf-variants.md) — Instant-NGP, Mip-NeRF, Nerfacto
- [03. 3D Gaussian Splatting 기초](17-neural-rendering/03-3dgs-basics.md) — 가우시안 기반 실시간 렌더링
- [04. 3DGS 심화](17-neural-rendering/04-3dgs-advanced.md) — 동적 장면, 아바타 생성
- [05. Text-to-3D](17-neural-rendering/05-text-to-3d.md) — DreamFusion, Zero-1-to-3

### Ch18. 멀티모달 AI 최전선
- [01. 통합 멀티모달 모델](18-multimodal-frontier/01-unified-models.md) — 이미지-텍스트-오디오 통합
- [02. World Models](18-multimodal-frontier/02-world-models.md) — 물리 세계 이해와 시뮬레이션
- [03. Embodied AI](18-multimodal-frontier/03-embodied-ai.md) — 로봇 비전과 액션
- [04. 미래 연구 방향](18-multimodal-frontier/04-future-directions.md) — AGI를 향한 비전 기술

---

## Part 5: 실무

### Ch19. 배포와 최적화
- [01. 모델 최적화](19-deployment/01-model-optimization.md) — 양자화, 프루닝, 지식 증류
- [02. ONNX와 TensorRT](19-deployment/02-onnx-tensorrt.md) — 추론 가속화
- [03. 엣지 배포](19-deployment/03-edge-deployment.md) — Jetson, 라즈베리파이, 모바일
- [04. CV MLOps](19-deployment/04-mlops.md) — 학습 파이프라인과 모니터링
- [05. 모델 서빙](19-deployment/05-serving.md) — Triton, TorchServe, FastAPI

---

## 부록

### Resources
- [필수 논문 목록](resources/essential-papers.md) — 꼭 읽어야 할 핵심 논문들
- [주요 데이터셋](resources/datasets.md) — 학습과 벤치마크용 데이터셋
- [개발 도구](resources/tools.md) — 유용한 라이브러리와 프레임워크

---

## 학습 로드맵

```
입문자                  중급자                  고급자
  │                       │                       │
  ▼                       ▼                       ▼
Part 1 ──────────► Part 2 ──────────► Part 3
(기초)              (핵심)              (고급)
Ch1-4               Ch5-9               Ch10-14
  │                   │                   │
  │                   │                   ▼
  │                   │               Part 4
  │                   │             (전문가)
  │                   │              Ch15-18
  │                   │                   │
  └───────────────────┴───────────────────┘
                      │
                      ▼
                   Part 5
                   (실무)
                    Ch19
```

## 기술 스택

- **프레임워크**: PyTorch, OpenCV
- **언어**: Python 3.9+
- **라이브러리**: torchvision, Hugging Face Transformers, Diffusers
- **배포**: ONNX, TensorRT, TorchServe, Triton

## 기여하기

오타, 오류, 개선 사항은 Issue나 PR로 알려주세요.

## 라이선스

MIT License
