# 개발 도구 및 라이브러리

> 컴퓨터 비전 개발에 필요한 도구 모음

## 딥러닝 프레임워크

| 도구 | 용도 | 설치 | 문서 |
|------|------|------|------|
| PyTorch | 연구/프로덕션 | `pip install torch torchvision` | [Link](https://pytorch.org/) |
| TensorFlow | 프로덕션/모바일 | `pip install tensorflow` | [Link](https://tensorflow.org/) |
| JAX | 고성능 연구 | `pip install jax jaxlib` | [Link](https://github.com/google/jax) |

## 이미지 처리

| 도구 | 용도 | 설치 | 문서 |
|------|------|------|------|
| OpenCV | 전통적 CV | `pip install opencv-python` | [Link](https://opencv.org/) |
| Pillow | 이미지 I/O | `pip install pillow` | [Link](https://pillow.readthedocs.io/) |
| scikit-image | 이미지 처리 | `pip install scikit-image` | [Link](https://scikit-image.org/) |
| Albumentations | 데이터 증강 | `pip install albumentations` | [Link](https://albumentations.ai/) |

## 모델 라이브러리

| 도구 | 용도 | 설치 | 문서 |
|------|------|------|------|
| timm | 이미지 분류 모델 | `pip install timm` | [Link](https://github.com/huggingface/pytorch-image-models) |
| torchvision | PyTorch 공식 모델 | PyTorch와 함께 설치 | [Link](https://pytorch.org/vision/) |
| Hugging Face Transformers | ViT, CLIP 등 | `pip install transformers` | [Link](https://huggingface.co/transformers/) |
| Ultralytics | YOLO 시리즈 | `pip install ultralytics` | [Link](https://docs.ultralytics.com/) |
| MMDetection | 객체 탐지 | `pip install mmdet` | [Link](https://mmdetection.readthedocs.io/) |
| Detectron2 | Meta AI 탐지/분할 | [설치 가이드](https://detectron2.readthedocs.io/) | [Link](https://detectron2.readthedocs.io/) |

## 생성 모델

| 도구 | 용도 | 설치 | 문서 |
|------|------|------|------|
| Diffusers | Diffusion 모델 | `pip install diffusers` | [Link](https://huggingface.co/docs/diffusers/) |
| ComfyUI | 노드 기반 워크플로우 | [GitHub](https://github.com/comfyanonymous/ComfyUI) | [Link](https://github.com/comfyanonymous/ComfyUI) |
| AUTOMATIC1111 | WebUI | [GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui) | [Link](https://github.com/AUTOMATIC1111/stable-diffusion-webui) |

## 3D Vision

| 도구 | 용도 | 설치 | 문서 |
|------|------|------|------|
| Open3D | 포인트 클라우드 | `pip install open3d` | [Link](http://www.open3d.org/) |
| PyTorch3D | 3D 딥러닝 | [설치 가이드](https://pytorch3d.org/) | [Link](https://pytorch3d.org/) |
| Nerfstudio | NeRF 학습 | `pip install nerfstudio` | [Link](https://docs.nerf.studio/) |

## 모델 최적화 & 배포

| 도구 | 용도 | 설치 | 문서 |
|------|------|------|------|
| ONNX | 모델 변환 | `pip install onnx onnxruntime` | [Link](https://onnx.ai/) |
| TensorRT | NVIDIA 추론 가속 | [설치 가이드](https://developer.nvidia.com/tensorrt) | [Link](https://developer.nvidia.com/tensorrt) |
| OpenVINO | Intel 추론 최적화 | `pip install openvino` | [Link](https://docs.openvino.ai/) |
| TorchServe | PyTorch 서빙 | `pip install torchserve` | [Link](https://pytorch.org/serve/) |
| Triton | NVIDIA 모델 서빙 | [설치 가이드](https://developer.nvidia.com/triton-inference-server) | [Link](https://developer.nvidia.com/triton-inference-server) |

## 실험 관리

| 도구 | 용도 | 설치 | 문서 |
|------|------|------|------|
| Weights & Biases | 실험 추적 | `pip install wandb` | [Link](https://wandb.ai/) |
| MLflow | ML 라이프사이클 | `pip install mlflow` | [Link](https://mlflow.org/) |
| TensorBoard | 시각화 | `pip install tensorboard` | [Link](https://www.tensorflow.org/tensorboard) |

## 레이블링 도구

| 도구 | 용도 | 링크 |
|------|------|------|
| Label Studio | 범용 레이블링 | [Link](https://labelstud.io/) |
| CVAT | 비디오/이미지 어노테이션 | [Link](https://cvat.ai/) |
| Roboflow | 자동 레이블링 + 배포 | [Link](https://roboflow.com/) |

## 추천 개발 환경

```bash
# 기본 환경 설정
conda create -n cv python=3.10
conda activate cv

# 핵심 라이브러리
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow matplotlib numpy
pip install timm transformers diffusers
pip install albumentations
pip install wandb tensorboard

# 객체 탐지
pip install ultralytics

# Jupyter
pip install jupyterlab ipywidgets
```
