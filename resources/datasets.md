# 주요 데이터셋 가이드

> 컴퓨터 비전 학습과 연구를 위한 데이터셋 모음

## 이미지 분류

| 데이터셋 | 규모 | 용도 | 링크 |
|---------|------|------|------|
| MNIST | 70K 이미지 | 손글씨 숫자 분류 입문 | [Link](http://yann.lecun.com/exdb/mnist/) |
| CIFAR-10/100 | 60K 이미지 | 소규모 이미지 분류 | [Link](https://www.cs.toronto.edu/~kriz/cifar.html) |
| ImageNet | 14M+ 이미지 | 대규모 분류 벤치마크 | [Link](https://www.image-net.org/) |
| ImageNet-1K | 1.2M 이미지 | 1000 클래스 분류 | [Link](https://www.image-net.org/) |

## 객체 탐지

| 데이터셋 | 규모 | 용도 | 링크 |
|---------|------|------|------|
| PASCAL VOC | 11K 이미지 | 20 클래스 탐지 | [Link](http://host.robots.ox.ac.uk/pascal/VOC/) |
| MS COCO | 330K 이미지 | 80 클래스 탐지/분할 | [Link](https://cocodataset.org/) |
| Open Images | 9M 이미지 | 대규모 탐지 | [Link](https://storage.googleapis.com/openimages/web/index.html) |

## 세그멘테이션

| 데이터셋 | 규모 | 용도 | 링크 |
|---------|------|------|------|
| ADE20K | 25K 이미지 | 시맨틱 세그멘테이션 | [Link](https://groups.csail.mit.edu/vision/datasets/ADE20K/) |
| Cityscapes | 25K 이미지 | 자율주행 세그멘테이션 | [Link](https://www.cityscapes-dataset.com/) |
| COCO-Stuff | 164K 이미지 | 물체+배경 분할 | [Link](https://github.com/nightrome/cocostuff) |

## 얼굴/사람

| 데이터셋 | 규모 | 용도 | 링크 |
|---------|------|------|------|
| CelebA | 200K 이미지 | 얼굴 속성 분석 | [Link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| FFHQ | 70K 이미지 | 고품질 얼굴 생성 | [Link](https://github.com/NVlabs/ffhq-dataset) |
| COCO-Pose | 250K 이미지 | 사람 포즈 추정 | [Link](https://cocodataset.org/) |

## 3D Vision

| 데이터셋 | 규모 | 용도 | 링크 |
|---------|------|------|------|
| ShapeNet | 51K 3D 모델 | 3D 객체 분류/생성 | [Link](https://shapenet.org/) |
| ScanNet | 1.5K 스캔 | 실내 3D 장면 이해 | [Link](http://www.scan-net.org/) |
| KITTI | 다양함 | 자율주행/깊이추정 | [Link](https://www.cvlibs.net/datasets/kitti/) |
| NYU Depth V2 | 1.4K 이미지 | 실내 깊이 추정 | [Link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) |

## 비디오

| 데이터셋 | 규모 | 용도 | 링크 |
|---------|------|------|------|
| Kinetics-400/600/700 | 650K 비디오 | 행동 인식 | [Link](https://deepmind.com/research/open-source/kinetics) |
| UCF101 | 13K 비디오 | 행동 분류 입문 | [Link](https://www.crcv.ucf.edu/data/UCF101.php) |
| YouTube-VOS | 4.5K 비디오 | 비디오 객체 분할 | [Link](https://youtube-vos.org/) |

## 텍스트-이미지

| 데이터셋 | 규모 | 용도 | 링크 |
|---------|------|------|------|
| LAION-5B | 5B 이미지-텍스트 쌍 | 대규모 사전학습 | [Link](https://laion.ai/blog/laion-5b/) |
| CC3M/CC12M | 3M/12M 쌍 | 이미지 캡셔닝 | [Link](https://github.com/google-research-datasets/conceptual-captions) |
| COCO Captions | 330K 이미지 | 이미지 캡셔닝 벤치마크 | [Link](https://cocodataset.org/) |

## 데이터셋 활용 팁

1. **입문**: MNIST → CIFAR-10 → ImageNet-1K 순으로 진행
2. **탐지/분할**: PASCAL VOC로 시작 → COCO로 확장
3. **생성 모델**: CelebA, FFHQ로 얼굴 생성 학습
4. **대규모 학습**: LAION 데이터셋 활용 (필터링 주의)
