# 🎥 Swin Transformer Video Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Swin-T](https://img.shields.io/badge/Model-Swin%20Transformer-lightgrey.svg)

본 프로젝트는 최신 비전 AI 모델인 **Swin Transformer**를 활용하여 비디오 영상 내의 주요 객체를 실시간으로 분석하고 분류(Classification)하는 파이썬 기반의 관제 어플리케이션입니다. (캡스톤 프로젝트 제출용)

## 📌 주요 기능 (Features)

* **Swin-T 기반 이미지 분류:** `timm` 라이브러리의 `swin_tiny_patch4_window7_224` 사전 학습 모델을 사용하여 영상 프레임 속 객체를 빠르고 정확하게 분석합니다.
* **Top-5 예측 대시보드:** 단순히 1위 결과만 보여주는 것이 아니라, 모델이 예측한 상위 5개의 객체 후보와 그 확률(%)을 직관적인 UI로 제공합니다.
* **CPU 환경 최적화:** 외장 GPU가 없는 랩탑 환경에서도 원활하게 구동되도록, 5프레임당 1번씩 AI 추론을 수행하는 최적화 로직이 적용되어 있습니다.
* **자동 라벨링 시스템:** 실행 시 인터넷에서 ImageNet의 1,000개 클래스 라벨 데이터를 자동으로 다운로드하여, 예측된 인덱스 번호를 실제 영문 사물 이름으로 변환합니다.
* **실시간 FPS 모니터링:** 시스템의 실시간 처리 속도를 나타내는 FPS(초당 프레임 수)를 실시간으로 계산하여 화면 상단에 표기합니다.

## 🛠️ 요구 환경 (Prerequisites)

이 코드를 실행하기 위해서는 가상환경에 아래의 파이썬 패키지들이 설치되어 있어야 합니다.

```bash
pip install torch torchvision torchaudio
pip install timm opencv-python Pillow
