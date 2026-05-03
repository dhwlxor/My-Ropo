# 🎥 Swin Transformer Video Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Swin-T](https://img.shields.io/badge/Model-Swin%20Transformer-lightgrey.svg)

본 프로젝트는 최신 비전 AI 모델인 **Swin Transformer**를 활용하여 비디오 영상 내의 주요 객체를 실시간으로 분석하고 분류(Classification)하는 파이썬 기반의 관제 어플리케이션입니다. (캡스톤 프로젝트 제출용)

## 주요 기능

* **Swin-T 기반 이미지 분류:** `timm` 라이브러리의 `swin_tiny_patch4_window7_224` 사전 학습 모델을 사용하여 영상 프레임 속 객체를 빠르고 정확하게 분석합니다.
* **Top-5 예측 대시보드:** 단순히 1위 결과만 보여주는 것이 아니라, 모델이 예측한 상위 5개의 객체 후보와 그 확률(%)을 직관적인 UI로 제공합니다.
* **CPU 환경 최적화:** 외장 GPU가 없는 랩탑 환경에서도 원활하게 구동되도록, 5프레임당 1번씩 AI 추론을 수행하는 최적화 로직이 적용되어 있습니다.
* **자동 라벨링 시스템:** 실행 시 인터넷에서 ImageNet의 1,000개 클래스 라벨 데이터를 자동으로 다운로드하여, 예측된 인덱스 번호를 실제 영문 사물 이름으로 변환합니다.
* **실시간 FPS 모니터링:** 시스템의 실시간 처리 속도를 나타내는 FPS(초당 프레임 수)를 실시간으로 계산하여 화면 상단에 표기합니다.

## 요구 환경

이 코드를 실행하기 위해서는 가상환경에 아래의 파이썬 패키지들이 설치되어 있어야 합니다.

```bash
pip install torch torchvision torchaudio
pip install timm opencv-python Pillow
```

## 실행 방법

* 저장소 다운로드 및 환경 준비 본 깃허브 저장소의 코드를 로컬 PC에 다운로드(Clone)하고 파이참(PyCharm) 등 IDE에서 엽니다.
* 필수 라이브러리 설치 : 프로그램 구동에 필요한 패키지들을 터미널에 입력하여 설치합니다.
* 테스트 비디오 경로 설정 : swin_video_analyzer.py 코드 내부의 video_path 변수를 본인 PC에 있는 테스트용 비디오 파일(.mp4, .avi)의 절대 경로로 수정합니다.
* 실행 : 터미널에서 아래 명령어를 실행하여 프로그램을 구동합니다. (최초 실행 시 1,000개 라벨 데이터와 모델 가중치를 다운로드하므로 1~2분 정도 소요될 수 있습니다.)
* 종료 : 분석 중인 영상 창이 활성화된 상태에서 키보드의 영문 q 키를 누르면 안전하게 종료됩니다.

## 한계점

* Classification vs Detection: 본 코드는 객체의 위치를 사각형으로 찾아내는 객체 탐지(Object Detection)가 아닌, 화면 전체의 맥락을 보고 어떤 사물이 주로 있는지 판단하는 이미지 분류(Image Classification) 방식입니다.
* Tiny 모델 채택 이유: 학술적인 Base/Large 모델 대신 파라미터가 적은 Tiny 모델을 채택하여, 엣지 디바이스 및 노트북(CPU) 환경에서도 실시간에 가까운 분석 속도를 보장할 수 있도록 설계했습니다.
* 초기 실행 시 모델 가중치(Weights)와 라벨 JSON 파일을 다운로드하는 데 시간이 다소 소요될 수 있습니다.
