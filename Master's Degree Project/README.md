# 📷 IDIS Focus Inspector - CCTV Automated Focus & Decentering Inspection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt-5-green.svg)](https://pypi.org/project/PyQt5/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)](https://github.com/ultralytics/ultralytics)

화각 고정형 CCTV IP 카메라의 양산 공정에서 초점(Focus) 선명도, 상면 편심(Lens Decentering/Tilt), 그리고 JIG 안착 상태를 정량적으로 자동 판정하는 Vision Inspection GUI 시스템입니다.

---

## 주요 특징 (Key Features)

- **하이브리드 초점 평가 알고리즘 (Sobel 70% + Laplacian 30%)**:
  - 1차 미분 기반 Tenengrad(Sobel)와 2차 미분 기반 Laplacian Variance를 결합한 복합 점수 산출
  - 배경 마스킹(Thresholding > 200)을 통한 차트 외곽 여백 자동 제외 및 순수 패턴 영역 가중 점수화
- **YOLOv8 기반 Real-time JIG 안착 감지**:
  - 실시간 프레임 내 차트 패턴 자동 인식 (`conf=0.12`) 및 JIG 안착 개수 검증
- **3x3 ROI Grid & 상면 편심(기울어짐) 분석**:
  - $1920 \times 1080$ 영상을 9개 구역으로 분할하여 영역별 초점 점수 및 Left-Right 편차율($Dev\%$) 계산
- **이중 매개변수 관리 구조 (Models & Presets)**:
  - **10종 모델 지원**: 카메라 제품별 초점 임계값(R, C, L, Dev%) 독립 저장 및 콤보박스 변경 즉시 로드
  - **3종 프리셋 지원**: 캡처 오프셋(X, Y), 크롭 비율, 해상도 등 기계적 화면 캡처 설정 통합 관리
- **AI 조도(Lux) 실시간 모니터링**: 암실 환경 타깃 조도 모니터링 및 조도 부족 시 JIG 경고 알람 출력
- **자동 데이터 데이터베이스 로깅**: 검사 수행 시 엑셀 결과 리포트 및 오버레이 JPG 이미지 자동 생성

---

## 광학계 및 알고리즘 사양

- **광학계 구성**: $f=500\text{mm}$ 이중 렌즈 콜리메이터 지그 (가상 거리 $Z_{virtual} \approx 125\text{m}$ 모사)
- **타깃 차트**: ISO 12233 & Siemens Star Chart
- **초점 평가 하이브리드 수식**:
  $$S_{focus} = 0.7 \times \overline{G_{sobel}} + 0.3 \times \text{Var}(L_{laplacian})$$

---

## 시작하기 (Getting Started)

### 1. 요구 사항 (Prerequisites)

- Python 3.8 이상
- Windows OS (Screen Capture 및 Window Handle 의존)

### 2. 필수 라이브러리 설치

```bash
pip install opencv-python numpy pandas openpyxl mss pygetwindow PyQt5 ultralytics
