import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image
import urllib.request
import json
import time  # 추가: FPS 계산용


def get_imagenet_labels():
    """인터넷에서 ImageNet 1,000개 라벨 리스트를 가져오는 함수"""
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        print("⏳ 라벨 리스트를 다운로드 중...")
        response = urllib.request.urlopen(url)
        labels = json.loads(response.read().decode())
        print(f"✅ 라벨 로드 완료! (총 {len(labels)}개)")
        return labels
    except Exception as e:
        print(f"⚠️ 라벨 로드 실패: {e}\n숫자로만 표시됩니다.")
        return None


def run_swin_advanced():
    # 1. 경로 설정 및 파일 로드
    video_path = r"C:\Users\dhwlxor\Desktop\AVI\Rec_20260429_073805_D.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: 파일을 열 수 없습니다. 경로를 확인하세요: {video_path}")
        return

    # 2. 라벨 리스트 가져오기
    labels = get_imagenet_labels()

    # 3. Swin Transformer 모델 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 {device} 환경에서 모델을 실행합니다.")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    model = model.to(device)
    model.eval()

    # 4. 전처리(Preprocessing) 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("🎥 고급 비디오 분석을 시작합니다... (Q를 누르면 종료)")

    # 상태 저장을 위한 변수 초기화
    prev_time = 0
    frame_count = 0
    top5_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        curr_time = time.time()

        # [기능 1] FPS (초당 프레임 수) 계산
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # [최적화] CPU 부하를 줄이기 위해 5프레임당 1번만 모델 추론 수행
        if frame_count % 5 == 0 or frame_count == 1:
            # OpenCV 프레임(BGR)을 PIL 이미지(RGB)로 변환
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            input_tensor = transform(img_pil).unsqueeze(0).to(device)

            # 5. 추론 (Inference)
            with torch.no_grad():
                output = model(input_tensor)

                # [기능 2] Top-5 확률 계산
                probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Softmax로 0~1 사이 확률 변환
                top5_prob, top5_catid = torch.topk(probabilities, 5)  # 상위 5개 추출

                # 결과를 리스트에 저장
                top5_results = []
                for i in range(5):
                    idx = top5_catid[i].item()
                    prob = top5_prob[i].item() * 100  # 퍼센트로 변환
                    name = labels[idx] if labels and idx < len(labels) else f"ID {idx}"
                    top5_results.append(f"{name}: {prob:.1f}%")

        # 6. 화면 UI 그리기 (분석기 대시보드 스타일)
        # 상단 배경 박스 그리기 (반투명 느낌을 내기 위해 검은색 사각형 삽입)
        cv2.rectangle(frame, (10, 10), (400, 220), (0, 0, 0), -1)

        # 타이틀 및 FPS 출력
        cv2.putText(frame, f"Swin-T Analyzer | FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(frame, (20, 50), (390, 50), (255, 255, 255), 1)  # 구분선

        # Top-5 리스트 출력
        for i, res in enumerate(top5_results):
            # 1위(TOP 1)는 눈에 띄게 녹색, 나머지는 흰색으로 표시
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            cv2.putText(frame, f"Top {i + 1}: {res}", (20, 85 + (i * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # 7. 결과 화면 출력
        cv2.imshow('Swin Transformer Video Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_swin_advanced()