import cv2
import numpy as np
import os
import csv

# =====================================================================q
# 검사기준
# =====================================================================
AVG_THRESHOLD = 5000           # 초점 평균 합격 커트라인
DEVIATION_TOLERANCE = 120000   # 구역 편차 합격 허용치
BASE_DUST_NOISE = 1000       # [추가됨] 기본 배경 노이즈 무시 (영점 조절용)
MAX_DUST_THRESHOLD = 100000   # 이물 허용 픽셀

img_path = r"C:\Users\dhwlxor\Desktop\철거사진"
img_list = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png'))]

if len(img_list) == 0:
    print(f"지정한 경로에 이미지 파일이 없습니다: {img_path}")
    exit()

rw, rh = 1280, 720
current_idx = 0

report_file = "sensor_board_status_report.csv"
if not os.path.exists(report_file):
    with open(report_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["파일명", "건강지수(%)", "초점_수준", "고착_이물(px)", "판정결과", "조치사항"])

# =====================================================================
# [1]Temporal Background & Morphology Engine (전체화면 고착 이물 추출)
# =====================================================================
print("[시스템 초기화] 센서보드 전체 화면(Full-Screen) 시공간 분석 중...")
sample_imgs = []

for f in img_list[:15]:
    p = os.path.join(img_path, f)
    arr = np.fromfile(p, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        # 원본 2560x1440 전체를 1280x720으로 줄여서 모두 연산에 포함 (일부 자르기 X)
        std_img = cv2.resize(img, (rw, rh))
        sample_imgs.append(std_img)

if len(sample_imgs) > 0:
    pure_background = np.median(sample_imgs, axis=0).astype(np.uint8)
    pure_gray = cv2.cvtColor(pure_background, cv2.COLOR_BGR2GRAY)

    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(pure_gray, cv2.MORPH_BLACKHAT, bh_kernel)
    _, thresh_dust = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)

    lens_dust_pixels = cv2.countNonZero(thresh_dust)
else:
    lens_dust_pixels = 0

print(f"초기화 완료 -> 전체 화면 내 고착 이물 픽셀: {lens_dust_pixels} px 추출됨")


# =====================================================================
# [2] Spatial Focus Engine (초점 해상력 검사)
# =====================================================================
def get_focus_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


window_name = "Industrial Vision CMS - Sensor Board Status"
cv2.namedWindow(window_name)

# =====================================================================
# 메인 실시간 관제 루프
# =====================================================================
while True:
    full_path = os.path.join(img_path, img_list[current_idx])
    img_array = np.fromfile(full_path, np.uint8)
    raw_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if raw_frame is None: break

    # 전체 화면 크기 조정
    video_area = cv2.resize(raw_frame, (rw, rh))
    display_img = video_area.copy()

    # 🌟 전체 화면을 정확히 4등분 (여백 없음)
    mw, mh = 0, 0
    sub_h, sub_w = rh // 2, rw // 2
    zone_scores = []

    for i in range(2):
        for j in range(2):
            roi = video_area[mh + i * sub_h: mh + (i + 1) * sub_h, mw + j * sub_w: mw + (j + 1) * sub_w]
            score = get_focus_metrics(roi)
            zone_scores.append(score)
            # 4등분 영역 테두리 그리기 (전체 화면이 4칸의 큰 창문처럼 나뉨)
            cv2.rectangle(display_img, (mw + j * sub_w, mh + i * sub_h), (mw + (j + 1) * sub_w, mh + (i + 1) * sub_h),
                          (0, 255, 0), 2)

    avg_score = np.mean(zone_scores)

    # -------------------------------------------------------------
    # 센서 보드 건강 지수(Health Index)
    # -------------------------------------------------------------
    # 1. 초점 건강도 (1500점 이상이면 무조건 100점)
    focus_health = min((avg_score / AVG_THRESHOLD) * 100, 100) if avg_score > 0 else 0

    # 2. 이물질 오염도 (기본 배경 노이즈 12만 픽셀 이하는 100점 만점 처리)
    if lens_dust_pixels <= BASE_DUST_NOISE:
        dust_health = 100
    else:
        dust_penalty = ((lens_dust_pixels - BASE_DUST_NOISE) / MAX_DUST_THRESHOLD) * 100
        dust_health = max(100 - dust_penalty, 0)

    # 3. 최종 건강 지수 계산
    sensor_health_index = (focus_health * 0.6) + (dust_health * 0.4)

    # 4. 판정 로직
    if sensor_health_index >= 80:
        action_plan = "NORMAL OPERATION"  # 정상 가동
        action_color = (0, 255, 0)
        status_txt = "PASS"
    elif sensor_health_index >= 50:
        action_plan = "SCHEDULE CLEANING"  # 청소 일정 예약
        action_color = (0, 255, 255)
        status_txt = "WARNING"
    else:
        action_plan = "IMMEDIATE REPLACE"  # 즉시 교체 요망
        action_color = (0, 0, 255)
        status_txt = "FAIL"

    # =====================================================================
    # [대시보드 UI 랜더링]
    # =====================================================================
    side = np.zeros((rh, 400, 3), dtype=np.uint8)

    # 1. 타이틀 영역
    cv2.putText(side, "[ Camera Diagnostic ]", (20, 35), 2, 0.9, (255, 255, 255), 1)
    cv2.putText(side, "System: Active", (260, 35), 1, 0.8, (0, 255, 0), 1)
    cv2.line(side, (15, 50), (385, 50), (100, 100, 100), 1)

    # 2. 헬스 인덱스 (건강 지수) 표시
    cv2.putText(side, "SENSOR BOARD STATUS", (20, 85), 1, 1.2, (200, 200, 200), 1)
    cv2.putText(side, f"{sensor_health_index:.1f} %", (20, 130), 2, 1.5, action_color, 2)

    # 배터리 스타일 게이지 바
    bar_w = 350
    cv2.rectangle(side, (20, 150), (20 + bar_w, 170), (40, 40, 40), -1)
    cv2.rectangle(side, (20, 150), (20 + int(bar_w * (sensor_health_index / 100)), 170), action_color, -1)
    cv2.rectangle(side, (20, 150), (20 + bar_w, 170), (200, 200, 200), 1)

    # 3. 진단 세부 데이터 표시
    cv2.putText(side, "[ DIAGNOSTIC DATA ]", (20, 220), 1, 1.0, (255, 255, 0), 1)
    cv2.putText(side, f"- Focus Level : {int(avg_score)}", (20, 250), 1, 1.0, (255, 255, 255), 1)
    cv2.putText(side, f"- Surface Dust: {lens_dust_pixels} px", (20, 275), 1, 1.0, (255, 255, 255), 1)

    names = ["LT", "RT", "LB", "RB"]
    for idx, name in enumerate(names):
        cv2.putText(side, f"  {name}: {zone_scores[idx]:.0f}", (20 + (idx % 2) * 120, 305 + (idx // 2) * 25), 1, 0.9,
                    (150, 150, 150), 1)

    # 4. 유지보수 알람 영역
    cv2.line(side, (15, 365), (385, 365), (100, 100, 100), 1)
    cv2.putText(side, "MAINTENANCE ACTION:", (20, 390), 1, 1.0, (200, 200, 200), 1)
    cv2.putText(side, action_plan, (20, 420), 2, 0.9, action_color, 2)

    # 최종 화면 병합 및 출력
    final = np.hstack((display_img, side))
    cv2.imshow(window_name, final)

    # 조작 단축키 (다음/이전/저장/종료)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_idx = (current_idx + 1) % len(img_list)
    elif key == ord('b'):
        current_idx = (current_idx - 1) % len(img_list)
    elif key == ord('w'):
        with open(report_file, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(
                [img_list[current_idx], f"{sensor_health_index:.1f}", int(avg_score), lens_dust_pixels, status_txt,
                 action_plan])
        print(f"✅ 리포트 저장 완료: {img_list[current_idx]}")

cv2.destroyAllWindows()
