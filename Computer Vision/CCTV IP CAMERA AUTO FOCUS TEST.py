import cv2
import numpy as np
import os
import csv

# --- [공정 기준 설정] ---
AVG_THRESHOLD = 15000
DEVIATION_TOLERANCE = 10000
# -----------------------

img_path = r"C:\Users\dhwlxor\Desktop\새 폴더"
img_list = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png'))]
RED_RECT_ROI = (12, 38, 628, 428)

current_idx = 0
enhance_mode = False
algo_mode = '1'

report_file = "focus_report.csv"
if not os.path.exists(report_file):
    with open(report_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["파일명", "평균_선명도(Avg)", "구역_편차(Dev)", "좌상(LT)", "우상(RT)", "좌하(LB)", "우하(RB)", "판정결과"])


def get_focus_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


window_name = "IDIS Industrial Vision Project - QA Dashboard"
cv2.namedWindow(window_name)

while True:
    full_path = os.path.join(img_path, img_list[current_idx])
    img_array = np.fromfile(full_path, np.uint8)
    raw_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if raw_frame is None: break

    std_frame = cv2.resize(raw_frame, (1280, 720))
    rx, ry, rw, rh = RED_RECT_ROI
    video_area = std_frame[ry:ry + rh, rx:rx + rw]

    if enhance_mode:
        gaussian = cv2.GaussianBlur(video_area, (0, 0), 2)
        video_area = cv2.addWeighted(video_area, 1.5, gaussian, -0.5, 0)

    display_img = video_area.copy()
    gray_area = cv2.cvtColor(video_area, cv2.COLOR_BGR2GRAY)

    if algo_mode == '1':
        corners = cv2.goodFeaturesToTrack(gray_area, maxCorners=50, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            for i in np.intp(corners):
                cv2.circle(display_img, tuple(i.ravel()), 3, (0, 0, 255), -1)
    elif algo_mode == '2':
        display_img = cv2.cvtColor(cv2.Canny(gray_area, 50, 150), cv2.COLOR_GRAY2BGR)
    elif algo_mode == '3':
        _, thresh = cv2.threshold(gray_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        display_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif algo_mode == '4':
        kp = cv2.ORB_create().detect(video_area, None)
        display_img = cv2.drawKeypoints(video_area, kp, None, color=(0, 255, 0))

    mw, mh = int(rw * 0.1), int(rh * 0.1)
    sub_h, sub_w = (rh - 2 * mh) // 2, (rw - 2 * mw) // 2
    zone_scores = []

    for i in range(2):
        for j in range(2):
            roi = video_area[mh + i * sub_h: mh + (i + 1) * sub_h, mw + j * sub_w: mw + (j + 1) * sub_w]
            score = get_focus_metrics(roi)
            zone_scores.append(score)
            cv2.rectangle(display_img, (mw + j * sub_w, mh + i * sub_h), (mw + (j + 1) * sub_w, mh + (i + 1) * sub_h),
                          (0, 255, 0), 1)

    avg_score = np.mean(zone_scores)
    deviation = np.max(zone_scores) - np.min(zone_scores)

    # [핵심] 두 조건이 모두 만족해야 PASS
    is_pass = (avg_score >= AVG_THRESHOLD) and (deviation <= DEVIATION_TOLERANCE)

    # --- 사이드 패널 텍스트 ---
    side = np.zeros((rh, 400, 3), dtype=np.uint8)
    cv2.putText(side, "ENHANCE: " + ("ON" if enhance_mode else "OFF"), (20, 20), 1, 1.0, (255, 255, 0), 1)
    cv2.putText(side, "RESULT: " + ("PASS" if is_pass else "FAIL"), (20, 55), 2, 1.2,
                (0, 255, 0) if is_pass else (0, 0, 255), 2)
    cv2.putText(side, f"AVG Score: {avg_score:.1f}", (20, 85), 1, 1.0, (255, 255, 255), 1)
    cv2.putText(side, f"Deviation: {deviation:.1f}", (20, 110), 1, 1.0, (255, 255, 255), 1)

    names = ["LT", "RT", "LB", "RB"]
    for idx, name in enumerate(names):
        cv2.putText(side, f"{name}: {zone_scores[idx]:.1f}", (20, 145 + idx * 25), 1, 1.0, (200, 200, 200), 1)

    # --- [수정됨] 양불 판정 기준에 맞춘 시각화 (AVG & DEV 2개 막대) ---
    chart_x, chart_y, chart_w, chart_h = 20, 400, 360, 140
    # 그래프 배경 및 중앙선, X축
    cv2.rectangle(side, (chart_x, chart_y - chart_h), (chart_x + chart_w, chart_y), (40, 40, 40), -1)
    cv2.line(side, (chart_x + chart_w // 2, chart_y - chart_h), (chart_x + chart_w // 2, chart_y), (100, 100, 100), 1)
    cv2.line(side, (chart_x, chart_y), (chart_x + chart_w, chart_y), (255, 255, 255), 2)

    # 1. 왼쪽 차트 (AVG - 초점 점수)
    max_avg_val = max(avg_score * 1.2, AVG_THRESHOLD * 1.5) if avg_score > 0 else 1
    avg_h = int((avg_score / max_avg_val) * chart_h)
    avg_color = (0, 255, 0) if avg_score >= AVG_THRESHOLD else (0, 165, 255)  # Threshold 이상이면 녹색
    avg_bx = chart_x + 60
    cv2.rectangle(side, (avg_bx, chart_y - avg_h), (avg_bx + 50, chart_y), avg_color, -1)
    cv2.putText(side, "AVG", (avg_bx + 10, chart_y + 20), 1, 1.0, (255, 255, 255), 1)
    cv2.putText(side, f"{int(avg_score / 1000)}k", (avg_bx - 5, chart_y - avg_h - 5), 1, 0.9, (200, 200, 200), 1)

    avg_th_h = int((AVG_THRESHOLD / max_avg_val) * chart_h)
    cv2.line(side, (chart_x, chart_y - avg_th_h), (chart_x + chart_w // 2, chart_y - avg_th_h), (0, 0, 255), 2)
    cv2.putText(side, f"MIN {int(AVG_THRESHOLD / 1000)}k", (chart_x + 5, chart_y - avg_th_h - 5), 1, 0.8, (0, 0, 255),
                1)

    # 2. 오른쪽 차트 (DEV - 편차 점수)
    max_dev_val = max(deviation * 1.2, DEVIATION_TOLERANCE * 1.5) if deviation > 0 else 1
    dev_h = int((deviation / max_dev_val) * chart_h)
    dev_color = (0, 255, 0) if deviation <= DEVIATION_TOLERANCE else (0, 165, 255)  # Tolerance 이하면 녹색 (낮을수록 좋음)
    dev_bx = chart_x + chart_w // 2 + 60
    cv2.rectangle(side, (dev_bx, chart_y - dev_h), (dev_bx + 50, chart_y), dev_color, -1)
    cv2.putText(side, "DEV", (dev_bx + 10, chart_y + 20), 1, 1.0, (255, 255, 255), 1)
    cv2.putText(side, f"{int(deviation / 1000)}k", (dev_bx - 5, chart_y - dev_h - 5), 1, 0.9, (200, 200, 200), 1)

    dev_th_h = int((DEVIATION_TOLERANCE / max_dev_val) * chart_h)
    cv2.line(side, (chart_x + chart_w // 2, chart_y - dev_th_h), (chart_x + chart_w, chart_y - dev_th_h), (0, 0, 255),
             2)
    cv2.putText(side, f"MAX {int(DEVIATION_TOLERANCE / 1000)}k", (chart_x + chart_w // 2 + 5, chart_y - dev_th_h - 5),
                1, 0.8, (0, 0, 255), 1)
    # ----------------------------------------------------

    final = np.hstack((display_img, side))
    cv2.imshow(window_name, final)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_idx = (current_idx + 1) % len(img_list)
    elif key == ord('b'):
        current_idx = (current_idx - 1) % len(img_list)
    elif key == ord('s'):
        enhance_mode = not enhance_mode
    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        algo_mode = chr(key)
    elif key == ord('w'):
        status = "PASS" if is_pass else "FAIL"
        with open(report_file, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([img_list[current_idx], f"{avg_score:.2f}", f"{deviation:.2f}",
                             f"{zone_scores[0]:.2f}", f"{zone_scores[1]:.2f}",
                             f"{zone_scores[2]:.2f}", f"{zone_scores[3]:.2f}", status])
        print(f"✅ 저장 완료: {img_list[current_idx]} 데이터 저장됨")

cv2.destroyAllWindows()