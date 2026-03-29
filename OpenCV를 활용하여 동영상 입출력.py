import cv2

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"동영상 파일을 열 수 없습니다: {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("동영상 재생이 끝났습니다.")
        break

    cv2.imshow('Video Playback', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()