import cv2
import numpy as np

image = cv2.imread('Candies.png')

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    b, g, r = cv2.split(image)
    red_mask = cv2.inRange(r, 200, 255)   # 원하는 임계값으로 변경 가능
    filtered_image = cv2.bitwise_and(image, image, mask=red_mask)

    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()