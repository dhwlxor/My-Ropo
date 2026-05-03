import cv2

image = cv2.imread('Hawkes.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.equalizeHist(image)


cv2.imshow('Equalized Image', dst)
cv2.waitKey(0)   # 키 입력 대기
cv2.destroyAllWindows()


import cv2

image = cv2.imread('Hawkes.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('Normalized Image', dst)
cv2.waitKey(0)   # 키 입력 대기
cv2.destroyAllWindows()
