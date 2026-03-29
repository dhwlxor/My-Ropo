import cv2
import matplotlib.pyplot as plt

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    cv2.imshow('Grayscale Image', image)

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()