import cv2

image = cv2.imread('Lenna.png')

if image is None:
    print("이미지를 읽을 수 없습니다. 경로를 확인하세요.")
else:
    cv2.imshow('Lenna', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    import cv2

    # 흑백(그레이스케일) 이미지 읽기
image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)


if image is None:
    print("이미지를 읽을 수 없습니다. 경로를 확인하세요.")
else:

    cv2.imshow('Lenna Gray', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()