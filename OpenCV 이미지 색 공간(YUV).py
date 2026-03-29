import cv2

image = cv2.imread('Lenna.png')

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_image)

    print("Y Component:")
    print(y)
    print("\nU Component:")
    print(u)
    print("\nV Component:")
    print(v)

    cv2.imshow('Original Image', image)
    cv2.imshow('Y Component', y)
    cv2.imshow('U Component', u)
    cv2.imshow('V Component', v)

    cv2.waitKey(0)
    cv2.destroyAllWindows()