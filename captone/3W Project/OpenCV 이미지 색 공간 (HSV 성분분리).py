import cv2

image = cv2.imread('Lenna.png')

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    print("Hue Component:")
    print(h)
    print("\nSaturation Component:")
    print(s)
    print("\nValue Component:")
    print(v)

    cv2.imshow('Original Image', image)
    cv2.imshow('Hue Component', h)
    cv2.imshow('Saturation Component', s)
    cv2.imshow('Value Component', v)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
