import cv2

image = cv2.imread('Lenna.png')

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    blue, green, red = cv2.split(image)

    print("Blue Component:")
    print(blue)
    print("\nGreen Component:")
    print(green)
    print("\nRed Component:")
    print(red)

    cv2.imshow('Original Image', image)
    cv2.imshow('Blue Component', blue)
    cv2.imshow('Green Component', green)
    cv2.imshow('Red Component', red)

    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()  # 모든 창 닫기
