import cv2
import numpy as np

img = cv2.imread("candies.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

red_candies = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("red_candies.png", red_candies)

import cv2
import numpy as np

img = cv2.imread("candies.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
blue_candies = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("blue_candies.png", blue_candies)