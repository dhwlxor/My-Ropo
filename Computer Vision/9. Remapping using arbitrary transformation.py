import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Lena.png')

xmap = np.zeros((img.shape[0], img.shape[1]), np.float32)
ymap = np.zeros((img.shape[0], img.shape[1]), np.float32)

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        xmap[y, x] = x + 30 * math.cos(20 * x / img.shape[0])
        ymap[y, x] = y + 30 * math.sin(20 * y / img.shape[1])

remapped_img = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)

plt.figure(0)
plt.axis('off')
plt.imshow(remapped_img[:, :, [2, 1, 0]])
plt.show()
