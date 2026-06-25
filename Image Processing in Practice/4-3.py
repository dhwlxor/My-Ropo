import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
canny=cv.Canny(gray,100,200)

contours,hierarchy=cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

lcontours=[]
for i in range(len(contours)):
    if contours[i].shape[0]>100:
        lcontours.append(contours[i])

cv.drawContours(img,lcontours,-1,(0,0,255),3)

cv.imshow('Original with contours',img)
cv.imshow('Canny',canny)

cv.waitKey()
cv.destroyAllWindows()

