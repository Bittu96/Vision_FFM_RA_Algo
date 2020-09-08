import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test2.png')

#IMREAD_GRAY SCALE = 0
#IMREAD_COLOR = 1
#IMREAD_UNCHANGED = -1

#add = img1 + img2
#add = cv2.add(img1,img2)
#cv2.imshow('add',add)

rows,cols,chamels = img.shape
roi = img[0:rows,0:cols]
 
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print img2gray
ret, mask = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

img_fg = cv2.bitwise_and(img,img,mask= mask)
img_bg = cv2.bitwise_and(roi,roi,mask= mask_inv)


dst = cv2.add(img_bg,img_fg)
img[0:rows,0:cols] = dst

cv2.imshow('res',img)
cv2.imshow('mask',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
