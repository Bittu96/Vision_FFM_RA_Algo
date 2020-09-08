import numpy as np
import pylab as plt
import skfmm
import math
from math import sqrt,pi
from numpy import sin, cos,exp
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('arena_map1.png')
#img = cv2.imread('test2.png')


rows,cols,chamels = img.shape
roi = img[0:rows,0:cols]

img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#print img2gray
ret, mask = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

blur = cv2.GaussianBlur(mask, (499, 499), 3)
#cv2.imshow('blur',blur)
img_fg = cv2.bitwise_and(img,img,mask= mask)
img_bg = cv2.bitwise_and(roi,roi,mask= mask_inv)

dst = cv2.add(img_bg,img_fg)
img[0:rows,0:cols] = dst

#cv2.imshow('res',img)
#cv2.imshow('mask',mask)

X, Y = np.meshgrid(np.linspace(0,499,500), np.linspace(0,499,500))
phi = -1*np.ones_like(X)

phi = ((X-58)**2 + (Y-365)**2) 


speed = np.ones_like(X)
speed[abs(Y)>0] = 1
'''
ret, clearance = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('mask',mask)
#cv2.imshow('clearance',clearance)
for i in range(490):
	for j in range(490):
		if mask[i][j] > 0:
			for M in range(-10,11,1):
				for N in range (-10,11,1):
					clearance[i+M][j+N] = 255
'''
#obstacle mask
#mask = img2gray


phi  = np.ma.MaskedArray(phi, blur)
#print phi

t    = skfmm.travel_time(phi, speed, dx=1e-2)
print t	

plt.title('Travel time from the target with boundary conditions')
plt.contour(X, Y, phi, [0], linewidths=(3), colors='black')
#plt.contour(X, Y, mask, [0], linewidths=(3), colors='red')
plt.contour(X, Y, t, 200)
plt.colorbar()
plt.imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()
