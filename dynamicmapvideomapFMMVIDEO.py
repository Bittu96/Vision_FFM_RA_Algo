import skfmm
import math
from math import sqrt,pi
from numpy import sin, cos,exp
import numpy as np
import cv2

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('obstacle.mp4')

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('frame',frame)
	#cv2.imshow('gray',hsv)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	lower_blue = np.array([0])
	upper_blue = np.array([150])
	mask = cv2.inRange(gray, lower_blue, upper_blue)
	ret, mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
	blur = cv2.GaussianBlur(mask, (499, 499), 10)
	#cv2.imshow('blur',blur)
	img_fg = cv2.bitwise_and(frame,frame,mask= mask)
	
	X, Y = np.meshgrid(np.linspace(0,499,500), np.linspace(0,499,500))
	#X, Y = np.meshgrid(np.linspace(0,639,640), np.linspace(0,479,480))
	
	phi = -1*np.ones_like(X)
	phi = ((X-320)**2 + (Y-240)**2)-1
	speed = np.ones_like(X)
	speed[abs(Y)>0] = 1
	phi  = np.ma.MaskedArray(phi, mask)
	t    = skfmm.travel_time(phi, speed, dx=1e-2)
	#print t
	ret, mask = cv2.threshold(mask,200,255,cv2.THRESH_BINARY_INV)
	cv2.imshow('mask',mask)
	image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contourss = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
	img = cv2.drawContours(frame, contourss, -1, (0,255,0), 3)

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()


