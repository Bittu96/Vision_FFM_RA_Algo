import numpy as np
import skfmm
import math
from math import sqrt,pi
from numpy import sin, cos,exp
import cv2
import time

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture('obstacle.mp4')

Tx = 200
Ty = 200

xpos = 400
ypos = 400

m = 0
n = 0

distanc = sqrt((xpos-320)**2 + (ypos-240)**2)
position = (xpos,ypos)

def dist(x,y):
	distance = t[x][y]
	return distance 

l = 10
d = 100
path = [(xpos,ypos)]
while distanc > 20:
		ret, frame = cap.read()
		
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		lower = np.array([0])
		upper = np.array([150])
		
		#track red color
		lower_red = np.array([0, 30, 50])
		upper_red = np.array([20, 200, 255])
		
		mask = cv2.inRange(hsv, lower_red, upper_red)

		#mask = cv2.inRange(gray, lower, upper)
		#ret, mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
		img_fg = cv2.bitwise_and(frame,frame,mask= mask)
		
		#X, Y = np.meshgrid(np.linspace(0,499,500), np.linspace(0,499,500))
		X, Y = np.meshgrid(np.linspace(0,639,640), np.linspace(0,479,480))
		phi = -1*np.ones_like(X)
		phi = ((X-Tx)**2 + (Y-Ty)**2)-1
		blur = cv2.GaussianBlur(mask, (499, 499), 5)
		#cv2.imshow('blur',blur)
		speed = np.ones_like(X)
		speed[abs(Y)>0] = 1
		phi  = np.ma.MaskedArray(phi, blur)
		t    = skfmm.travel_time(phi, speed, dx=1e-2)
		print t	
		
		value = []
		index = []
		position = (xpos,ypos)
		
		for n in range (0,d):
			m = xpos + l*cos(n*2*pi/d)
			n = ypos + l*sin(n*2*pi/d)
			P = dist(round(n,0),round(m,0))
			if P > 0:
				value.append(P)
			else:
				value.append(100)
		
		N = np.argmin(value)
		distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
		m = xpos + l*cos(N*2*pi/d)
		n = ypos + l*sin(N*2*pi/d)
		path.append((int(m),int(n)))
		for i in range(0,len(path)-1):
			cv2.line(frame,path[i] ,path[i+1] ,(0,0,0),1)
		xpos = m
		ypos = n
		
		cv2.imshow('fr',frame)
		
		

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
