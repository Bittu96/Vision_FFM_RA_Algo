import numpy as np
import pylab as plt
import skfmm
import math
from math import sqrt,pi,atan2

from numpy import sin, cos,exp

import cv2
import time

import matplotlib.pyplot as plt
import serial

#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('obstacle.mp4')

Tx = 12
Ty = 130

xpos = 400
ypos = 400

m = 0
n = 0

distance = sqrt((xpos-320)**2 + (ypos-240)**2)

position = (xpos,ypos)
plt.plot(xpos,ypos,'bs')

plt.plot(0,0,'rs')

def dist(x,y):
	distance = t[x][y]
	return distance 

l = 20
d = 1000
th = 0



roborient  = 45

pathdistance = 0

while distance > 50:
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		#cv2.imshow('frame',frame)
		#cv2.imshow('gray',hsv)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		lower_blue = np.array([50])
		upper_blue = np.array([200])
		
		#lower_blue = np.array([50,50,50])
		#upper_blue = np.array([200,255,255])

		mask = cv2.inRange(gray, lower_blue, upper_blue)
		#mask = cv2.inRange(hsv, lower_blue, upper_blue)
		
		#cv2.imshow('gray',mask)
		
		#ret, mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
		#blur = cv2.G#aussianBlur(mask, (499, 499), 10)
		
		img_fg = cv2.bitwise_and(frame,frame,mask= mask)
		
		#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
		
		X, Y = np.meshgrid(np.linspace(0,499,500), np.linspace(0,499,500))
		#X, Y = np.meshgrid(np.linspace(0,639,640), np.linspace(0,479,480))
		
		#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
		phi = -1*np.ones_like(X)
		phi = ((X-Tx)**2 + (Y-Ty)**2)-1
		blur = cv2.GaussianBlur(mask, (499, 499), 10)
		
		cv2.imshow('blur',blur)
		
		speed = np.ones_like(X)
		speed[abs(Y)>0] = 1

		phi  = np.ma.MaskedArray(phi, blur)

		t    = skfmm.travel_time(phi, speed, dx=1e-2)
		#print t	
		
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
			#plt.plot( m, n,'ro')
		
		N = np.argmin(value)
		#path_distance+=l
		#print path_distance
		
		th = (N*2*pi/d)*180/pi - 180
		#print th
		
		distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
		
		m = xpos + 10*cos(N*2*pi/d)
		n = ypos + 10*sin(N*2*pi/d)

		line, = plt.plot([xpos,m], [ypos,n], 'green', lw=3)
		line, = plt.plot([xpos,Tx], [ypos,Ty], 'green', lw=1)
		
		circle =plt.Circle((xpos,ypos),l,color = 'black',fill=False)
		plt.plot(xpos,ypos,'ro')
		
		line, = plt.plot([xpos,xpos + l*cos(N*2*pi/d)], [ypos,ypos + l*sin(N*2*pi/d)], 'black', lw=1)
		
		xpos = m
		ypos = n
		
		pathorient = th
		#print 'pathorient',pathorient
		

		
		if roborient < pathorient-2 :
			for n in (0,abs(pathorient - roborient)):
				#ser.write('R')
				plt.imshow(frame)
				roborient+=2
				#print 'roborient :',roborient
				#circle =plt.Circle((100,100),20,color = 'green')
				plt.gcf().gca().add_artist(circle)
				plt.pause(0.00000000001)
			
		elif roborient > pathorient+2 :
			for n in (0,abs(roborient - pathorient+2) ):
				#ser.write('L')
				plt.imshow(frame)
				roborient-=2
				#print 'roborient :',roborient
				#circle =plt.Circle((100,100),20,color = 'green')
				plt.gcf().gca().add_artist(circle)
				plt.pause(0.00000000001)
			
		elif roborient > pathorient-2 or roborient < pathorient+2 :
			#ser.write('S')
			plt.imshow(frame)
			#print 'roborient :',roborient
			#circle =plt.Circle((100,100),20,color = 'green')
			plt.gcf().gca().add_artist(circle)
			plt.pause(0.00000000001)
		
		print 'diff   :',pathorient-roborient
	
		plt.title('Travel time from the target with boundary conditions')
		#plt.contour(X, Y, phi, [0], linewidths=(3), colors='black')
		#plt.contour(X, Y, mask, [0], linewidths=(3), colors='red')
		
		pathdistance+=10
		print 'pathdistance :',pathdistance
		plt.imshow(frame)

		#plt.pause(0.00000000001)
		plt.clf()
		plt.cla()

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

plt.show()

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''
		pathorient = atan2( (n-m) , (ypos-xpos) )
		if roborient < ((pathorient)*180/pi)-10:
			for n in (0,((pathorient)*180/pi)-10-roborient):
				roborient+=1
				#ser.write('L')
				print 'left'
				print roborient
				print pathorient
			
		if roborient > ((pathorient)*180/pi)+10:
			for n in (0,roborient-((pathorient)*180/pi)+10):
				roborient-=1
				#ser.write('R')
				print 'right'
				print roborient
			
		if roborient > ((pathorient)*180/pi)-10 or roborient < ((pathorient)*180/pi)+10:
			#ser.write('S')
			print 'straight'
			print roborient
'''   
