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
'''
port = "COM3"
baud = 9600

ser = serial.Serial(port, baud, timeout=1)
if ser.isOpen():
    print(ser.name + ' is open...')

'''
cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture('obstacle.mp4')

Tx = 175
Ty = 100

xi = xpos = 490
yi = ypos = 390


m = 0
n = 0

position = (xpos,ypos)
plt.plot(xpos,ypos,'bs')

plt.plot(0,0,'rs')

def dist(x,y):
	distance = t[x][y]
	return distance 

l = 20
d = 100
th = 0
angle = 0
roborient  = 0
distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)

path = []
k =0

char = 'nothing'

while distanc > 20:
		
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		#cv2.imshow('frame',frame)
		#cv2.imshow('gray',hsv)
		cv2.imshow('gray',gray)
		
		lower_green = np.array([0,100,50])
		upper_green = np.array([10,255,255])

		mask = cv2.inRange(hsv, lower_green, upper_green)
		#cv2.imshow('gray',mask)
		
		#ret, mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
		#blur = cv2.G#aussianBlur(mask, (499, 499), 10)
		
		img_fg = cv2.bitwise_and(frame,frame,mask= mask)
		
		#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
		
		#X, Y = np.meshgrid(np.linspace(0,499,500), np.linspace(0,499,500))
		X, Y = np.meshgrid(np.linspace(0,639,640), np.linspace(0,479,480))
		
		#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

		phi = -1*np.ones_like(X)
		
		phi = ((X-Tx)**2 + (Y-Ty)**2)-1
		
		blur = cv2.GaussianBlur(mask, (499, 499), 35)
		
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
			P = dist(int(n),int(m))
			if P > 0:
				value.append(P)
			else:
				value.append(100)
		
		N = np.argmin(value)
		
		th = (N*2*pi/d)*180/pi - 180
		#print th
		
		distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
		
		m = xpos + 30*cos(N*2*pi/d)
		n = ypos + 30*sin(N*2*pi/d)

		xpos = m
		ypos = n
		
		#plt.title('Travel time from the target with boundary conditions')
		#plt.contour(X, Y, phi, [0], linewidths=(3), colors='black')
		#plt.contour(X, Y, mask, [0], linewidths=(3), colors='red')
		
		path.append((int(m),int(n)))
		print ('--------------------GENERATING PATH----------------------')

cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture(1)

xpos = xi
ypos = yi
robopos = []

distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
j = 0
c = 0
while distanc > 10:
		print ('/////////////////////////  STATUS      ////////////////////////////////////')
		ret, frame = cap.read()
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
		lower_blue = np.array([80,50,50])
		upper_blue = np.array([150,200,255])
		
		mask = cv2.inRange(hsv, lower_blue, upper_blue)
		
		image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contourss = sorted(contours, key = cv2.contourArea, reverse = True)[:2]
		img = cv2.drawContours(frame, contourss, -1, (0,255,0), 3)
		
		(x,y),radius = cv2.minEnclosingCircle(contourss[0])
		(x1,y1),radius1 = cv2.minEnclosingCircle(contourss[1])
		
		center = (int(x),int(y))
		center1 = (int(x1),int(y1))
		robohead = ( int((x+x1)/2),int((y+y1)/2) )
		
		radius = int(radius)
		radius1 = int(radius1)
		
		cv2.circle(img,(Tx,Ty),20,(0,0,255),2)
		#cv2.circle(img,(xpos,ypos),20,(0,0,255),2)
		
		cv2.circle(img,center,radius,(255,255,255),1)
		cv2.circle(img,center1,radius1,(255,255,255),1)
		cv2.circle(img,robohead,40,(255,255,255),3)
		
		cv2.line(img,center,center1,(255,255,255),1)
		cv2.line(img,path[0],(xpos,ypos),(0,255,255),3)
		
		for i in range(0,len(path)-2):
			cv2.line(img,path[i],path[i+1],(0,255,255),3)
		
		for i in range(0,len(robopos)-1):
			cv2.circle(img,robopos[i],4,(255,0,0),1)
	
		angle = (atan2(center1[1]-center[1],center[0]-center1[0]))*180/pi 
		
		if angle > 0:
			roborient = abs((angle)-90)
		if angle < 0:
			roborient = abs(270+(angle))
		print ('\nroboorient : '),roborient
			
		distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
		
		track = sqrt((robohead[0]-path[j+3][0])**2 + (robohead[1]-path[j+3][1])**2)
		cv2.line(img,robohead,path[j+3],(255,255,255),1)
		if track < 50  and j < len(path)-2 and distanc > 60 :
			j+=1
		
		pathorient = abs(-(atan2(path[j+3][1]-robohead[1],path[j+3][0]-robohead[0]))*180/pi) 
		
		print('\npathorient : '),pathorient
		print('\nroboposition : '),robohead
		print('\nMoving To Position'),j+2,'..............'
		print('\n\ndistance to position'),j+2,(':  '),track
		
		#----------------------------------------------------------------------control
		
		if roborient > pathorient+20:
			print( '\nspot command :: right')
		elif roborient < pathorient-20:
			print( '\nspot command :: left')
		else :
			print ('\nspot command :: straight')
		
		#print '\nrunning command--------------------------- ', c	,char
		print ('\nrunning command--------------------------- ') ,(c),(char)

		
		
		a = k%350
		
		if a == 0:
			c += 1
			if roborient > pathorient+20:
				print ('\n ----------------------------------------------right')
				#ser.write('R')
				char = 'right'
			elif roborient < pathorient-20:
				print ('\n ----------------------------------------------left')
				#ser.write('L')
				char = 'left'
			else :
				print ('n --------------------------------------------straight')
				#ser.write('S')
				char = 'straight' 
				k+=150
			robopos.append(robohead)
		k+=1
		
		#----------------------------------------------------------------------
		cv2.imshow('img',img)



cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
