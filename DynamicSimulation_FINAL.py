import numpy as np
import pylab as plt
import skfmm
from math import sqrt,pi
from numpy import sin, cos,exp
import cv2
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt

#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('arena1FINAL.mp4')

'''
#4obs
Tx = 203
Ty = 68
xi = xpos = 420
yi = ypos = 420

'''

'''
#3obs
Tx = 100
Ty = 100
xi = xpos = 393
yi = ypos = 393
'''

#1obs
Tx = 50
Ty = 50
xi = xpos = 400
yi = ypos = 400



m = 0
n = 0

distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)

def dist(x,y):
	distance = t[int(x)][int(y)]
	return distance 

l = 1
d = 100

path = [(xpos,ypos)]
rect = []

while distanc > 5:
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
		lower_blue = np.array([50,50,50])
		upper_blue = np.array([200,255,200])
		#mask = cv2.inRange(hsv, lower_blue, upper_blue)
		
		lower = np.array([0])
		upper = np.array([200])
		mask = cv2.inRange(gray, lower, upper)
		
		ret, mask1 = cv2.threshold(mask,200,255,cv2.THRESH_BINARY_INV)

		img_fg = cv2.bitwise_and(frame,frame,mask= mask)
		X, Y = np.meshgrid(np.linspace(0,499,500), np.linspace(0,499,500))
		#X, Y = np.meshgrid(np.linspace(0,639,640), np.linspace(0,479,480))
		phi = -1*np.ones_like(X)
		phi = ((X-Tx)**2 + (Y-Ty)**2)-1
		
		blur = cv2.GaussianBlur(mask, (499, 499), 7)
		cv2.imshow('blur',blur)
		
		speed = np.ones_like(X)
		speed[abs(Y)>0] = 1
		phi  = np.ma.MaskedArray(phi, blur)
		t    = skfmm.travel_time(phi, speed, dx=1e-2)
		#print t	
		
		box = []
		value = []
		index = []
		position = (xpos,ypos)
		
		'''
		for n in range (0,d):
			m = xpos + l*cos(n*2*pi/d)
			n = ypos + l*sin(n*2*pi/d)
			P = dist(round(n,0),round(m,0))
			if P > 0:
				value.append(P)
			else:
				value.append(100)
			#plt.plot( m, n,'ro')
		
		#plt.plot( xpos, ypos,'ro')
		
		N = np.argmin(value)
		#path_distance+=l
		#print path_distance
		distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
		m = xpos + l*cos(N*2*pi/d)
		n = ypos + l*sin(N*2*pi/d)
		line, = plt.plot([xpos,m], [ypos,n], 'black', lw=1)
		xpos = m
		ypos = n
		
		'''
		
		for M in range(-3,3,1):
			for N in range (-3,3,1):
					m = xpos + M
					n = ypos + N
					P = dist(n,m)
					if P > 0:
						value.append(P)
					else:
						value.append(10)
					#plt.plot(m,n,'ro')
					index.append((M,N))
					
		#print value
		#print index
		#print index[1]
		
		print('position :',position)
		
		N = np.argmin(value)
		#print( N)
		
		print( 'distance:',distanc)
		xindex = N/6
		yindex = N%6

		m = xpos
		n = ypos
		
		xpos = xpos + index[N][0]/2
		ypos = ypos + index[N][1]/2

		distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
		#print xindex,yindex
		
		path.append((int(xpos),int(ypos)))
		
		#cv2.circle(frame,(Tx,Ty),15,(0,0,255),2)
		#cv2.circle(frame,(xi,yi),7,(0,255,0),2)
		
		#cv2.circle(frame,(xpos,ypos),6,(255,0,0),1)
		#cv2.line(frame,(xpos,ypos),(m,n),(0,0,255),2)
		image,contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contourss = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
		#frame = cv2.drawContours(frame, contourss, -1, (0,255,0), 1)
		'''
		rect = cv2.minAreaRect(contourss)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(frame,[box],0,(0,0,255),2)
		'''
		
		#x4,y4,w4,h4 = cv2.boundingRect(contourss[6])
		#x3,y3,w3,h3 = cv2.boundingRect(contourss[5])
		#x2,y2,w2,h2 = cv2.boundingRect(contourss[4])
		#x1,y1,w1,h1 = cv2.boundingRect(contourss[3])
		x, y, w, h  = cv2.boundingRect(contourss[2])
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		rect.append(( (x,y),(x+w,y+h) ))
		#rect.append(( (x,y),(x+w,y+h),(x1,y1),(x1+w1,y1+h1) ))
		#rect.append(( (x,y),(x+w,y+h),(x1,y1),(x1+w1,y1+h1),(x2,y2),(x2+w2,y2+h2) ))
		#rect.append(( (x,y),(x+w,y+h),(x1,y1),(x1+w1,y1+h1),(x2,y2),(x2+w2,y2+h2),(x3,y3),(x3+w3,y3+h3) ))
		#rect.append(( (x,y),(x+w,y+h),(x1,y1),(x1+w1,y1+h1),(x2,y2),(x2+w2,y2+h2),(x3,y3),(x3+w3,y3+h3),(x4,y4),(x4+w4,y4+h4) ))

		for i in range(0,len(path)-1):
			cv2.line(frame,path[i],path[i+1],(0,150,0),3)
			if i%5 == 0:
				cv2.rectangle(frame,rect[i][0],rect[i][1],(0,0,0),1)
				#cv2.rectangle(frame,rect[i][2],rect[i][3],(150,200,150),1)
				#cv2.rectangle(frame,rect[i][4],rect[i][5],(200,150,150),1)
				#cv2.rectangle(frame,rect[i][6],rect[i][7],(150,150,200),1)
				#cv2.rectangle(frame,rect[i][8],rect[i][9],(150,150,200),1)
		cv2.imshow('frame',frame)
		

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
