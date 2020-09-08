import numpy as np
import pylab as plt
import skfmm
import math
from math import sqrt,pi,atan2
from numpy import sin, cos,exp
import cv2
import time
import matplotlib.pyplot as plt

Tx = 50
Ty = 100

xi = xpos = 420
yi = ypos = 470
path = []
reg  = [(xi,yi)]
rect = []
FMM=0
FMMRA=0
distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
cap = cv2.VideoCapture('3obs.mp4')
while distanc > 1:
	
	m = 0
	n = 0

	def dist(x,y):
		distance = t[x][y]
		return distance 

	l = 5
	d = 10
	angle = 0
	roborient  = 0
	distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)

	a=0
	while a<5:
			FMMRA=0
			FMM=0
			ret, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#cv2.imshow('gray',gray)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			lower = np.array([0])
			upper = np.array([200])
			
			mask = cv2.inRange(gray, lower, upper)

			#cv2.imshow('gray',mask)
			
			img_fg = cv2.bitwise_and(frame,frame,mask= mask)
	
			X, Y = np.meshgrid(np.linspace(0,499,500), np.linspace(0,499,500))

			phi = 1*np.ones_like(X)
			
			phi = ((X-Tx)**2 + (Y-Ty)**2)-1
			
			blur = cv2.GaussianBlur(mask, (499, 499), 15)
			
			cv2.imshow('blur',blur)
			
			speed = np.ones_like(X)
			speed[abs(Y)>0] = 1

			phi  = np.ma.MaskedArray(phi, blur)

			t    = skfmm.travel_time(phi, speed, dx=1e-2)
			'''

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
			'''
			value = []
			index = []
			position = (xpos, ypos)

			for M in range(-6, 7, 6):
				for N in range(-6, 7, 6):
					m = xpos + M
					n = ypos + N
					P = dist(n, m)
					if P > 0:
						value.append(P)
					else:
						value.append(10)
						reg.append((xpos,ypos))
					# plt.plot(m,n,'ro')
					index.append((M, N))

			N = np.argmin(value)
			# print 'distance:',distance
			xindex = N / 6
			yindex = N % 6

			m = xpos
			n = ypos

			xpos = xpos + index[N][0]
			ypos = ypos + index[N][1]
			N = np.argmin(value)
			
			image,contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			contourss = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
		
			path.append((int(m),int(n)))
			cv2.circle(frame,(xi,yi),5,(255,0,0),2)
			cv2.circle(frame,(Tx,Ty),5,(0,0,255),2)
			
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
				cv2.line(frame,path[i],path[i+1],(0,150,0),2)
				FMM+=sqrt((path[i+1][1]-path[i][1])**2 + (path[i+1][0]-path[i][0])**2 )
				
				if i%5 == 0:
					cv2.rectangle(frame,rect[i][0],rect[i][1],(0,0,0),1)
					#cv2.rectangle(frame,rect[i][2],rect[i][3],(0,0,0),1)
					#cv2.rectangle(frame,rect[i][4],rect[i][5],(0,0,0),1)
					#cv2.rectangle(frame,rect[i][6],rect[i][7],(0,0,0),1)
					#cv2.rectangle(frame,rect[i][8],rect[i][9],(0,0,0),1)
					
					
		
			for i in range(0,len(reg)-1):
				cv2.line(frame,reg[i],reg[i+1],(0,0,200),2)
				FMMRA+=sqrt((reg[i+1][1]-reg[i][1])**2 + (reg[i+1][0]-reg[i][0])**2 )
			if distanc<10:
				cv2.line(frame,reg[len(reg)-1],(Tx,Ty),(0,0,200),2)
				FMMRA+=sqrt((reg[len(reg)-1][1]-Ty)**2 + (reg[len(reg)-1][0]-Tx)**2 )
			#print('--------------------GENERATING PATH----------------------')
			print('FMM___path_length  :  ',FMM)
			print('\n\nFMMRA_path_length  :  ',FMMRA)
			cv2.imshow('frame',frame)
			a+=1
			
	
	distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
	
