import numpy as np
import cv2
import pygame
import skfmm
import math
from math import sqrt,pi
from numpy import sin, cos,exp

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

width = 540
height = 400
blue = (0, 0, 255)
red = (200, 200, 0)

pygame.init()

gameDisplay = pygame.display.set_mode((width, height))

Tx = 20
Ty = 20

xpos = 300
ypos = 450

m = 0
n = 0

p = 10
q = 10

X, Y = np.meshgrid(np.linspace(0,639,640), np.linspace(0,479,480))

phi = -1*np.ones_like(X)
phi = ((X-Tx)**2 + (Y-Ty)**2)-1       
speed = np.ones_like(X)
speed[abs(Y)>0] = 1
mask = np.logical_and(abs(X-p)<80, abs(Y-q)<10) 
phi  = np.ma.MaskedArray(phi, mask)
t    = skfmm.travel_time(phi, speed, dx=1e-2)


distanc = 1000

def dist(x,y):
	distance = t[x][y]
	return distance 

def drawBox(gameDisplay):
    pygame.draw.line(gameDisplay, red, (10, 10), (590, 10))
    pygame.draw.line(gameDisplay, red, (590, 10), (590, 390))
    pygame.draw.line(gameDisplay, red, (590, 390), (10, 390))
    pygame.draw.line(gameDisplay, red, (10, 10), (10, 390))

def heady(xy):
    pygame.draw.rect(gameDisplay, blue, xy)

def gameloop():
    gameRunning = True
    Tx = 20
    Ty = 20

    xpos = 300
    ypos = 450
    t    = skfmm.travel_time(phi, speed, dx=1e-2)


    m = 0
    n = 0
    p = 10
    q = 10
    
    while gameRunning:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame, quit()
                quit()
        gameDisplay.fill((0, 0, 0))
        drawBox(gameDisplay)
     
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
	    p = x
	    q = y
  
	heady((p,q,10,10))
	X, Y = np.meshgrid(np.linspace(0,639,640), np.linspace(0,479,480))
		
	phi = -1*np.ones_like(X)

        phi = ((X-Tx)**2 + (Y-Ty)**2)-1
       
        speed = np.ones_like(X)
        speed[abs(Y)>0] = 1
	mask = np.logical_and(abs(X-p)<80, abs(Y-q)<10) 
        phi  = np.ma.MaskedArray(phi, mask)
	t    = skfmm.travel_time(phi, speed, dx=1e-2)
	for M in range(-6,7,6):
	    for N in range (-6,7,6):
		m = xpos + M
		n = ypos + N
		P = dist(n,m)
		if P > 0:
		    value.append(P)
		else:
		    value.append(10)
		index.append((M,N))
					
	N = np.argmin(value)
	xindex = N/6
	yindex = N%6
		
	m = xpos
	n = ypos
		
	xpos = xpos + index[N][0]
	ypos = ypos + index[N][1]
	distance = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
        
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        pygame.display.update()

if __name__ == "__main__":
   gameloop()


