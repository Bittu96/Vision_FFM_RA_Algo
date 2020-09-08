import numpy as np
import skfmm
import math
from math import sqrt,pi
from numpy import sin, cos,exp
import cv2
import time

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('obstacle.mp4')

import pygame

width = 600
height= 440

blue = (0,0,255)


pygame.init()

gameDisplay = pygame.display.set_mode((width,height))


def drawBox(gameDisplay):
	pygame.draw.line(gameDisplay,blue,(100,100),(100,width))
	pygame.draw.line(gameDisplay,blue,(100,width),(width,width))
	pygame.draw.line(gameDisplay,blue,(width,width),(width,100))
	pygame.draw.line(gameDisplay,blue,(width,100),(100,100))

Tx = 20
Ty = 20

xpos = 300
ypos = 450


m = 0
n = 0

global distance
diatance = 0
def dist(x,y):
	distance = t[x][y]
	return distance 


def bot(xy):
	pygame.draw.rect(display,blue,xy)


l = 10
d = 1000

def simulation():
    Tx = 20
    Ty = 20

    xpos = 300
    ypos = 450

    m = 0
    n = 0

    distance = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
	    
    while distance > 10:
        ret, frame = cap.read()
        
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #cv2.imshow('frame',frame)
        #cv2.imshow('gray',hsv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        lower_blue = np.array([150, 150, 50])
        upper_blue = np.array([180, 255, 200])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        
        #cv2.imshow('gray',mask)
        #ret, mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
        #blur = cv2.G#aussianBlur(mask, (499, 499), 10)
        #cv2.imshow('blur',blur)
        img_fg = cv2.bitwise_and(frame,frame,mask= mask)
                
        X, Y = np.meshgrid(np.linspace(0,499,500), np.linspace(0,499,500))
        #X, Y = np.meshgrid(np.linspace(0,639,640), np.linspace(0,479,480))
        '''
        phi = -1*np.ones_like(X)

        phi = ((X-Tx)**2 + (Y-Ty)**2)-1
        blur = cv2.GaussianBlur(mask, (499, 499), 5)

        speed = np.ones_like(X)
        speed[abs(Y)>0] = 1

        phi  = np.ma.MaskedArray(phi, blur)

        t    = skfmm.travel_time(phi, speed, dx=1e-2)
        print t	
                
        value = []
        index = []
        position = (xpos,ypos)
                
        for M in range(-6,7,6):
            for N in range (-6,7,6):
            m = xpos + M
            n = ypos + N
            P = dist(n,m)
            if P > 0:
                value.append(P)
            else:
                value.append(10)
            #plt.plot(m,n,'ro')
            index.append((M,N))
                            
        print value
        print index
        print index[1]
                
        print('position :',position)
                
        #plt.plot(xpos,ypos,'go')
        N = np.argmin(value)
        print N
        hero((xpos,ypos,10,10))
            
        print 'distance:',distance
        xindex = N/6
        yindex = N%6
                
        m = xpos
        n = ypos
                
        xpos = xpos + index[N][0]
        ypos = ypos + index[N][1]
            '''    
        distance = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
        print xindex,yindex
        display.fill((0,0,0))
        pygame.display.update()
            

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()


if __name__ == "__main__":
	simulation()

