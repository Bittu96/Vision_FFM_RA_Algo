import numpy as np
from numpy import sin, cos,exp

import pylab as plt
import skfmm

import math
from math import sqrt,pi

import matplotlib.pyplot as plt
import cv2

import time
from scipy.interpolate import spline



start_time = time.time()

img = cv2.imread('arena_map1.png')
#img = cv2.imread('map4.bmp')


rows,cols,chamels = img.shape
roi = img[0:rows,0:cols]
 
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#print img2gray
ret, mask = cv2.threshold(img2gray,254,255,cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

img_fg = cv2.bitwise_and(img,img,mask= mask)
img_bg = cv2.bitwise_and(roi,roi,mask= mask_inv)

dst = cv2.add(img_bg,img_fg)
img[0:rows,0:cols] = dst

#cv2.imshow('res',img)
#cv2.imshow('mask',mask)

X, Y = np.meshgrid(np.linspace(0,499,500), np.linspace(0,499,500))
phi = -1*np.ones_like(X)




#_----------------------------

Tx = 360
Ty = 320

ix = xpos = 202
iy = ypos = 64

#____________________________-----------




phi = (X-Tx)**2 + (Y-Ty)**2

speed = np.ones_like(X)
speed[abs(Y)>0] = 1

blur = cv2.GaussianBlur(mask, (499, 499), 5)
#cv2.imshow('blur',blur)

phi  = np.ma.MaskedArray(phi, blur)


t    = skfmm.travel_time(phi, speed, dx=1e-2)
plt.title('Travel time from the target with boundary conditions')
plt.contour(X, Y, phi, [0], linewidths=(3), colors='black')
#plt.contour(X, Y, phi.mask, [0], linewidths=(3), colors='blue')
#plt.contour(X, Y, blur, [0], linewidths=(3), colors='red')
#plt.contour(X, Y, t, 90)
#plt.colorbar()
plt.imshow(img)

def dist(x,y):
	distance = t[x][y]
	return distance 


m = 0
n = 0

distanc = sqrt((xpos-Tx)**2 + (ypos-Ty)**2)
position = (xpos,ypos)
plt.plot(xpos,ypos,'bs')
plt.plot(Tx,Ty,'rs')

vx = ix
vy = iy

vmx = xpos
vmy = ypos



l = 10
d = 1000


while distanc > 20:
    value = []
    
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
    print("--- %s seconds ---" % (time.time() - start_time)) 

print("--- %s seconds ---" % (time.time() - start_time))		


cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()

