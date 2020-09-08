import numpy as np
# import pylab as plt
import skfmm
import math
from math import sqrt, pi
from numpy import sin, cos, exp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from scipy.interpolate import spline

time_start = time.clock()
start_time = time.time()

img = cv2.imread('map3.bmp')

rows, cols, chamels = img.shape
roi = img[0:rows, 0:cols]

img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print img2gray
ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

img_fg = cv2.bitwise_and(img, img, mask=mask)
img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

dst = cv2.add(img_bg, img_fg)
img[0:rows, 0:cols] = dst

# cv2.imshow('res',img)
# cv2.imshow('mask',mask)

X, Y = np.meshgrid(np.linspace(0, 499, 500), np.linspace(0, 499, 500))
phi = -1 * np.ones_like(X)

# _----------------------------

Tx = 23
Ty = 438

ix = xpos = 314
iy = ypos = 16

# ____________________________-----------

plt.plot(xpos, ypos, 'bs')
plt.plot(Tx, Ty, 'rs')

phi = (X - Tx) ** 2 + (Y - Ty) ** 2

speed = np.ones_like(X)
speed[abs(Y) > 0] = 1

blur = cv2.GaussianBlur(mask, (499, 499), 3)
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
# obstacle mask
# mask = img2gray
phi = np.ma.MaskedArray(phi, blur)

t = skfmm.travel_time(phi, speed, dx=1e-2)
#plt.title('Travel time from the target with boundary conditions')
plt.contour(X, Y, phi, [0], linewidths=(1), colors='black')
plt.contour(X, Y, phi.mask, [0], linewidths=(1), colors='blue')

plt.contour(X, Y, t, 90)
plt.colorbar()
plt.imshow(img)


def dist(x, y):
    distance = t[x][y]
    return distance

m = 0
n = 0
distance = sqrt((xpos - Tx) ** 2 + (ypos - Ty) ** 2)
position = (xpos, ypos)
plt.plot(xpos, ypos, 'bs')

path = [[], []]
pathlength =0
while distance > 10:

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
            # plt.plot(m,n,'ro')
            index.append((M, N))

    path[0].append(m)
    path[1].append(n)
    print('position :', position)

    # plt.plot(xpos,ypos,'go')
    N = np.argmin(value)


    print
    'distance:', distance
    xindex = N / 6
    yindex = N % 6

    m = xpos
    n = ypos

    xpos = xpos + index[N][0]
    ypos = ypos + index[N][1]

    line, = plt.plot([xpos, m], [ypos, n], 'green', lw=2)
    distance = sqrt((xpos - Tx) ** 2 + (ypos - Ty) ** 2)
    print
    xindex, yindex
    pathlength+=sqrt((xpos - m) ** 2 + (ypos - n) ** 2)
'''
time_elapsed = (time.clock() - time_start)
print
'timeelapsed', time_elapsed
print("--- %s seconds ---" % (time.time() - start_time))

print
path[0], path[1]'''
print ('pathlength :',pathlength)
plt.title(pathlength)
plt.gca().invert_yaxis()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()
