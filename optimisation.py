import numpy as np
import pylab as plt
import skfmm
import math
from math import sqrt, pi
from numpy import sin, cos, exp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from scipy.interpolate import spline

start_time = time.time()

# img = cv2.imread('test2.png')
img = cv2.imread('map2.bmp')
# img = cv2.imread('arena_map1.png')

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
phi = 1 * np.ones_like(X)
# print phi

# _----------------------------

Tx = 470
Ty = 440

ix = xpos = 10
iy = ypos = 10
'''
Tx = 161
Ty = 470

ix = xpos = 10
iy = ypos = 10
'''
# plt.text(Tx+5, Ty+5,'goal', fontsize=10,color = 'red')
# plt.text(ix+5, iy+5,'start', fontsize=10,color = 'blue')
# ____________________________-----------

phi = (X - Tx) ** 2 + (Y - Ty) ** 2

speed = np.ones_like(X)
speed[abs(Y) > 0] = 1

blur = cv2.GaussianBlur(mask, (499, 499), 5)
#cv2.imshow('blur',blur)
phi = np.ma.MaskedArray(phi, blur)
# print phi

t = skfmm.travel_time(phi, speed, dx=1e-2)

plt.contour(X, Y, phi, [0], linewidths=(1), colors='black')
plt.contour(X, Y, phi.mask, [0], linewidths=(1), colors='blue')

plt.contour(X, Y, t, 90)
plt.colorbar()

plt.imshow(img)

def dist(x, y):
	distance = t[x][y]
	return distance


# print dist(200,200)

m = 0
n = 0

distance = sqrt((xpos - Tx) ** 2 + (ypos - Ty) ** 2)
position = (xpos, ypos)
plt.plot(xpos, ypos, 'bs')
plt.plot(Tx, Ty, 'rs')

vx = ix
vy = iy

vmx = xpos
vmy = ypos

p=ix
q=iy

dis = 0
disreg = 0
path=[]
reg  = [(ix,iy)]
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
				reg.append((xpos,ypos))
			# plt.plot(m,n,'ro')
			index.append((M, N))

	#print(value)
	#print(index)
	#print(index[1])

	#print('position :', position)

	# plt.plot(xpos,ypos,'go')
	N = np.argmin(value)
	# print N

	# print 'distance:',distance
	xindex = N / 6
	yindex = N % 6

	m = xpos
	n = ypos

	xpos = xpos + index[N][0]
	ypos = ypos + index[N][1]

	line, = plt.plot([xpos, m], [ypos, n], 'green', lw=2)
	distance = sqrt((xpos - Tx) ** 2 + (ypos - Ty) ** 2)
	# print xindex,yindex

	dis = dis + sqrt((xpos - m) ** 2 + (ypos - n) ** 2)

	vmx = m
	vmy = n
	# plt.plot(vx,vy,'go')

	for i in range(1, 2):
		if (blur[int(((2 ** i - 1) * vx + vmx) / (2 ** i))][int((((2 ** i - 1) * vy + vmy) / (2 ** i)))] > 0 or
					blur[int((vx + (2 ** i - 1) * vmx) / (2 ** i))][
						int(((vy + (2 ** i - 1) * vmy) / (2 ** i)))] > 0):
			line, = plt.plot([vx, vmx], [vy, vmy], 'blue', lw=2)

			disreg = disreg + sqrt((vx - vmx) ** 2 + (vy - vmy) ** 2)
			vx = m
			vy = n

			#plt.plot(vx, vy, 'ro')
			break
		if distance < 10:
			line, = plt.plot([vx, Tx], [vy, Ty], 'blue', lw=2)
			break
	
	#plt.pause(0.000000000000000000001)
	path.append([xpos,ypos])
print(path)

for i in range(0,len(reg)-2):
				line, = plt.plot([reg[i][0], reg[i+1][0]], [reg[i][1], reg[i+1][1]], 'red', lw=2)
line, = plt.plot([xpos, Tx], [ypos, Ty], 'green', lw=2)
line, = plt.plot([reg[i+1][0], Tx], [reg[i+1][1], Ty], 'red', lw=2)
disreg = disreg + sqrt((vx - Tx) ** 2 + (vy - Ty) ** 2)

print('FMM path length : ',dis)
print('FMM + RA path length : ',disreg)

plt.gca().invert_yaxis()
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()
