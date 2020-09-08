import numpy as np
import pylab as plt
import skfmm
import math
from math import sqrt, pi
from numpy import sin, cos, exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

'''
fig = plt.figure()
ax = Axes3D(fig)
'''
distance = 0

X, Y = np.meshgrid(np.linspace(0, 500, 501), np.linspace(0, 500, 501))
phi = -1 * np.ones_like(X)

Tx = 30
Ty = 30

phi = ((X - Tx) ** 2 + (Y - Ty) ** 2)
'''
#phi[np.logical_and(np.abs(Y)<0.25, X>-0.75)] = 1
plt.contour(X, Y, phi,[0], linewidths=(3), colors='black')
plt.title('Boundary location: the zero contour of phi')
#plt.savefig('2d_phi.png')
'''
'''
d = skfmm.distance(phi, dx=1e-2)
plt.title('Distance from the boundary')
plt.contour(X, Y, phi,[0], linewidths=(3), colors='black')
plt.contour(X, Y, abs(d), 100)
plt.colorbar()
#plt.savefig('2d_phi_distance.png')
plt.show()
print d
'''
speed = np.ones_like(X)
speed[abs(Y) > 0] = 1
t = skfmm.travel_time(phi, speed, dx=1e-2)

# obstacle mask
mask = (np.logical_and(abs(X - 300) < 5, abs(Y - 500) < 400) + np.logical_and(abs(X - 170) < 5, abs(Y - 6) < 400))

phi = np.ma.MaskedArray(phi, mask)
t = skfmm.travel_time(phi, speed, dx=1e-2)
plt.title('Travel time from the boundary with an obstacle')
plt.contour(X, Y, phi, [0], linewidths=(3), colors='black')
plt.contour(X, Y, phi.mask, [0], linewidths=(3), colors='red')
plt.contour(X, Y, t, 100)
plt.colorbar()


# plt.savefig('2d_phi_travel_time_mask.png')

# print t

def dist(x, y):
    distance = t[y][x]
    return distance


# print dist(200,200)


xpos = 400
ypos = 200
m = 0
n = 0

while distance > 10:
    value = []
    index = []
    position = (xpos, ypos)
    for M in range(0, 6):
        for N in range(0, 3):
            m = xpos + M - 2
            n = ypos + N - 2
            P = dist(m, n)
            if P > 0:
                value.append(P)
            else:
                value.append(10)
                # plt.plot(m,n,'ro')
            index.append((M - 1, N - 1))
    # print value
    # print index
    #print P
    plt.plot(xpos, ypos, 'bo')
    #print position
    N = np.argmin(value)
    #print N
    xindex = N / 3
    yindex = N % 3
    xpos = xpos + index[N][0]
    ypos = ypos + index[N][1]
    plt.plot(m, n, 'ro')
    distance = sqrt((xpos - Tx) ** 2 + (ypos - Ty) ** 2)
    plt.pause(0.000000000001)
    # print xindex,yindex

plt.show()
