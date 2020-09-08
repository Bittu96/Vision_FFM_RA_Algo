#dynamic_distance_field
import numpy as np
import pylab as plt
import skfmm
import math
from math import sqrt,pi
from numpy import sin, cos,exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

'''
fig = plt.figure()
ax = Axes3D(fig)
'''
distance = 0

X, Y = np.meshgrid(np.linspace(0,200,201), np.linspace(0,200,201))
phi = -1*np.ones_like(X)

phi = (X**2 + Y**2)
'''
#phi[np.logical_and(np.abs(Y)<0.25, X>-0.75)] = 1
plt.contour(X, Y, phi,[0], linewidths=(3), colors='black')
plt.title('Boundary location: the zero contour of phi')
#plt.savefig('2d_phi.png')
'''

d = skfmm.distance(phi, dx=1e-2)
'''
plt.title('Distance from the boundary')
plt.contour(X, Y, phi,[0], linewidths=(3), colors='black')
plt.contour(X, Y, abs(d), 100)
plt.colorbar()
#plt.savefig('2d_phi_distance.png')
plt.show()
'''


#plt.savefig('2d_phi_travel_time_mask.png')

#print t

def dist(x,y):
	distance = t[x][y]
	return distance 
#print dist(200,200)


xpos = 199
ypos = 199
m = 0
n = 0
distance = sqrt(xpos**2 + ypos**2)
position = (xpos,ypos)



speed = np.ones_like(X)
speed[abs(Y)>0] = 1
t = skfmm.travel_time(phi, speed, dx=1e-2)
a = 50
b = 90
i = 199



#obstacle mask
while distance > 10 :
		mask = np.logical_and(abs(X-a)<10, abs(Y-b)<10) 
		phi  = np.ma.MaskedArray(phi, mask)
		t    = skfmm.travel_time(phi, speed, dx=1e-2)
		plt.title('Travel time from the target with boundary conditions')
		plt.contour(X, Y, phi, [0], linewidths=(3), colors='black')
		plt.contour(X, Y, phi.mask, [0], linewidths=(3), colors='red')
		plt.contour(X, Y, t, 50)
		plt.colorbar()
		a = a+0.2
		plt.pause(0.00000000000000000001)
		plt.clf()	
	

		value = []
		index = []
		position = (xpos,ypos)

		for M in range(0,3):
			for N in range (0,3):
					m = xpos + M-1
					n = ypos + N-1
					P = dist(n,m)
					if P > 0:
						value.append(P)
					else:
						value.append(10)
					#plt.plot(m,n,'ro')
					index.append((M-1,N-1))
				
		#print value
		#print index
		#print index[1]
		#print('position :',position)
		plt.plot(xpos,ypos,'bo')
		N = np.argmin(value)
		#print N
		#print 'distance:',distance
		xindex = N/3
		yindex = N%3
		

		xpos = xpos + index[N][0]
		ypos = ypos + index[N][1]
		distance = sqrt(xpos**2 + ypos**2)
		#print xindex,yindex
		
plt.title('finished')
plt.show()
