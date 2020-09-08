import math
from math import sqrt,pi
from numpy import sin, cos,exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

global path_distance,a,b,p,q
a=25
b=25
p=25
q=30
path_distance=0
fig = plt.figure()
ax = Axes3D(fig)

X = np.arange(0, 50, 1)
Y = np.arange(0, 50, 1)
X, Y = np.meshgrid(X, Y)

Z = (np.sqrt(X**2 + Y**2))/2 + 10*exp((-(X-a)**2-(Y-b)**2)*0.1)  + 10*exp((-(X-p)**2-(Y-q)**2)*0.1) + 10*exp((-(X-q)**2-(Y-p)**2)*0.1)

xpos = 50
ypos = 50
l = 4

obstacle1 = plt.Circle((a,b),3,color = 'black')
obstacle2 = plt.Circle((p,q),3,color = 'black')
obstacle3 = plt.Circle((q,p),3,color = 'black')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 50), ylim=(0, 50))
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)

plt.plot( xpos, ypos,'ro')
ax.add_artist(obstacle1)
ax.add_artist(obstacle2)
ax.add_artist(obstacle3)
d = 100
N=5
while xpos > 0:
    value = []
    
    for n in range (0,d):
        m = xpos + l*cos(n*2*pi/d)
        n = ypos + l*sin(n*2*pi/d)
        P = (np.sqrt(m**2 + n**2))/2 + 10*exp((-(m-a)**2-(n-b)**2)*0.1) + 10*exp((-(m-p)**2-(n-q)**2)*0.1) + 10*exp((-(m-q)**2-(n-p)**2)*0.1)
        value.append(P)
        #plt.plot( m, n,'ro')
    
    
    #plt.plot(m,n,'ro')
    
    n = np.argmin(value)
    path_distance+=l
    print path_distance
    
    m = xpos + l*cos(n*2*pi/d)
    n = ypos + l*sin(n*2*pi/d)
    line, = ax.plot([xpos,m], [ypos,n], 'black', lw=2)
    
    xpos = m
    ypos = n 
    plt.plot( xpos, ypos,'ro')
    circle = plt.Circle((m,n),l,fill=False)
    ax.add_artist(circle) 
    plt.pause(0.0000001)

ax.grid()
plt.title('potential field plot')
plt.show()
