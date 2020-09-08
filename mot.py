import math
from math import sqrt,pi
from numpy import sin, cos,exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = Axes3D(fig)

X = np.arange(0, 50, 1)
Y = np.arange(0, 50, 1)
X, Y = np.meshgrid(X, Y)

Z = (   (np.sqrt(X**2 + Y**2))/2 
+   10*exp((-(X-35)**2-(Y-30)**2)*0.1) 
+   10*exp((-(X-15)**2-(Y-18)**2)*0.1) 
+   10*exp((-(X-25)**2-(Y-30)**2)*0.1)
 
+   10*exp((-(X-30)**2-(Y-40)**2)*0.1)
+   10*exp((-(X-15)**2-(Y-30)**2)*0.1)   )

global a,b,path_distance
a=0
b=0
path_distance=0


xpos = 50
ypos = 50
l = 0.1

fig = plt.figure()

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 50), ylim=(0, 50))
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)

plt.plot( xpos, ypos,'ro')

d = 100

while xpos > 0:
    value = []
    
    for n in range (0,d):
        m = xpos + l*cos(n*2*pi/d)
        n = ypos + l*sin(n*2*pi/d)
        
        P = ( (np.sqrt(m**2 + n**2))/2 
        + 10*exp((-(m-35)**2-(n-30)**2)*0.1) 
        + 10*exp((-(m-15)**2-(n-18)**2)*0.1) 
        + 10*exp((-(m-30)**2-(n-40)**2)*0.1)
        + 10*exp((-(m-35)**2-(n-40)**2)*0.1)  
        + 10*exp((-(m-25)**2-(n-30)**2)*0.1)  )
        
        value.append(P)
        #plt.plot( m, n,'ro')
    
    #plt.plot( xpos, ypos,'ro')
    
    
    n = np.argmin(value)
    
    path_distance+=l
    print path_distance
    
    m = xpos + l*cos(n*2*pi/d)
    n = ypos + l*sin(n*2*pi/d)
    line, = ax.plot([xpos,m], [ypos,n], 'black', lw=2)
    xpos = m
    ypos = n   
                                                
     
ax.grid()
plt.title('potential field plot')

plt.show()
