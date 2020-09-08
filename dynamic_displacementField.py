import math
from math import sqrt,pi
from numpy import sin, cos,exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

X = np.arange(0, 50, 1)
Y = np.arange(0, 50, 1)
X, Y = np.meshgrid(X, Y)

#obs1=(a,b)
#obs2=(c,d)

global a,b,p,q,path_distance

path_distance = 0
a = 20
b = 24
q = 15
p = 18

xpos = 50
ypos = 50
l = 0.5

fig = plt.figure()

ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 50), ylim=(0, 50))


plt.plot( xpos, ypos,'ro')

d = 100
plt.title('Dynamic field_initialised.....')
while xpos > 0:
    value = []
    
    Z = (np.sqrt(X**2 + Y**2))/2 + 10*exp((-(X-a)**2-(Y-b)**2)*0.1)+  10*exp((-(X-p)**2-(Y-q)**2)*0.1) 
    
    
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    
    for n in range (0,d):
        m = xpos + l*cos(n*2*pi/d)
        n = ypos + l*sin(n*2*pi/d)
        P = (np.sqrt(m**2 + n**2))/2  + 10*exp((-(m-a)**2-(n-b)**2)*0.1) + 10*exp((-(m-p)**2-(n-q)**2)*0.1)
        value.append(P)
        #plt.plot( m, n,'ro')
        
    plt.plot(a,b,'yo')
    plt.plot(p,q,'yo')
    
    
    n = np.argmin(value)
    
    path_distance+=l
    print path_distance
    
    a+=0.2
    b+=0.2
    p+=0.2
    q+=0.2
    m = xpos + l*cos(n*2*pi/d)
    n = ypos + l*sin(n*2*pi/d)
    line, = ax.plot([xpos,m], [ypos,n], 'black', lw=2)
    plt.plot( m, n,'ro')
    xpos = m
    ypos = n  
    plt.pause(0.000000001)
                                                     
     
ax.grid()
plt.title('Dynamic field_finished!')

plt.show()
