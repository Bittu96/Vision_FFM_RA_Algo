#moving_target

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

global a,b,p,q,u,v,target_distance,path_distance

path_distance = 0

#initial positions......

#obstacle1_position
a = 20
b = 24

#obstacle2_position
q = 15
p = 18

#target_position
u = 0
v = 0

target_distance = 0

#robot initial position
xpos = 50
ypos = 50

#detection range---and--moving speed
l = 1

fig = plt.figure()

ax = fig.add_subplot(111, autoscale_on=True, xlim=(0, 50), ylim=(0, 50))

plt.plot( xpos, ypos,'ro')

target_distance = sqrt((u-xpos)**2 + (v-ypos)**2)

#no. of detecting nodes
d = 100

plt.title('Dynamic field_initialised.....')

while target_distance > 1:
    value = []
    
    Z = (np.sqrt((X-u)**2 + (Y-v)**2))/2 + 10*exp((-(X-a)**2-(Y-b)**2)*0.1)+  10*exp((-(X-p)**2-(Y-q)**2)*0.1) 
    
    
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    
    for n in range (0,d):
        m = xpos + l*cos(n*2*pi/d)
        n = ypos + l*sin(n*2*pi/d)
        P = (np.sqrt((m-u)**2 + (n-v)**2))/2  + 10*exp((-(m-a)**2-(n-b)**2)*0.1) + 10*exp((-(m-p)**2-(n-q)**2)*0.1)
        value.append(P)
        #plt.plot( m, n,'ro')
    
        
    plt.plot(a,b,'yo')
    plt.plot(p,q,'yo')
    
    n = np.argmin(value)
    
    path_distance+=l
    print ('path_distance : ',path_distance)
    
    #obstacle1_velocity_components
    a+=0.2
    b+=0.2
    
    #obstacle2_velocity_components
    p+=0.2
    q+=0.2
    
    #goal_velocity_components
    u+=0.1
    v+=0.2
    
    plt.plot(u,v,'go')
    
    m = xpos + l*cos(n*2*pi/d)
    n = ypos + l*sin(n*2*pi/d)
    
    line, = ax.plot([xpos,m], [ypos,n], 'black', lw=2)
    
    plt.plot( m, n,'ro')
    
    xpos = m
    ypos = n  
    
    target_distance = sqrt((u-xpos)**2 + (v-ypos)**2)
    
    plt.pause(0.000000001)
                                                     
     
ax.grid()
plt.title('Dynamic field_finished!')
plt.show()
