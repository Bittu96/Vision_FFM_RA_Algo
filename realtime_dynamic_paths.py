import math
from math import sqrt,pi
from numpy import sin, cos,exp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)

#X = np.arange(0, 50, 1)
#Y = np.arange(0, 50, 1)
#X, Y = np.meshgrid(X, Y)

#obs1=(a,b)
#obs2=(p,q)
global a,b
a = 10
b = 10
x = y =0
fig = plt.figure()
circle =plt.Circle((a,b),2,color = 'green')
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 50), ylim=(0, 50))

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
   
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    a = 50- x/10
    b = 50- y/10
    print a,b
    ax.add_artist(circle)
    #Z = (np.sqrt(X**2 + Y**2))/2 + 10*exp((-(X-a)**2-(Y-b)**2)*0.1)+  10*exp((-(X-p)**2-(Y-q)**2)*0.1) 
    #ax.contourf(X, Y, Z, zdir='z', offset=-10, cmap=plt.cm.hot)
    #ax.grid()
    #plt.title('Dynamic field_real-time')
    cv2.imshow('img',img)
    ax = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
    ax = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imshow('plot',ax)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


