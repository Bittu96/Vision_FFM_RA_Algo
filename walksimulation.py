import cv2
import numpy as np
from math import pi,atan2
import matplotlib
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('test1.mp4')

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv = hue sat value
    lower_blue = np.array([60, 50, 50])
    upper_blue = np.array([150, 200, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #img = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    #cv2.imshow('mask1', img1)
    
    contourss = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    img = cv2.drawContours(frame, contourss, -1, (0,255,0), 3)
    #Mprint contours[0][0][0][0]
    
    (x,y),  radius  = cv2.minEnclosingCircle(contourss[0])
    (x1,y1),radius1 = cv2.minEnclosingCircle(contourss[1])
    (x2,y2),radius2 = cv2.minEnclosingCircle(contourss[2])
    (x3,y3),radius3 = cv2.minEnclosingCircle(contourss[3])
    (x4,y4),radius4 = cv2.minEnclosingCircle(contourss[4])
    
    center  = (int(x), int(y))
    center1 = (int(x1),int(y1))
    center2 = (int(x2),int(y2))
    center3 = (int(x3),int(y3))
    center4 = (int(x4),int(y4))
    
    radius  =  int(radius)
    radius1 =  int(radius1)
    radius2 =  int(radius2)
    radius3 =  int(radius3)
    radius4 =  int(radius4)
    
    cv2.circle(img,center,radius,(255,255,255),2)
    cv2.circle(img,center1,radius1,(255,255,255),2)
    cv2.circle(img,center2,radius2,(255,255,255),2)
    cv2.circle(img,center3,radius3,(255,255,255),2)
    cv2.circle(img,center4,radius4,(255,255,255),2)
    
    cv2.line(img,center1,center,(255,255,255),1)
    cv2.line(img,center1,center2,(255,255,255),1)
    cv2.line(img,center2,center3,(255,255,255),1)
    cv2.line(img,center3,center4,(255,255,255),1)
    
    slope1 = (atan2(center[1]-center1[1],center[0]-center1[0]))*180/pi
    slope2 = (atan2(center2[1]-center1[1],center2[0]-center1[0]))*180/pi
    
    slope3 = (atan2(center1[1]-center2[1],center1[0]-center2[0]))*180/pi
    slope3 = (atan2(center3[1]-center2[1],center3[0]-center2[0]))*180/pi
    
    slope3 = (atan2(center2[1]-center3[1],center2[0]-center3[0]))*180/pi
    slope4 = (atan2(center4[1]-center3[1],center4[0]-center3[0]))*180/pi
    
    a1 = slope2-slope1
    a2 = slope2-slope3
    a3 = slope3-slope4
    print(a1,a2,-a3)
    
    gaitReadings = ([a1, a2, -a3])

    plt.plot(1080,1920,'ro')
    plt.plot(0,0,'ro')
    plt.plot(center4[0],center4[1],'ro')
    plt.plot(center3[0],center3[1],'ro')
    plt.plot(center2[0],center2[1],'ro')
    plt.plot(center1[0],center1[1],'ro')
    plt.plot(center[0],center[1],'ro')
    line, = plt.plot([center[0], center1[0]], [center[1], center1[1]], 'white', lw=2)
    line, = plt.plot([center1[0],center2[0]], [center1[1],center2[1]], 'white', lw=2)
    line, = plt.plot([center2[0],center3[0]], [center2[1],center3[1]], 'white', lw=2)
    line, = plt.plot([center3[0],center4[0]], [center3[1],center4[1]], 'white', lw=2)
    plt.imshow(frame)
    plt.pause(0.0000000000000000001)
    
    plt.clf()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
