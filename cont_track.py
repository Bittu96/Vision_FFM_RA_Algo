import cv2
import numpy as np
from math import pi,atan2

#cap = cv2.VideoCapture('arena_dynamicFINAL3.mp4')
cap = cv2.VideoCapture(0)

while(True):
    _, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv = hue sat value
    
    '''
    #track blue color
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([150, 200, 255])
    
    #track red color
    lower_red = np.array([0, 150, 50])
    upper_red = np.array([50, 300, 150])
    
    #track green color
    lower_green = np.array([50, 50, 0])
    upper_green = np.array([70, 200, 255])
    
    #track yellow color
    lower_yellow = np.array([22, 230, 100])
    upper_yellow = np.array([25, 255, 255])
    
    #track orange color
    lower_orange = np.array([5, 80, 50])
    upper_orange = np.array([15, 190, 255])
    
    '''
    
    #track red color
    #track blue color
    lower_blue = np.array([0 , 150, 50 ])
    upper_blue = np.array([50, 300, 150])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower = np.array([0])
    upper = np.array([200])
    mask = cv2.inRange(gray, lower, upper)
    '''
    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contourss = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    img = cv2.drawContours(frame, contourss, -1, (0,0,255), 3)
    #print contours[0][0][0][0]
    '''
    (x,y),radius = cv2.minEnclosingCircle(contourss[0])
    (x1,y1),radius1 = cv2.minEnclosingCircle(contourss[1])
    
    rect = cv2.minAreaRect(contourss[1])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(frame,[box],0,(0,0,255),2)
    
    x,y,w,h = cv2.boundingRect(contours[0])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    center  = (int(x), int(y))
    center1 = (int(x1),int(y1))
    
    radius  =  int(radius)
    radius1 =  int(radius1)
    
    cv2.circle(img,center,radius,(255,255,255),2)
    cv2.circle(img,center1,radius1,(255,255,255),2)
    
    cv2.line(img,center1,center,(255,255,255),1)
    
    angle = (atan2(center[1]-center1[1],center[0]-center1[0]))*180/pi
    print 180-angle
    '''
    cv2.imshow('mask', mask)
    cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
