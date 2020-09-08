import cv2
import numpy as np

#cap = cv2.VideoCapture('obstacle.mp4')
cap = cv2.VideoCapture(1)

while(True):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv = hue sat value
    lower_blue = np.array([10, 50, 50])
    upper_blue = np.array([15, 200, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #img = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    img = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.imshow('frame', frame)
    cv2.imshow('res', res)
    cv2.imshow('mask', mask)
    cv2.imshow('mask', img)
    '''
    for (x,y,w,h) in img:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
