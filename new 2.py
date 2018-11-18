import numpy as np
import cv2

img=cv2.imread('watch.jpg',cv2.IMREAD_COLOR)

cv2.line(img, (0,0),(150,150),(255,255,255),15) #white
cv2.rectangle(img,(15,25),(200,150),(0,255,0),5)
cv2.circle (img,(100,63),55,(0,0,255),-1)

pts=np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
#pts=pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True ,(0,255,255),5)

font=cv2.FONT_HERSHEY_SIMPLX
cv2.putText(img,'OpenCV Tuts !',(0,130),font , 1,(200,255,255),2,cv2.LINE_AA)

cv2.imshow('image',img)
cv2.waitkey(0)
cv2.destroyAllWindows()

import cv2 
import numpy as np 

cap=cv2.VideoCapture(1)

while True:

_,frame=cap.read()
hsv    =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

## hsv hue sat value
#lower_red=np.array([0,0,0])
#upper_red=np.array([255,255,255])
lower_red=np.array([150,150,50])
upper_red=np.array([180,255,150])

mask=cv2.inRange(hsv,lower_red,upper_red)
res=cv2.bitwise_and(frame,frame,mask=mask)
kernel=np.ones((5,5),np.uint8)
erosion=cv2.erode(mask,kernel,iterations=1)
dilation=cv2.dilate(mask,kernel,iterations=1)

opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
closing=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

##it is the difference between input image and opening of the image
#cv2.imshow('Tophat',tophat)
##it is the difference between the closing of input image and the image
#cv2.imshow('Blackhat',blackhat)

#kernel=np.ones((15,15),np.float32/225)
#smoothed=cv2.filter2D(res,-1,kernel)
#blur=cv2.GaussianBlur(res,(15,15),0)
#median=cv2.medianBlur(res,15)
#bilateral=cv2.bilateralFilter(res,15,75,75)

cv2.imshow('frame',frame)
cv2.imshow('res',res)
cv2.imshow('erosion',erosion)
cv2.imshow('dilation',dilation)
cv2.imshow('opening',opening)
cv2.imshow('closing',closing)
# cv2.imshow('mask',mask)
#cv2.imshow('smoothed',smoothed)
#cv2.imshow('bilateral',bilateral)

k=cv2.waitkey(5) & 0xFF
if k==27:
   break

cv2.destroyAllWindows()
cap.release()



#dark_red=np.uint8([[12,22,121]])
#dark_red=cv2.cvtColor(dark_red,cv2.COLOR_BGR2HSV)