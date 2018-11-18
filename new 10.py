import cv2
import numpy as np

cap =cv2.VideoCapture('people-walking-mp4')
fgbg=cv2.createBackgroundsubtractorMOG2()

while True :
 ret,frame=cap.read()
 fgmask=fgbg.apply(frame)
 
 cv2.imshow('original',frame)
 cv2.imshow('fg',fgmask)
 
 k=cv2.waitkey(30) & 0xff
 if k==27:
  break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np 

face_cascade.detectMultiScale(gray)

      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	  roi_gray=gry[y:y+h,x:x+w]
	  roi_color=img[y:y+h,x:x+w]
	  eyes=eye_cascade.detectMultiScale(roi_gray)
	  
	      cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		  
for (x,y,w,h) in watches :

 font=cv2.FONT_HERSHEY_SIMPLEX
 cv2.puttext(img,(x,y),(x-w,y-h),font,0.5,(0,255,255),2,cv2.LINE 