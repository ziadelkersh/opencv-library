import cv2
import numpy as np 

img1= cv2.imread('3D-Matplotlib.png')
img2= cv2.imread('mainlogo.png')
#img2= cv2.imread('mainsvmimage.png')
#(155,211,79) + (50,170,200)=205,381,279...translated to (205,255,255)
#add= img1 + img2  concatinate 2 pics but with another colors
#add = cv2.add(img1,img2)
#weighted= cv2.addWeighted(img1,0.6,img2,0.4,0)

rows,cols,channels=img2.shape
roi=img1[0:rows,0:cols]

img2gray =cv2.cvtColor(imag2,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)

#cv2.imshow('mask',mask)

mask_inv=cv2.bitwise_not(mask)

img1_bg=cv2.bitwise_and(roi,roi,mask=mask_inv)
img2_fg=cv2.bitwise_and(img2,img2,mask=mask)

dst=cv2.add(img1_bg,img2_fg)
img1[0:rows,0:cols]=dsl

cv2.imshow('res',img1)
cv2.imshow('mask_inv',mask_inv)
cv2.imshow('img1_bg',img1_bg)
cv2.imshow('img2_fg',img2_fg)
cv2.imshow('dst',dst)

cv2.imshow('add',add)
cv2.waitkey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

cap=cv2.VideoCapture(1)

while True :
 _,frame=cap.read()
 
 laplacian=cv2.laplacian(frame,cv2.CV_64F)
 sobelx=cv2.sobel(frame,cv2.CV_64,1,0,ksize=5)
 sobely=cv2.sobel(frame,cv2.CV_64,0,1,ksize=5)
 edges=cv2.Canny(frame,100,200)
 
 
 cv2.imshow('original',frame)
 cv2.imshow('laplacian',laplacian)
 cv2.imshow('sobelx',sobelx)
 cv2.imshow('sobely',sobely)
 cv2.imshow('sobely',edges)
 k=cv2.waitkey(5) & 0xFF
 if k==27 
  break 
 
 
 cv2.destroyAllWindows()
 cap.release()
