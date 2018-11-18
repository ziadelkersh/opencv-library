import numpy as np 
import cv2

img=cv2.imread('watch.jpg',cv2.IMREAD_COLOR)

img[55,55]=[255,255,255]
px=img[55,55]
#print (px)

roi=img[100:150,100:150]=[255,255,255]

watch_face=img[37:111,107:194]
img[0:74,0:87]=watch_free


cv2.imshow('image',img)
cv2.waitkey(0)
cv2.destroyAllWindows()
#print(roi)

import cv2 
import numpy as np 

img=cv2.imread('bookpage.jpg')

retval,threshold=cv2.threshold(img,12,255,cv2.THRESH_BINARY)

grayscaled=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval2,threshold2 =cv2.threshold(grayscaled,12,255,cv2.THRESH_BINARY)
gaus=cv2.adaptiveThreshold(grayscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
retval2,otsu=cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.imshow('threshold2',threshold2)
cv2.imshow('gaus',gaus)
cv2.imshow('otsu',otsu)

cv2.waitkey(0)
cv2.destroyAllWindows()