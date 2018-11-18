import cv2
import numpy as np

img_bgr =cv2.imread('opencv-template-matching-python-tutorial.jpg')
img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

template =cv2.imread('opencv-template-for-matching.jpg',0)
w,h=template.shape[::-1]

res=cv2.matchtemplate(img_gray,template,cv2.TM_COEFF_MORMED)
threshold=0.8
loc=np.where(res>=threshold)

for pt in zip(*loc[::-1]):
 cv2.rectangle(img_bgr,pt,(pt[0]+w),(pt[1]+h),(0,255,255),2)
 
cv2.imshow('detected',img_bgr)

import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('opencv-python-foreground-extraction-tutorial.jpg')
mask=np.zeros(img.shape[:2],np.uint8)

bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
rect=(161,79,150,150)


cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==2)|(mask==0),0,1,astype('uint8))
img=img*mask2[:,:,np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()