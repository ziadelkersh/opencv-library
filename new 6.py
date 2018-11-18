import cv2
import numpy as np
import matplotlib.pyplot as plt 
#                             0
img=cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE) #one color only 
#IMREAD_COLOR - 1
#IMREAD_UNCHANGED = -1

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img,cmap='gray', interpolation='bicubic')
plt.plot([50,100],[80,100],'c',linewidth=5)
plt.show()

cv2.imwrite('watchgray.png',img)

import cv2
import numpy as np

img=cv2.imread('opencv-corner-detection-sample.jpg')
gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray =np.float32(gray)

corners=cv2.goodFeaturesToTrack(gray,100,0.01,10) ##5aly a5er rakam waty
corners=np.int0(corners)

for corner in corners:
 x,y=corner.ravel()
 cv2.circle(img,(x,y),3,255,-1)

cv2.imshow('Corner',img) 

import cv2 
import numpy as np
import matplotlib.pyplot as plt

img1=cv2.imread('opencv-feature-matching-template.jpg',0)
img2=cv2.imread('opencv-feature-matching-image.jpg',0)

orb=cv2.ORB_create()

kp1,des1=orb.detetAndCompute(img1,None)
kp2,des1=orb.detetAndCompute(img2,None)

bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

matches=bf.match(des1,des2)
matches=sorted(matches,key=lambda x:x.distance)

img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],none,flags=2)
plt.imshow(img3)
plt.show()


