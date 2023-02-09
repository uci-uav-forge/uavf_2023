import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('circleblursample-80graybrown.jpg')
#img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
h,w,c= img.shape
assert img is not None, "file could not be read, check with os.path.exists()"
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (0,0,h,w)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv.imshow('img', img)
cv.waitKey(0)
# plt.imshow(img),plt.colorbar(),plt.show()