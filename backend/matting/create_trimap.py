import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('/Users/winstonww/Downloads/shirt.jpg')
d = max(img.shape)
res = np.full(img.shape, 0, dtype=img.dtype)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=5)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
res = np.full(img.shape, 0, dtype=img.dtype)
res[markers==0] = 128
res[markers==1] = 0
res[markers==2] = 255
print(markers)

#dst = cv2.pyrDown(img)
#print(img.shape)

cv2.namedWindow("preview")
cv2.startWindowThread()
cv2.imshow('preview',res)
np.save('lolol',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
