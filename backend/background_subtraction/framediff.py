#!/usr/python
import numpy as np
import cv2
from scipy.misc import imsave
import matplotlib.pyplot as plt

h = 1728//3
w = 1296//3
#before = cv2.imread('/Users/winstonww/Downloads/before.jpg')
#after =  cv2.imread('/Users/winstonww/Downloads/after.jpg')
before = cv2.cvtColor( cv2.resize( cv2.imread('/Users/winstonww/Downloads/before.jpg'), (w,h) ),cv2.COLOR_BGR2RGB)
after = cv2.cvtColor( cv2.resize( cv2.imread('/Users/winstonww/Downloads/after.jpg'), (w,h) ), cv2.COLOR_BGR2RGB)

diff = cv2.absdiff(before,after)
gray = (255-cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY))
#thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,11,2)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


kernel1 = np.ones((2,2),np.uint8)
opening1 = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel1, iterations = 2)

dk = np.ones((5,5),np.uint8)
p1 = cv2.dilate(opening1,dk,iterations=3)

ek = np.ones((2,2),np.uint8)
p2 = cv2.erode(opening1,ek,iterations=2)


p3 = np.full( p1.shape,128,dtype=np.uint8)
p3[ (p1 == 0) & (p2 == 0) ] = 0
p3[ (p1 == 255) & (p2 == 255) ] = 255
#grad = cv2.morphologyEx(thresh,cv2.MORPH_GRADIENT,kernel1, iterations = 2)

#sure foreground
dist_transform = cv2.distanceTransform(opening1,cv2.DIST_L2,5)
#ret, p2 = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)



#print(diff)
imsave('trimap.png', p3)
imsave('original.png', after)

cv2.namedWindow("p1")
cv2.namedWindow("p2")
cv2.namedWindow("p3")
cv2.namedWindow("after")
cv2.namedWindow("opening1")
cv2.namedWindow("dist_transform")
cv2.namedWindow("diff")
cv2.namedWindow("gray")
cv2.startWindowThread()
#cv2.imshow('preview',diff)
cv2.imshow('p1',p1)
cv2.imshow('p2',p2)
cv2.imshow('p3',p3)
cv2.imshow('after',after)
cv2.imshow('opening1',opening1)
cv2.imshow('dist_transform',dist_transform)
cv2.imshow('diff',diff)
cv2.imshow('gray',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

