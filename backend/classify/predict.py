#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
#print(data)

import cv2
from keras.models import load_model
import mnist_reader
import sys
import numpy as np
from nn import conv3fc2
from utils import reshape

def find_bounding_box(gray):
    # find bounding box
    rsum = np.sum(gray,axis=1)
    n,s,e,w = 0,0,0,0
    for i,r in enumerate(rsum):
        if r > 0:
            n = i
            break

    for j,r in reversed(list(enumerate(rsum))):
        if r > 0: 
            s = j
            break

    csum = np.sum(gray,axis=0)
    for i,r in enumerate(csum):
        if r > 0:
            e = i
            break

    for j,r in reversed(list(enumerate(csum))):
        if r > 0: 
            w = j
            break
    return n,s,e,w

def adjust_brightness(gray):
    gray[ gray > 0 ] += 60
    gray = np.clip(gray, 0, 255)
    gray = np.uint8(gray)
    return gray

if __name__=='__main__':
    img = cv2.imread('/Users/winstonww/konmari/src/backend/cfm_output.png')
    #img = cv2.imread('/Users/winstonww/Downloads/shirt2.jpg')
    model_name, batch_size ='conv3fc2_v1', 256
    test_x, test_y = mnist_reader.load_mnist('data/fashion', kind='t10k')
    test_x,test_y=reshape(test_x),np.expand_dims(test_y,-1)
    model = load_model('conv3fc2_v1.h5')
    print(test_x[2].shape)
    cv2.namedWindow("preview")

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    n,s,e,w = find_bounding_box(gray)

    # increase contrast and brightness
    gray = adjust_brightness(gray)

    print([n,s,e,w])
    d = max(s-n,w-e)
    print(d)
    res = np.full([d,d], 0, dtype=img.dtype)
    if s-n == d: res[:,(d-(w-e))//2:(d-(w-e))//2+(w-e)] = gray[n:s,e:w]
    else: res[(d-(s-n))//2:(d-(s-n))//2+(s-n),:] = gray[n:s,e:w]

    res = np.expand_dims(cv2.resize(res,(28,28)),-1)
    cv2.startWindowThread()
    lol = np.array(res,dtype=img.dtype)
    cv2.imshow('preview',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(res.shape)
    
    print(model.predict(np.expand_dims(test_x[2],0)))
    print(model.predict(np.expand_dims(res,0)))
