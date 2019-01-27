#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
#print(data)
#
from keras.models import load_model
import mnist_reader
import sys
import numpy as np
from nn import conv3fc2
from utils import reshape

if __name__=='__main__':
    model_name, batch_size ='conv3fc2_v1', 256
    test_x, test_y = mnist_reader.load_mnist('data/fashion', kind='t10k')
    test_x,test_y=reshape(test_x),np.expand_dims(test_y,-1)
    model = load_model('conv3fc2_v1.h5')
    print(test_x[0].shape)
    print(model.predict(np.expand_dims(test_x[0],0)))
