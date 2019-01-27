'''
A script to train classication model.
'''
import sys
import numpy as np
from nn import conv3fc2
from keras.callbacks import ModelCheckpoint
import mnist_reader
from utils import reshape
    
if __name__=='__main__':
    model_name, batch_size ='conv3fc2_v1', 256
    train_x, train_y = mnist_reader.load_mnist('data/fashion', kind='train')
    test_x, test_y = mnist_reader.load_mnist('data/fashion', kind='t10k')
    train_x,test_x,train_y,test_y=reshape(train_x),reshape(test_x),np.expand_dims(train_y,-1),np.expand_dims(test_y,-1)
    model=conv3fc2(train_x.shape[1:])
    model.summary()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    model.fit(x=train_x,y=train_y,validation_data=(test_x,test_y),batch_size=batch_size,epochs=100,
              callbacks=[ ModelCheckpoint(filepath=model_name+'.h5',save_best_only=True,monitor='val_acc')])
