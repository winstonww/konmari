'''
A script to train classication model.
'''
import sys
import numpy as np
from vgg19 import vgg_fm
from keras.callbacks import ModelCheckpoint
import mnist_reader

def reshape(data):
    return np.reshape(data, ( data.shape[0], 28, 28, 1 ) )
    
if __name__=='__main__':
    model_name, batch_size ='vgg_v1', 256
    train_x, train_y = mnist_reader.load_mnist('data/fashion', kind='train')
    test_x, test_y = mnist_reader.load_mnist('data/fashion', kind='t10k')
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    train_x,test_x,train_y,test_y=reshape(train_x),reshape(test_x),np.expand_dims(train_y,-1),np.expand_dims(test_y,-1)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    model=vgg_fm(train_x.shape[1:])
    model.summary()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    #model.fit(x=train_x,y=train_y,validation_data=(test_x,test_y),batch_size=batch_size,epochs=100,
    #          callbacks=[ ModelCheckpoint(filepath=model_name+'.h5',save_best_only=True,monitor='val_acc')])
