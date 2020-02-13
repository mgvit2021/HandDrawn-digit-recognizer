#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os 
from keras.models import load_model
from keras.datasets import mnist
from keras.layers import Dropout,Conv2D,MaxPool2D,Dense,Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# In[18]:


def get_data():
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    X_train=X_train.astype('float32')
    X_test=X_test.astype('float32')
    X_train/=255
    X_test/=255
    X_train=X_train.reshape(X_train.shape[0],28,28,1)
    X_test=X_test.reshape(X_test.shape[0],28,28,1)
    y_train=to_categorical(y_train,num_classes=10)
    y_test=to_categorical(y_test,num_classes=10)

    return X_train,y_train,X_test,y_test

# In[23]:


def trainmodel(X_train,y_train,X_test,y_test):
    
    model=Sequential()

    model.add(Conv2D(filters=32,kernel_size=(5,5),activation="relu",input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation='softmax'))

    datagen=ImageDataGenerator(rotation_range=10,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1)

    model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])

    epochs=4
    batch_size=64

    history=model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch_size),epochs=epochs, validation_data=(X_test,y_test),steps_per_epoch=X_train.shape[0]//batch_size)

    model.save('mnist_model.h5')
    return model
################################ MAIN FUNCTION #############################################
X_train,y_train,X_test,y_test=get_data()

if(not os.path.exists('mnist_model.h5')):
    model=trainmodel(X_train,y_train,X_test,y_test)
    print('trained model')
    print(model.summary())
else:
    model=load_model('mnist_model.h5')
    print('loaded model')
    print(model.summary())





