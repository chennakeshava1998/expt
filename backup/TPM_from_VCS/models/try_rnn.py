#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
import tensorflow.contrib.keras as keras

from keras import layers
import numpy as np
import os
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.callbacks import TensorBoard
from keras import optimizers

import datetime

from etp import etp


# In[2]:


HIDDEN_SIZE = 1024
MAX_NODES = 100
MAX_VC_DIM = 10


# In[3]:


NAME = "double_lstm_encdec_{}".format(str(datetime.datetime.now()))


# In[4]:


tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


# In[5]:


data = np.load('../../datasets/2019-05-29 14:16:06.689081_100.npy')


# In[6]:


X, y, phy_coordinates = data[0], data[1], data[2]


# In[7]:


def custom_loss(y_true, y_pred):
    print('ETP: Matrix y_true shape: {}'.format(y_true.shape))
    print('ETP: Matrix y_pred shape: {}'.format(y_pred.shape))
    return etp.get_best_etp(y_true, y_pred)


# In[8]:


adam = optimizers.Adam(lr=1e-6)


# In[9]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(HIDDEN_SIZE, input_shape=(MAX_NODES, MAX_VC_DIM)))
model.add(RepeatVector(MAX_NODES))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(Dense(MAX_VC_DIM))

model.compile(loss=custom_loss,
              optimizer=adam,
              metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(X, phy_coordinates, epochs=2, validation_split=0.33, callbacks=[tensorboard])


# In[ ]:


model.save(str(datetime.datetime.now()) + '.h5')


# In[ ]:




