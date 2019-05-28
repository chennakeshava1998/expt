#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector

import datetime


# In[2]:


HIDDEN_SIZE = 1024
MAX_NODES = 100
MAX_VC_DIM = 10


# In[3]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(HIDDEN_SIZE, input_shape=(MAX_NODES, MAX_VC_DIM)))
model.add(RepeatVector(MAX_NODES))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(Dense(MAX_VC_DIM))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


# In[4]:


data = np.load('../../../expt/datasets/2019-05-28 11:37:37.368580_30.npy')


# In[5]:


X, y = data[0], data[1]


# In[7]:


X.shape


# In[8]:


model.fit(X, y, epochs=10, validation_split=0.33)


# In[9]:


get_ipython().system('pwd')


# In[25]:


model.save(str(datetime.datetime.now()) + '.h5')


# In[ ]:




