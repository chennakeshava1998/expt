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


# In[2]:


np.random.seed(23)
MAX_NODES = 100 # Atmost 10 points in the network for now
MAX_VC_DIM = 10
HIDDEN_SIZE = 1024


# In[16]:


def get_vector(m, v):
    a = np.zeros((MAX_NODES, MAX_VC_DIM))
    
#     print('Original Shape {}'.format(v.shape))
#     print('New Shape {}'.format(a.shape))
    
    for i in range(0, v.shape[0]):
        for j in range(0, v.shape[1]):
            a[i, j] = v[i, j]
    
    
#     print('DEBUG: Original Vector = {}'.format(v))
#     print('DEBUG: Final Vector = {}'.format(a))

    return a


# In[4]:


def generate_tpm_from_vcs(P):

    U, s, V = np.linalg.svd(P)
    S = np.zeros((U.shape[0], V.shape[0]))
    S[:len(s), :len(s)] = np.diag(s)
    
    TC = np.dot(U, S)[:, [1, 2]] # extracting columnss 2 and 3
    return TC


# In[5]:


def create_dataset(size):
    X = [] # array of tuples
    y = []
    
    
    for i in range(0, size):
        # effectively 100 numbers in one-timestep is fed into rnn (m distances)
        m = np.random.randint(low=3, high=10)
        # number of anchors could be 1% of total nodes
        n = np.random.randint(low=m, high=100) 
        P = np.random.rand(n, m)

        ans = generate_tpm_from_vcs(P)

        if np.iscomplex(ans.flatten()).any():
            print("Is the TC matrix complex : {}".format(np.iscomplex(ans.flatten())))
        
        # broadcast the input vector into MAX_VC_DIM length
        P = get_vector(MAX_VC_DIM, P)
        ans = get_vector(MAX_VC_DIM, ans)

        # dataset[i] = (P.tolist(), ans.tolist())
#         dataset.append((P, ans))
        X.append(P)
        y.append(ans)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# In[8]:


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


# In[9]:


inp_data = np.random.randn(2, 100, 10)


# In[10]:


out_data = np.random.randn(2, 100, 10)


# In[11]:


model.fit(inp_data, out_data, epochs=1)


# In[21]:


X, y = create_dataset(10)


# In[22]:


X.shape


# In[23]:


y.shape


# In[ ]:


model.fit(X, y, epochs=100, validation_split=0.33)


# In[ ]:





# In[26]:


temp = model.predict(inp_data, verbose=0)


# In[ ]:




