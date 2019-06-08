#!/usr/bin/env python
# coding: utf-8

# https://www.tensorflow.org/guide/eager#object_based_saving

# In[1]:


import tensorflow as tf
from TPM_from_VCS.data import toy_data_generator
from TPM_from_VCS.models.etp import etp
import numpy as np


# In[2]:


class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tf.Variable(tf.random_normal((2, 5), dtype=tf.float32), name='weight', trainable=True)
    self.B = tf.Variable(tf.random_normal((2, 1), dtype=tf.float32), name='bias', trainable=True)

    
  def call(self, inputs):
    ans = []

    for i in range(0,inputs.shape[0],5):
        inp = [inputs[i], inputs[i+1], inputs[i+2], inputs[i+3], inputs[i+4]]
        ans.append(tf.add(tf.matmul(self.W, tf.reshape(inp, (5, -1))), self.B))
    return ans

# The loss function to be optimized
def loss(model, inputs, targets):
  ans = etp.get_best_etp(model(inputs), targets)
  

  return ans

def grad(model, inputs, targets):
  with tf.GradientTape(persistent=True) as tape:
    loss_value = loss(model, inputs, tf.cast(targets, dtype=tf.float64))
  return tape.gradient(loss_value, [model.W, model.B])

def modulate_loss(loss_value, inputs):
    return loss_value/inputs.shape[0]
    

def temp_grad(model, inputs, targets):
    loss_value = loss(model, inputs, tf.cast(targets, dtype=tf.float64))
    loss_value = modulate_loss(loss_value, inputs)
    w_grad = tf.ones((2, 5), dtype=tf.float32) * tf.cast(loss_value, dtype=tf.float32)
    b_grad = tf.ones((2, 1), dtype=tf.float32) * tf.cast(loss_value, dtype=tf.float32)
    
    tf.summary.histogram("weight_gradients", w_grad)
    tf.summary.histogram("bias_gradients", b_grad)
    
    return [w_grad, b_grad]
    

# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # use Adam later


# In[3]:


a = toy_data_generator.create_dataset(1)
training_inputs = a[0][0]
training_outputs = a[0][1]


# In[1]:


# Training loop
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    loss_history = []
    for j in range(2):
        for i in range(1):
          grads = temp_grad(model, training_inputs, training_outputs)
          optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                                    global_step=tf.train.get_or_create_global_step())
          if i % 2 == 0:
            print("Loss at step {}: {}".format(i, loss(model, training_inputs, training_outputs)))

          loss_history.append(loss(model, training_inputs, training_outputs))

        a = toy_data_generator.create_dataset(1)
        training_inputs = a[0][0]
        training_outputs = a[0][1]

#     print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
#     print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
    
    sess.run(init_op)
    loss_history = sess.run(loss_history)

#     merge, final_loss = sess.run([merge, model.loss])


# In[1]:


len(loss_history)


# In[6]:


loss_history


# https://www.tensorflow.org/guide/summaries_and_tensorboard   : use tensorboard

# https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
