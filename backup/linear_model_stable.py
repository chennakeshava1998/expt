#!/usr/bin/env python
# coding: utf-8

# https://www.tensorflow.org/guide/eager#object_based_saving

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# tf.enable_eager_execution()

from TPM_from_VCS.data import toy_data_generator
from TPM_from_VCS.models.etp import etp

tf.executing_eagerly()

tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

tf.trainable_variables()


# In[ ]:


# utility function to plot the points
def display_points(data1, data2):
    data = (data1, data2)
    groups = ('pred_coordinates', 'physical_coordinates')
    color = ("red", "green")
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, color, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()


# In[ ]:


class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tf.Variable(tf.random_normal((2, 5), dtype=tf.float64), name='weight', trainable=True)
    self.B = tf.Variable(tf.random_normal((2, 1), dtype=tf.float64), name='bias', trainable=True)

    
  def call(self, inputs):
    ans = []

    for i in range(0,inputs.shape[0],5):
        inp = [inputs[i], inputs[i+1], inputs[i+2], inputs[i+3], inputs[i+4]]
        ans.append(tf.add(tf.matmul(self.W, tf.reshape(inp, (5, -1))), self.B))
    return ans

# The loss function to be optimized
def loss(model, inputs, targets):
  ans = etp.get_best_etp(model(inputs), targets)
  tf.summary.histogram("loss", ans)

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
    w_grad = np.ones((2, 5), dtype=np.float64) * loss_value
    b_grad = np.ones((2, 1), dtype=np.float64) * loss_value
    
    tf.summary.histogram("weight_gradients", w_grad)
    tf.summary.histogram("bias_gradients", b_grad)
    
    return [w_grad, b_grad]
    

# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # use Adam later


# In[ ]:


dataset_size = 10
a = toy_data_generator.create_dataset(dataset_size)


training_inputs = a[0][0]
training_outputs = a[0][1]

for i in range(dataset_size-1):
    training_inputs = np.concatenate((training_inputs, a[i][0]))
    training_outputs = np.concatenate((training_outputs, a[i][1]))


print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))


# display_points(model(training_inputs), training_outputs)


# In[ ]:


# Training loop
loss_history = []
for j in range(1):
    for i in range(10):
      grads = temp_grad(model, training_inputs, training_outputs)
      optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                                global_step=tf.train.get_or_create_global_step())
      if i % 2 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))
        
      loss_history.append(loss(model, training_inputs, training_outputs))

#           merge = tf.summary.merge_all()

#           train_writer.add_summary(summary, i*j)



    a = toy_data_generator.create_dataset(1)
    training_inputs = a[0][0]
    training_outputs = a[0][1]

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

#     merge, final_loss = sess.run([merge, model.loss])


# https://www.tensorflow.org/guide/summaries_and_tensorboard   : use tensorboard

# https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
