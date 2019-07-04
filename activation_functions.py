import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

# function which plots a surface for an arbitrary activation function.
# The plot will be done for all possible values of weigt and bias between -05 and 0.5 with a
# step of 0.05. The input, the weight and the bias are one-dimensional. Input can be passed as an argument.
def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-.5, .5, .05)
    bs = np.arange(-.5, .5, .05)

    X, Y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w*i + b)).eval(session=session) \
                   for w,b in zip(np.ravel(X), np.ravel(Y))])
    Z = os.reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)

# here it is illustrated how in tensorflow to compute the weighted sum
# that would go into an artificial neuron for example and direct it to
# the activation function:

session = tf.compat.v1.Session()
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
w = tf.random.normal(shape=[3, 3])
b = tf.random.normal(shape=[1, 3])
def func(x): return x # example activation function
act = func(tf.matmul(i, w) + w)
r = act.eval(session=session)
print(r)

plot_act(1.0, func)
plt.savefig("linear_activation_function_weights_tf.png")

# sigmoid function f(x) = 1/(1+exp(-x)), interval: (0, 1)
plot_act(1, tf.sigmoid)
plt.savefig("sigmoid_activation_function_weights_tf.png")

# using the sigmoid function in a neural net layer:
act = tf.sigmoid(tf.matmul(i, w) + b)
r = act.eval(session=session)
print(r)

# the arctangent and hyperbolic tangent functions are based on tangent functions
# argtangent: f(x) = tan^-1x, interval: (-pi/2, pi/2)

# hyperbolic tangent (tanh), f(x) = 2/(1 + exp(-2x)) - 1, interval: (-1, 1)
plot_act(1, tf.tanh)
plt.savefig("tanh_activation_function_weights_tf.png")

# using tanh in neural net layer
act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=session)

# linear unit function or rectified linear unit (ReLU), 0 for x <= 0, f(x) for x > 0
# ReLU take care of vanishing and exploding gradient, plus: biological plausible
plot_act(1, tf.nn.relu)
plt.savefig("relu_activation_function_weights_tf.png")

# relu in a neural net later
act = tf.nn.relu(tf.matmul(i, w) + b)
r = act.eval(session=session)
print(r)



