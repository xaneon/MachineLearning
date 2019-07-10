import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# example data:
num = 1000
x = np.arange(1000)
theta = np.array([4, 3])  # intersection, slope
y1 = theta[0] + theta[1] * x
y2 = np.dot(theta, np.array([1, x]))

print(y1.shape, y2.shape)
print(y1[:10], y2[:10])

# let us create sample data
y_sample = y2 + np.random.randn(len(y2)) * 1000

plt.figure()
plt.plot(x, y_sample, "k.", label="sample data")
plt.plot(x, y2, "b", label="original")

# let up define the cost function
def cal_cost(theta, x, y):
    m = len(y)
    y_hat = np.dot(theta, np.array([1, x]))
    J_of_theta = (1/2*m) * np.sum(np.square(y_hat-y))
    return J_of_theta

J = cal_cost(theta, x, y_sample)
print(J)

# hypothesis
def hypothesis(theta, X):
    h = np.ones((X.shape[0], 1))
    for i in range(0, X.shape[0]):
        x = np.concatenate((np.ones(1), np.array([X[i]])), axis=0)
        h[i] = float(np.matmul(theta, x))
        return h

htest = hypothesis(theta, x)
print(htest.shape)

# now we can implement the stochastic gradient descent:
def SGD(theta, alpha, num_iters, h, X, y):
    for i in range(0, num_iters):
        theta[0] = theta[0] - (alpha) * (h - y)
        theta[1] = theta[1] - (alpha) * ((h - y) * X)
        h = theta[1] * X + theta[0]
    return theta

# now we apply this to linear regression:
def sgd_linear_regression(X, y, alpha, num_iters):
    theta = np.zeros(2)  # initialise with zeros
    h = hypothesis(theta, X)  # hypothesis
    for i in range(0, X.shape[0]):
        theta = SGD(theta, alpha, num_iters, h[i], X[i],y[i])
        # print(theta)
    theta = theta.reshape(1, 2)
    return theta

# make sure that alpha is small enough
theta_sgd = sgd_linear_regression(x, y_sample, alpha=0.000001, num_iters=100)
print(theta_sgd)

plt.plot(x, theta_sgd[0][0] + theta_sgd[0][1] * x, "r--", label="SGD")
plt.legend()
plt.savefig("sample_line.png")

