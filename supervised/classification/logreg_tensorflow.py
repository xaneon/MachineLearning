import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# let us have a look at the probability function first:
x = np.linspace(-10, 10, 100)
probfun = np.vectorize(lambda y: np.exp(y) / (1 + np.exp(y)))
minmaxnorm = np.vectorize(lambda y: (y - np.min(y)) / np.max(y) - np.min(y))

plt.figure()
plt.plot(x, minmaxnorm(probfun(x)), "b",
         label=r"$\frac{\exp(x)}{1 + \exp(x)}$")
plt.legend()
plt.savefig("probfunction_logistic_sigmoid.png")

# iris dataset
iris = load_iris()
data = iris.data
targets = iris.target
print(type(data), data.shape)
print(type(targets), targets.shape)

X = data # in the example the last datapoint was excluded with
y = targets # data[:-1, :], I do not know why

print(y.shape, np.unique(y))

# now the categories one hot encoded, because 1 is not greater than 2, etc.
Y = pd.get_dummies(y)
print(type(Y), Y.shape, np.unique(Y))

# from dataframe to numpy.ndarray
Y = Y.values
print(type(Y), Y.shape, np.unique(Y))

# now let us split the data into training and testing sets
# Testsize = 33 %
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33,
                                                    random_state=42)
print(X_test.shape, Y_test.shape)
print(X_train.shape, Y_train.shape)

# first manually again:
num_features = X_train.shape[1]
num_labels = Y_train.shape[1]
print(num_features, num_labels)

sigmoid = np.vectorize(lambda y: np.exp(y) / (1 + np.exp(y)))
squared_error = lambda y, y_hat: (y - y_hat)**2
mean_sqe = lambda y, y_hat: np.mean(squared_error(y, y_hat), axis=0)

b = 1
W = np.ones((num_features, num_labels))
Y_prob = sigmoid(Y_train)
Y_prob_hat = sigmoid(np.dot(X_train, W))
print(Y_prob_hat.shape, Y_prob.shape)
print(mean_sqe(Y_prob, Y_prob_hat))

# now lets try to find good weights:
def optimise(X, Y, num_steps):
    nfeatures = X.shape[1]
    nlabels = Y.shape[1]
    w_vals = np.linspace(-10, 10, num_steps)
    b_vals = np.linspace(-10, 10, num_steps)
    Y_prob = sigmoid(Y)
    rs = np.ones(nlabels) * 1e9
    Wopt = np.ones((nfeatures, nlabels))
    W0 = np.ones((nfeatures, nlabels))
    bopt = np.zeros(nlabels)
    for w, b in zip(w_vals, b_vals):
        W = W0 * w
        Y_prob_hat = sigmoid(np.dot(X, W) + b)
        params = mean_sqe(Y_prob, Y_prob_hat)
        for idx, param in enumerate(params):
            if param < rs[idx]:
                rs[idx] = param
                Wopt[:, idx] = W[:, idx]
                bopt[idx] = b
    return Wopt, bopt, rs

Wopt, bopt, rs = optimise(X_train, Y_train, 100)
print(Wopt, bopt, rs)
Y_train_prob = sigmoid(Y_train)
Y_train_prob_hat = sigmoid(np.dot(X_train, Wopt) + b)
print(Y_train_prob_hat.shape, Y_train_prob.shape)

# let us just take the smallest difference here:
diff = np.abs(Y_train_prob - Y_train_prob_hat)
y_hat = np.argmin(diff, axis=1)
y_hat_onehot = pd.get_dummies(y_hat).values
print(Y_train - y_hat_onehot)

# performance looks very good, not let us see how well it generalises
Y_test_prob = sigmoid(Y_test)
# Wopt = np.ones((X_test.shape[1], Y_test.shape[1])) * Wopt[0]
Y_test_prob_hat = sigmoid(np.dot(X_test, Wopt) + b)
diff_test = np.abs(Y_test_prob - Y_test_prob_hat)
y_hat_test = np.argmin(diff_test, axis=1)
y_hat_onehot_test = pd.get_dummies(y_hat_test).values
print(Y_test - y_hat_onehot_test)
# prediction works perfectly



# Later: maybe plot the weights in multiple dimensions and see how they converge










