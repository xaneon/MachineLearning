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
# minmaxnorm = np.vectorize(lambda y: (y - np.min(y)) / np.max(y) - np.min(y))
minmaxnorm = np.vectorize(lambda y: np.abs((y - np.min(y)) / np.max(y) - np.min(y)))

plt.figure()
plt.plot(x, minmaxnorm(probfun(x)), "b",
         label=r"$\mid\frac{\exp(x)}{1 + \exp(x)}\mid$")
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

# now let us try logistic regression in tensorflow

# we will start with placeholders, specify the shape of data but not the amount of data
# this has also the advantage of feeding "batches" of data to our algorithms without changing it
num_features = X_train.shape[1] # 4: petal width, petal length, sepal width, sepal lengthi:w
num_labels = Y_train.shape[1]  # 3: categories: Iris setosa, Iris virginica, Iris versicolor
print(num_features, num_labels)

# now let us define placeholders for the feature matrix and the target values
X = tf.placeholder(tf.float32, [None, num_features])
y = tf.placeholder(tf.float32, [None, num_labels])
print(X, y)

# now we initialise the Weight Matrix and bias vector with zeros in tf
# W = tf.Variable(tf.zeros([num_features, num_labels]))
W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))

# let us take a random sample drawn from a normal distribution with tf
weights = tf.Variable(tf.random_normal([num_features, num_labels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights"))
bias = tf.Variable(tf.random_normal([1, num_labels],
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))
print(weights, bias)

# let us now apply the logistic model: Y = sigmoid(XW + b)
# we need the tensorflow functions matmul(), add() and sigmoid() for this
# with placehoders (X, y) and variables (W, b):
apply_weigths_operation = tf.matmul(X, weights, name="apply_weights")
add_bias_operation = tf.add(apply_weigths_operation, bias, name="add_bias")
activation_operation = tf.nn.sigmoid(add_bias_operation, name="activation")
print(activation_operation)

# training: we are now looking for the optimal weights which optimises the error/cost measure
# loss function: squared mean error loss function
# minimise by gradient descent: batch gradient descent

# number of epochs
numEpochs = 700  # previously called num_steps

# definition of the learning rate:
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step=1,
                                          decay_steps=X_train.shape[0],
                                          decay_rate=0.95,
                                          staircase=True)

# cost function:
cost_operation = tf.nn.l2_loss(activation_operation-y, name="squared_error_cost")
print(cost_operation)

# defining the gradient descent
# training_operation = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_operation)
training_operation = tf.compat.v1.train.GradientDescentOptimizer(learningRate).minimize(cost_operation)
print(training_operation)

# running the operations
# first initialise the weights and biases:
# tf.initialize_allvariables(): the Initialization Operation will become node in graph, etc.
session = tf.Session()

init_operation = tf.global_variables_initializer()

session.run(init_operation)

# get correct labels with argmax, see above
correct_predictions_operation = tf.equal(tf.argmax(activation_operation, 1),
                                         tf.argmax(y, 1))

# with false predictions = 0 and true predictions = 1, we can calculate the accuracy with:
# accuracy = TP + TN / (TP + TN + FP + FN)
accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions_operation, "float"))

# summary for regression:
activation_summary_operation = tf.summary.histogram("output", activation_operation)

# summary of accuracy
accuracy_summary_operation = tf.summary.scalar("accuracy", accuracy_operation)

# summary for cost
cost_summary_operation = tf.summary.scalar("cost", cost_operation)

# summary operations to check how varaibales (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=session))
biasSummary = tf.summary.histogram("biases", bias.eval(session=session))

# Merge all summaries
merged = tf.summary.merge([activation_summary_operation, accuracy_summary_operation,
                           cost_summary_operation, weightSummary, biasSummary])

# summary writer:
writer = tf.summary.FileWriter("summary_logs", session.graph)

# now we can define and run the actual training loop
cost, diff = 0, 1
epoch_values = []
accuracy_values = []
cost_values = []

for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence" %diff)
        break
    else:
        step = session.run(training_operation, feed_dict={X: X_train, y: Y_train})
        # report occasional stats:
        if i % 10 == 0:
            epoch_values.append(i)
            train_accuracy, newCost = session.run([accuracy_operation, cost_operation],
                                                  feed_dict={X: X_train, y: Y_train})
            accuracy_values.append(train_accuracy)
            cost_values.append(newCost)
            diff = abs(newCost - cost)
            cost = newCost

            #prints:
            print("step", i, "training accuracy", train_accuracy, "cost", newCost,
                  "change in cost", diff)

print("final accuracy on test set:", session.run(accuracy_operation,
                                                 feed_dict={X: X_test, y: Y_test}))

# let us have a look how the cost changes with iterations
plt.figure()
plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.savefig("cost_with_iterations.png")






