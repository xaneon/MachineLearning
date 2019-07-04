import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

X = np.arange(0.0, 5.0, 0.1)
print(X, X.shape)

a = 1
b = 0
Y = b + X * a

plt.figure()
plt.plot(X, Y, "b--")
plt.xlabel("independent variable")
plt.ylabel("dependent variable")
plt.savefig("line_example.png")

# example fuel consumption:
df = pd.read_csv("FuelConsumption.csv")
print(df.columns)
print(df.head(5))

# let us have a look at the relation between ENGINESIZE and CO2EMISSIONS
x_train = np.asanyarray(df[["ENGINESIZE"]])
y_train = np.asanyarray(df[["CO2EMISSIONS"]])
print(type(x_train), x_train.shape)
print(df[["ENGINESIZE"]].as_matrix() == x_train) # alternative possibility

# let us have a look
plt.figure()
plt.plot(x_train, y_train, "b.", label="data")
plt.xlabel("engine size")
plt.ylabel("CO2 emission")

# 1. manual
sqe = lambda y, y_hat: (y - y_hat) ** 2

# e.g. for a = 20 and b = 30.2 as an initial guess:
a = 20.0
b = 30.2
y_hat = a * x_train + b
plt.plot(x_train, y_hat, "r--", label="initial prediction")
plt.legend()
plt.savefig("enginesize_vs_emission.png")

residuals = sqe(y_train, y_hat)
plt.figure()
plt.plot(x_train, residuals, "k.")
plt.savefig("residuals_before_optimisation.png")

# now we want to find the optimal model parameter a and b
# which minimised the sqe or maybe mean(sqe)

def optimise_linear(x, y, num_steps):
    a_vars = np.linspace(0, 100, num_steps)
    b_vars = np.linspace(-50, 50, num_steps)
    aopt, bopt, r = 0, 0, 1e9
    for a in a_vars:
        for b in b_vars:
            y_hat = b + a * x
            r_mean = np.mean(sqe(y, y_hat))
            if r_mean < r:
                r = r_mean
                aopt = a
                bopt = b
    return aopt, bopt, r

num_steps = 100
aopt, bopt, ropt = optimise_linear(x_train, y_train, num_steps)

plt.figure()
plt.plot(x_train, y_train, "b.", label="data")
plt.xlabel("engine size")
plt.ylabel("CO2 emission")
plt.plot(x_train, bopt + aopt * x_train, "g--",
         label=f"predicted (n={num_steps}, a={aopt:.2f}, b={bopt:.2f}, $r_m$={ropt:.2f})")
plt.legend()
plt.savefig("data_and_line.png")

# now the same in tensorflow
a = tf.Variable(a)
b = tf.Variable(b)
y_hat = a * x_train + b
print(a, b)

# now we use the loss function as well:
loss = tf.reduce_mean(tf.square(y_hat - y_train))

# now we need the optimiser:
learning_rate = 0.05  # steps per len(x) ?
print(learning_rate, num_steps / len(x_train))  # maybe: 0.05 vs. 0.09
# optimiser = tf.train.GradientDescentOptimizer(learning_rate)
# compat, compatibility
optimiser = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)

# now we minimise the error function
train = optimiser.minimize(loss)

# now initialising and running
init = tf.global_variables_initializer()
session = tf.compat.v1.Session()
session.run(init)

loss_values = []  # you could collect the mean(r) values above as well and show the convergence
train_data = []
for step in range(num_steps):
    # session.run([optimiser, lossfunction, *modelparamaters])
    _, loss_val, a_val, b_val = session.run([train, loss, a, b])
    loss_values.append(loss_val)
    if step % 5 == 0:
        print(step, loss_val, a_val, b_val)
        train_data.append([a_val, b_val])

# now let us have a look at the loss values over iterations
plt.figure()
plt.plot(loss_values, "ro")
plt.title("Loss values (mean residuals) over iterations")
plt.savefig("loss_over_iterations.png")

# finally we would like to see the progression of the line over time
cred, cgreen, cblue = 1.0, 1.0, 1.0
plt.figure()
for par in train_data:
    cblue = cblue + (1.0 / len(train_data))
    cgreen = cgreen - (1.0 / len(train_data))
    if cblue > 1.0: cblue = 1.0
    if cgreen < 0.0: cgreen = 0.0
    a, b = par
    f_y = np.vectorize(lambda x: a*x + b)  # create function
    y_hat = f_y(x_train)
    line = plt.plot(x_train, y_hat)
    plt.setp(line, color=(cred, cgreen, cblue))

# plt.plot(x_train, y_train, "b.", label="data")
plt.plot(x_train, y_train, "b.")
green_line = mpatches.Patch(color="red", label="Data Points")
plt.legend(handles=[green_line])
plt.xlabel("engine size")
plt.ylabel("CO2 emission")
plt.savefig("line_progression_fit.png")
