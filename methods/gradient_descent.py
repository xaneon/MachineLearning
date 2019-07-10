import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# in linear regression we optimise the slope and the interception

def line(x, interception, slope):
    return interception + (slope * x)

x = np.arange(1_000)
y1 = line(x, interception=2, slope=1)
y2 = line(x, interception=2, slope=2)
y3 = line(x, interception=2, slope=3)

plt.figure()
plt.plot(x, y1, "b-", label=r"$1x + 2$")
plt.plot(x, y2, "g-", label=r"$2x + 2$")
plt.plot(x, y3, "r-", label=r"$3x + 2$")
plt.legend()
plt.savefig("line_test.png")

# we start with a simple dataset:
ys = line(x, interception=42, slope=-4) + np.random.randn(len(x)) * 420
plt.figure()
plt.plot(x, ys, "k.", label="data")
plt.legend()
plt.savefig("example_data.png")

# now we want to find a line which best describes this data
# first we only determine the estimate for the slope by minimizing the
# squared error loss function:
def loss_sqe(y, y_pred):
    return (y - y_pred) ** 2

def min_linear(x, data, slopes, intercepts):
    errors = list()
    for i, slope in enumerate(slopes):
        for j, intercept in enumerate(intercepts):
            line = slope * x + intercept
            err = min(loss_sqe(data, line))
            errors.append((err, slope, intercept))
    return  errors

errors = min_linear(x, ys, range(10), range(10))
# print(errors)
slopes = np.array([elem[1] for elem in errors])
intercepts = np.array([elem[2] for elem in errors])
errs = np.array([elem[0] for elem in errors])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(slopes, intercepts, errs, c=errs, marker="o")
ax.set_xlabel("slopes")
ax.set_ylabel("interceptions")
ax.set_zlabel("errors")
plt.savefig("errors_slope_interception.png")

# now we keep the slope constant
slope = -4
intercepts_to_test = range(-100, 100)
# and get the residuals for a number of intersections

# this is the equation with we could differentiate to intercept
def sum_of_squared_residuals(x, y, intercept, slope):
   return sum(loss_sqe(x * slope + intercept, y))

# rs = [sum(loss_sqe(x * slope + intercept, ys)) for intercept in intercepts_to_test]
rs = [sum_of_squared_residuals(x, ys, intercept, slope) for intercept in intercepts_to_test]
plt.figure()
plt.plot(intercepts_to_test, rs, "k.")
plt.xlabel("interceptions")
plt.ylabel("residuals")
plt.savefig("residuals_vs_intercepts.png")

# now let us take the derivative of this function
# (d / d intercept) sum_of_quared_residuals --> same as derivatives of each part separately
# let us start with one example:
slope = -4
intercept = 39

y = (x * -4) + 42
y_hat = y - (intercept + (x * -4))

plt.figure()
plt.plot(x, y, "b", label="line")
plt.plot(x, y_hat, "r", label="guess")
plt.savefig("line_example_derivative.png")


