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
