import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

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