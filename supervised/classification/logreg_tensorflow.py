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