from matplotlib import pyplot as plt
import numpy as np
from numpy.random import randint, randn

num = 42
years = [randint(1950, 2010) for i in range(num)]
gdp = [round(randint(300, 15_000) + randn(1)[0], 2)
       for _ in range(num)]
movies = ["".join([chr(randint(0, 2**8))
          for _ in range(randint(6, 14))])
          for _ in range(num)]
num_oscars = [randint(0, 42) for _ in range(num)]
xs = [i + 0.1 for i, _ in enumerate(movies)]

def plot(func, *args, **kwargs):
    plt.figure()
    savefile = kwargs.pop("savefile")
    xlabel = kwargs.pop("xlabel")
    ylabel = kwargs.pop("ylabel")
    freturn = func(*args, **kwargs)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.savefig(savefile)
    return freturn


def line(x, y, *params, **kwargs):
    return plt.plot(x, y, **kwargs)

def bar(x, y, *params, **kwargs):
    return plt.bar(x, y, **kwargs)

if __name__ == "__main__":
    # Line chart
    plot(line, x=sorted(years), y=sorted(gdp),
         color="green", marker="o", linestyle="solid",
         savefile="lineplot.png", xlabel="Years",
         ylabel="GDP")

    plot(bar, x=xs, y=num_oscars, color="blue",
         savefile="barplot.png", xlabel="Movies",
         ylabel="# Oscars")

