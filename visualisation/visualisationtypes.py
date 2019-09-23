from matplotlib import pyplot as plt
import numpy as np
from numpy.random import randint, randn
from collections import Counter

num = 42
years = [randint(1950, 2010) for i in range(num)]
gdp = [round(randint(300, 15_000) + randn(1)[0], 2)
       for _ in range(num)]
movies = ["".join([chr(randint(0, 2**8))
          for _ in range(randint(6, 14))])
          for _ in range(num)]
num_oscars = [randint(0, 42) for _ in range(num)]
xs = [i + 0.1 for i, _ in enumerate(movies)]
grades = [randint(0, 100) for _ in range(num)]
mentions = [randint(500, 508) for _ in range(2)]
yrs = [years[randint(0, len(years))] for _ in range(len(mentions))]
x = np.linspace(0, 5, 42)
M = np.ones((2, len(x))) * np.nan
M[0, :] = 6 * x ** 2
M[1, :] = 3.5 * x ** 2 + 4
variances = [2 ** x for x in range(9)]
bias_squared = sorted(variances, reverse=True)
total_error = [x + y for x, y in zip(variances, bias_squared)]
M2 = np.ones((3, len(variances))) * np.nan
M2[0, :] = variances; M2[1, :] = bias_squared
M2[2, :] = total_error
colors = ["green", "red", "blue"]
linestyles = ["-", "-.", ":"]
markers = [".", ".", "."]
labels = ["variance", "bias^2", "total error"]
friends = [randint(60, 80) for _ in range(10)]
minutes = [randint(120, 200) for _ in range(10)]
ls = [chr(i) for i in range(97, 107)]


def myhist(data, binsize):
    tobins = lambda elem: elem // binsize * binsize
    counter_obj = Counter(tobins(dt) for dt in data)
    return counter_obj


def plot(func, *args, **kwargs):
    plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    savefile = kwargs.pop("savefile") if kwargs.get("savefile") else None
    xlabel = kwargs.pop("xlabel") if kwargs.get("xlabel") else None
    ylabel = kwargs.pop("ylabel") if kwargs.get("ylabel") else None
    title = kwargs.pop("title") if kwargs.get("title") else None
    labelseachpoint = kwargs.pop("labelseachpoint") if kwargs.get("labelseachpoint") else None
    freturn = func(*args, **kwargs)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    if labelseachpoint:
        for label, cx, cy in zip(labelseachpoint,
                                 kwargs["x"], kwargs["y"]):
            plt.annotate(label,
                         xy=(cx, cy),
                         xytext=(5, -5), textcoords="offset points")
    plt.savefig(savefile)
    return fig, ax, freturn


def line(x, y, *params, **kwargs):
    return plt.plot(x, y, **kwargs)


def lines(x, Y, *args, **kwargs):
    plts = list()
    markers = kwargs.pop("markers") if kwargs.get("markers") else []
    colors = kwargs.pop("colors") if kwargs.get("colors") else []
    labels = kwargs.pop("labels") if kwargs.get("labels") else []
    linestyles = kwargs.pop("linestyles") if kwargs.get("linestyles") else []
    for idx, row in enumerate(Y):
        kwargs["marker"] = markers[idx] if markers else None
        kwargs["color"] = colors[idx] if colors else None
        kwargs["label"] = labels[idx] if labels else None
        kwargs["linestyle"] = linestyles[idx] if linestyles else None
        plts.append(plt.plot(x, row, **kwargs))
    return tuple(plts)


def bar(x, y, *params, **kwargs):
    return plt.bar(x, y, **kwargs)


if __name__ == "__main__":
    # Line chart
    p1, *x1 = plot(line, x=sorted(years), y=sorted(gdp),
                   color="green", marker="o", linestyle="solid",
                   savefile="lineplot.png", xlabel="Years",
                   ylabel="GDP")

    p2, *x2 = plot(bar, x=xs, y=num_oscars, color="blue",
                   savefile="barplot.png", xlabel="Movies",
                   ylabel="# Oscars")

    hist = myhist(grades, 10)
    p3, *x3 = plot(bar, x=hist.keys(), y=hist.values(),
                   color="green", savefile="histplot.png",
                   xlabel="bins",
                   ylabel="# of points per bin",
                   width=4)

    p4, *x4 = plot(bar, x=yrs, y=mentions, savefile="barylim.png",
                   width=0.8) # evtl. plt.ticklabel_format(useOffset=False)

    p5, *x5 = plot(line, x, M.T, xlabel="x", ylabel="y",
                   savefile="simple_matrix.png")

    p6, *x6 = plot(lines, x, M, xlabel="x", ylabel="y",
                   markers=["o", "x"], linestyles=["", ""],
                   colors=["red", "blue"], labels=["a", "b"],
                   savefile="functions_matrix.png")

    p7, *x7 = plot(lines, [i for i, _ in enumerate(variances)],
                   M2, xlabel="model complexity", ylabel="f(x)",
                   title="The Bias-Variance Tradeoff",
                   markers=markers, linestyles=linestyles,
                   colors=colors, labels=labels,
                   savefile="bias_variance_tradeoff.png")

    p8, *x8 = plot(line, x=friends, y=minutes,
                   marker=".", linestyle="",
                   labelseachpoint=ls,
                   savefile="scatter_with_annotate.png")
