from numpy.random import randint
from collections import Counter
from matplotlib import pyplot as plt

num = 100
num_friends = [randint(0, 100) for _ in range(num)]


def bar(x, y, *args, **kwargs):
    savefile = kwargs.pop("savefile") if kwargs.get("savefile") else None
    xlabel = kwargs.pop("xlabel") if kwargs.get("xlabel") else None
    ylabel = kwargs.pop("ylabel") if kwargs.get("ylabel") else None
    title = kwargs.pop("title") if kwargs.get("title") else None
    plt.figure()
    plt.bar(x, y, *args, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefile)


def mean(x):
    return sum(x) / len(x)


def median(x):
    n = len(x)
    sorted_x = sorted(x)
    midpoint = n // 2
    if n % 2 == 1:
        return sorted_x[midpoint]
    else: # this depends on definition of the mean
          # in this case: median not necessarily the same as 50 % percentile
        lo = midpoint - 1
        hi = midpoint
        return (sorted_x[lo] + sorted_x[hi]) / 2


def quantile(x, p):
    p_index = int(p * len(x))
    return sorted(x)[p_index]


def mode(x):  # the most common values
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


if __name__ == "__main__":
    print(num_friends)
    friend_counts = Counter(num_friends)
    xs = list(friend_counts.keys())
    ys = [friend_counts[x] for x in xs]
    bar(xs, ys, color="blue", savefile="histogram_friendcounts.png",
        xlabel="# of friends", ylabel="# of people",
        title="Histogram of Friend Counts")
    print(f"Number of datapoints: {len(num_friends)}")
    sv = sorted(num_friends)
    print(f"Kleinster: {sv[0]}, zweitkleinster: {sv[1]},  zweitgroesster: {sv[-2]}")
    print(f"Mittelwert: {mean(num_friends)}")
    print(f"Zentralwert: {median(num_friends)}")
    print(f"25 % Quantile: {quantile(num_friends, 0.25)}")
    print(f"75 % Quantile: {quantile(num_friends, 0.70)}")
    print(f"50 % Quantile: {quantile(num_friends, 0.50)}")
    print(f"Die häufigsten Werte: {mode(num_friends)}")
