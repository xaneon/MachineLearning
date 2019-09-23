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


if __name__ == "__main__":
    print(num_friends)
    friend_counts = Counter(num_friends)
    xs = list(friend_counts.keys())
    ys = [friend_counts[x] for x in xs]
    bar(xs, ys, color="blue", savefile="histogram_friendcounts.png",
        xlabel="# of friends", ylabel="# of people",
        title="Histogram of Friend Counts")
