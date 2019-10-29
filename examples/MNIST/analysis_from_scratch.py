from load_data import loadall
from json import load
from os.path import join, sep
from sys import path
import matplotlib.pyplot as plt
from IPython.display import Markdown as md
import numpy as np
settings = load(open("foldersettings.json"))
path.append(join(f"{sep}".join(settings["projectdir"]),
                 "from_scratch"))
from mystats import avg, describe_matrix

if __name__ == "__main__":
    datadir = f"{sep}".join(settings["datadir"])
    data = loadall(datadir, prefix="*ubyte*")
    X_train, X_test = data["i60000"], data["i10000"]
    y_train, y_test = data["l60000"], data["l10000"]
    mdobj = md(describe_matrix(X_train))
    # non-square matrix
    # => no solution to Ax = b
    # What about A^T A x_hat = A^T b ?
    # If there would be a solution, it could be solved with:
    # np.linalg.solve(X_train, y_test)

    print(mdobj._repr_markdown_())
    # 1. Visualise and clean data
    plt.figure()
    plt.imshow(X_train[0, :].reshape(28, 28), cmap="gist_yarg")
    plt.savefig(join("img", "example_digit.png"))



