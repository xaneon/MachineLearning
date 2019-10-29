from load_data import loadall
from json import load
from os.path import join, sep
from sys import path
import numpy as np
settings = load(open("foldersettings.json"))
path.append(join(f"{sep}".join(settings["projectdir"]),
                 "from_scratch"))
from mystats import avg

def describe(data: np.ndarray) -> str:
    n, m = data.shape
    r = np.linalg.matrix_rank(data)
    dtype = data.dtype
    res = (f"# rows n: \t\t\t{n}\n" +
           f"# cols m: \t\t\t{m}\n" +
           f"dim N(A): \t\t\t{m - r}\n" +
           f"dim C(A) (=rank r): \t\t{r}\n" +
           f"Data type: \t\t\t{dtype}\n")
    return res

if __name__ == "__main__":
    datadir = f"{sep}".join(settings["datadir"])
    data = loadall(datadir, prefix="*ubyte*")
    X_train, X_test = data["i60000"], data["i10000"]
    y_train, y_test = data["l60000"], data["l10000"]
    print(describe(X_train))

