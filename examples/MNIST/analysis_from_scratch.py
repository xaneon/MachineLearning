from load_data import loadall
from json import load
from os.path import join, sep
from sys import path
settings = load(open("foldersettings.json"))
path.append(join(f"{sep}".join(settings["projectdir"]),
                 "from_scratch"))
from mystats import avg

if __name__ == "__main__":
    datadir = f"{sep}".join(settings["datadir"])
    data = loadall(datadir, prefix="*ubyte*")
    X_train, X_test = data["i60000"], data["i10000"]
    y_train, y_test = data["l60000"], data["l10000"]
