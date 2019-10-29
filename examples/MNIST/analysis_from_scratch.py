from load_data import loadall
from json import load
from os.path import join, sep
from sys import path
settings = load(open("foldersettings.json"))
path.append(join(f"{sep}".join(settings["projectdir"]),
                 "from_scratch"))
from mystats import avg

if __name__ == "__main__":
    print(avg([1, 3, 5]))
