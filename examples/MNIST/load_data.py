import struct
import numpy as np
from glob import glob
import os
import gzip 
import shutil

def load_images(imagesfile):
    with open(imagesfile, 'rb') as fid:
        (magic, num, 
         rows, cols) = struct.unpack(">IIII",
                                     fid.read(16))
        images = np.fromfile(fid,
                             dtype=np.uint8).reshape(num, 28*28)
    return images


def load_labels(labelsfile):
    with open(labelsfile, 'rb') as fid:
        magic, n = struct.unpack('>II',
                                 fid.read(8))
        labels = np.fromfile(fid, dtype=np.uint8)
    return labels


def load(imagesfile, labelsfile):
    return load_images(), load_labels()

def loadall(path, prefix="*"):
    filenames = glob(os.path.join(path, prefix))
    contents = dict()
    for file in filenames:
        if ".gz" in file:
            with open(file, "rb") as f_read:
                with open(file.replace(".gz", ""), "wb") as f_write:
                    shutil.copyfileobj(f_read, f_write)
        elif "images" in file:
            npstruct = load_images(file)
            name = "i" + str(npstruct.shape[0])
            contents[name] = npstruct
        elif "labels" in file:
            npstruct = load_labels(file)
            name = "l" + str(npstruct.shape[0])
            contents[name] = npstruct 
    return contents
