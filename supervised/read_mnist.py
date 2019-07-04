import gzip
import numpy as np
import matplotlib.pyplot as plt

fname_data_train = "train-images-idx3-ubyte.gz"
fname_data_test = "t10k-images-idx3-ubyte.gz"
fname_labels_train = "train-labels-idx1-ubyte.gz"
fname_labels_test = "t10k-labels-idx1-ubyte.gz"

image_size = 28
num_images = 5

def read(fname, num_images):
    f = gzip.open(fname, "r")
    f.read(16)
    out = np.ones((num_images, image_size, image_size))
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    raw = data.reshape(num_images, image_size, image_size, 1)
    for i in range(num_images):
        out[i, :, :] = np.asarray(raw[i]).squeeze()
    return out
def read_labels(fname, num_labels):
    f = gzip.open(fname, "r")
    f.read(8)
    # for i in range(num_labels):
    buf = f.read(1 * num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    labels = np.asarray(labels.reshape(1, num_labels, 1)).squeeze()

    return labels

def readall():
   num_images_train = 60_000
   num_images_test = 10_000
   data_train = read(fname_data_train, num_images_train)
   data_test = read(fname_data_test, num_images_test)
   labels_train = read_labels(fname_labels_train, num_images_train)
   labels_test = read_labels(fname_labels_test, num_images_test)
   return (data_train, data_test, labels_train, labels_test)

if __name__ == "__main__":
    dtrain, dtest, ltrain, ltest = readall()
    print(dtrain.shape, dtest.shape,
          ltrain.shape, ltest.shape)




