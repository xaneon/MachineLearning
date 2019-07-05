import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from read_mnist import readall

# we start with the input layer
dims = (28, 28)
ImageMatrix = np.ones(dims)
# ImageMatrix = np.arange(np.multiply(*dims)).reshape(dims)
for i, row in enumerate(ImageMatrix):
    for j, item in enumerate(row):
        ImageMatrix[i, j] = np.random.randint(0, 255, 1)

plt.figure()
plt.imshow(ImageMatrix)
plt.savefig("random_image.png")

dtrain, dtest, ltrain, ltest = readall()

example_num = 42
example_image, example_label = dtrain[example_num], ltrain[example_num]
plt.figure()
plt.imshow(example_image)
plt.title(f"{example_label}")
plt.savefig("example_digit.png")

# now let us illustrate the steps in the CNN
# 1. convolutional layer: extracting features with filters / kernels

# e.g. simple 3x3 kernel:
# example kernel for edge detection:
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
plt.figure()
plt.imshow(kernel)
plt.savefig("example_kernel.png")

# let us apply the kernel
first_step = example_image.ravel()[:9].reshape((3, 3))
first_value_of_feature_map = np.sum(first_step * kernel)

# now let us do that with the sliding kernel:
feature_map = np.zeros(example_image.shape)

for x in range(example_image.shape[0] - kernel.shape[0] + 1):
    for y in range(example_image.shape[1] - kernel.shape[0] + 1):
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                feature_map[x][y] += example_image[x + i][y + j] * kernel[i][j]

# apply the filter:
# feature_map = np.convolve(example_image, kernel, "valid")

# this could be continued with multiple kernels

print(feature_map)
plt.figure()
plt.imshow(feature_map)
plt.savefig("example_featuremap.png")

