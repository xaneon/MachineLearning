import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from read_mnist import readall
from numpy.lib.stride_tricks import as_strided

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

## e.g. simple 3x3 kernel:
## example kernel for edge detection:
# kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
# kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
##  example kernels for sharpening
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
## example kernels for box blur, gaussian blur
# kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1/9)
kernel_gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1/16)
kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
plt.figure()
plt.imshow(kernel)
plt.savefig("example_kernel.png")

# let us apply the kernel
first_step = example_image.ravel()[:9].reshape((3, 3))
first_value_of_feature_map = np.sum(first_step * kernel)

# now let us do that with the sliding kernel:
feature_map = np.zeros(example_image.shape)

def apply_filter(image, kernel):
    for x in range(image.shape[0] - kernel.shape[0] + 1):
        for y in range(image.shape[1] - kernel.shape[0] + 1):
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    feature_map[x][y] += image[x + i][y + j] * kernel[i][j]
    return feature_map

feature_map = apply_filter(example_image, kernel)
plt.figure()
plt.imshow(feature_map, cmap="gray")
plt.savefig("example_featuremap_edge.png")

feature_map_gauss = apply_filter(example_image, kernel_gauss)
plt.figure()
plt.imshow(feature_map_gauss, cmap="gray")
plt.savefig("example_featuremap_gauss.png")

# now apply the ReLU on feature_maps
idcs_smaller_0 = np.where(feature_map < 0 )
feature_map[idcs_smaller_0] = 0
plt.figure()
plt.imshow(feature_map, cmap="gray")
plt.savefig("example_featuremap_edge_relu.png")

idcs_smaller_0_gauss = np.where(feature_map_gauss < 0 )
feature_map_gauss[idcs_smaller_0_gauss] = 0
plt.figure()
plt.imshow(feature_map_gauss, cmap="gray")
plt.savefig("example_featuremap_gauss_relu.png")

# now we should switch to the pooling layer
# let up try the max pooling:
# we take (2 x 2) windows and determine the maximum value
# we take a window sliding step width (stripe) of 2 (non-overlapping)
# first, get 2 x 2 matrices of the overall 28 x 28 (namely 196 * (2 x 2)
# btw: np.sqrt(196) # is 14 => (14 x 14) matrix (downsampled)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    A = np.pad(A, padding, mode="constant")
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                     strides = (stride*A.strides[0],
                                stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    if pool_mode=="max":
        return A_w.max(axis=(1,2)).reshape(output_shape)
    if pool_mode=="avg":
        return A_w.mean(axis=(1,2)).reshape(output_shape)

feature_map_downsampled = pool2d(feature_map, kernel_size=2,
                                 stride=2, padding=0, pool_mode="max")
print(feature_map_downsampled)
plt.figure()
plt.imshow(feature_map_downsampled, cmap="gray")
plt.savefig("example_featuremap_map_relu_downsampled.png")

# and the same for our 2nd feature map:
feature_map_gauss_downsampled = pool2d(feature_map_gauss, kernel_size=2,
                                       stride=2, padding=0, pool_mode="max")
plt.figure()
plt.imshow(feature_map_gauss_downsampled, cmap="gray")
plt.savefig("example_featuremap_map_gauss_relu_downsampled.png")

