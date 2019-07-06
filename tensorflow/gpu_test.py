import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.log_device_placement=False
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
