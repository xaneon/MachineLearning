import tensorflow as tf
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.log_device_placement=False
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

def matrix_mul(device_name, matrix_sizes):
    time_values = []
    #device_name = "/cpu:0"
    for size in matrix_sizes:
        with tf.device(device_name):
            random_matrix = tf.random_uniform(shape=(2,2), minval=0, maxval=1)
            dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
            sum_operation = tf.reduce_sum(dot_operation)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
            startTime = datetime.now()
            result = session.run(sum_operation)
        td = datetime.now() - startTime
        time_values.append(td.microseconds/1000)
        print ("matrix shape:" + str(size) + "  --"+ device_name +" time: "+str(td.microseconds/1000))
    return time_values


# matrix_sizes = range(100,1000,100)
# matrix_sizes = range(1_000,10_000_000,1_000_000)
matrix_sizes = range(1_000, 10_000, 100)

time_values_gpu = matrix_mul("/gpu:0", matrix_sizes)
time_values_cpu = matrix_mul("/cpu:0", matrix_sizes)
print ("GPU time" +  str(time_values_gpu))
print ("CPU time" + str(time_values_cpu))

plt.figure()
plt.plot(matrix_sizes[:len(time_values_gpu)], time_values_gpu, label='GPU')
plt.plot(matrix_sizes[:len(time_values_cpu)], time_values_cpu, label='CPU')
plt.ylabel('Time (sec)')
plt.xlabel('Size of Matrix ')
plt.legend(loc='best')
plt.savefig("performance_comparison_cpu_vs_gpu_matrix_mutliplication.png")
