import tensorflow as tf
from pprint import pprint
import numpy as np

graph1 = tf.Graph()

with graph1.as_default():
    a = tf.constant([2], name='constant_a')
    b = tf.constant([3], name='constant_b')

print(a, b)

# Printing the value of a
session = tf.Session(graph=graph1)
result = session.run(a)
session.close()
print(result)

# Now let us have a look at the add() function:
with graph1.as_default():
    c = tf.add(a, b)
    d = a + b
print(c, b)

session = tf.Session(graph=graph1)
result = session.run(c)
print(result)
session.close()

# this is better within a context:
with tf.Session(graph=graph1) as session:
    result = session.run(d)
print(result)

# let us start with lists:
dim0 = 2
dim1 = [5, 6, 2]
dim2 = [[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]]
dim3 = [[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
        [[4, 5, 6], [5, 6, 7], [6, 7, 8]],
        [[7, 8, 9], [8, 9, 10], [9, 10, 11]]]
# pprint(dim2); pprint(dim3)

# Comparison to numpy
np_dim0 = np.array(dim0)
np_dim1 = np.array(dim1)
np_dim2 = np.array(dim2)
np_dim3 = np.array(dim3)
print(np_dim2, np_dim2.dtype, np_dim2.shape)
print(np_dim3, np_dim3.dtype, np_dim2.shape)


# Defining multidimensional arrays with tensorflow:
graph2 = tf.Graph()
with graph2.as_default():  # defining constants
    Scalar = tf.constant(dim0)
    Vector = tf.constant(dim1)
    Matrix = tf.constant(dim2)
    Tensor = tf.constant(dim3)

with tf.Session(graph=graph2) as session:
    result = session.run(Scalar)
    print("Scalar: ", result)
    result = session.run(Vector)
    print("Vector: ", result)
    result = session.run(Matrix)
    print("Matrix: ", result)
    result = session.run(Tensor)
    print("Tensor: ", result, type(result))

print(Scalar.shape, Vector.shape, Matrix.shape, Tensor.shape)

# let us try the add() function again
graph3 = tf.Graph()

M1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
M2 = [[3, 2, 3], [6, 5, 6], [9, 8, 9]]

with graph3.as_default():
    Matrix_A = tf.constant(M1)
    Matrix_B = tf.constant(M2)
    add1 = tf.add(Matrix_A, Matrix_B)
    add2 = Matrix_A + Matrix_B

with tf.Session(graph=graph3) as session:
    r1 = session.run(add1)
    r2 = session.run(add2)
    print(r1, r2, sep="\n")
# same as in numpy
print(np.array(M1) + np.array(M2))

graph4 = tf.Graph()

# element-wise multiplication, also known as Hadamard product:
M1 = [[1, 2], [3, 4]]
M2 = [[5, 6], [7, 8]]

with graph4.as_default():
    MM1 = tf.constant(M1)
    MM2 = tf.constant(M2)
    m1 = tf.matmul(MM1, MM2)  # matrix-multiplication
    m2 = MM1 * MM2  # element-wise multiplication

with tf.Session(graph=graph4) as session:
    r1 = session.run(m1)
    r2 = session.run(m2)
    print(r1, r2, sep="\n")

# use of variables
v = tf.Variable(0)
print(v)

# update step
update = tf.assign(v, v+1)
print(update)

# initialise the operation:
init_operation = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_operation)
    print(session.run(v))
    for _ in range(3):
        session.run(update)
        print(session.run(v))

# placeholders:
a = tf.placeholder(tf.float32)
b = a * 2
print(b)

# let us now execute this:
with tf.Session() as session:
    result = session.run(b, feed_dict={a: 2.5})
    result2 = session.run(b, feed_dict={a: M1})
    print(result, result2)

# other functions:
graph5 = tf.Graph()
with graph5.as_default():
    a = tf.constant([5.5])
    b = tf.constant([6.3])
    add = tf.add(a, b)
    multi = a * b
    subtr = tf.subtract(a, b)
    sigm = tf.nn.sigmoid(a)

with tf.Session(graph=graph5) as session:
    r1 = session.run(add)
    r2 = session.run(multi)
    r3 = session.run(subtr)
    r4 = session.run(sigm)
    print(r1, r2, r3, r4)


