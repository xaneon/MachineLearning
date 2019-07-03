import tensorflow as tf

# print(dir(tf))

a = tf.constant([2])
b = tf.constant([3])
c = tf.constant([4.5])
d = tf.constant(5.5)
e = tf.constant([1,2])
f = tf.constant([3,0])

print(a, b)
print(type(a), type(b))
print(c, d)
print(type(c), type(d))
print(e, f)
print(type(e), type(f))

g = tf.add(a, b)
g2 = a + b
print(g, g2)

h = tf.add(e, f)
h2 = e + f
print(h, h2)

# session = tf.Session()
session = tf.compat.v1.Session()
result = session.run(g) # return numpy array
print(result, type(result))
result2 = session.run(h2)
print(result2, type(result2))

session.close() # close session

with tf.compat.v1.Session() as session:
    result42 = session.run(e * f)

print(result42)



