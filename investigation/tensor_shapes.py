import tensorflow as tf

a = tf.constant([[0, 1], [2, 3], [4, 5]])

b = tf.reshape(a, [-1])

print(b)

c = tf.reshape(b, [1, -1, 2])

print(c)

d = tf.reshape(c, [1, -1, 1, 2])

print(d)
