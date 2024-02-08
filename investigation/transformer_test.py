"""This will test the functionality of transformer layers.
"""

import tensorflow as tf
import numpy as np
from layers.layers import EquivariantAttention, EquivariantSelfAttention, EquivariantTransformerBlock
from tfga import GeometricAlgebra

ga = GeometricAlgebra(metric=[1, 1])

attn = EquivariantAttention(ga)


q = tf.constant([[[1., 2., 3., 0.], [7., 6., 4., 0.]], [[1., 2., 3., 0.], [7., 6., 4., 0.]]])
k = tf.constant([[[1., 2., 3., 4.], [7., 6., 4., 2.]], [[1., 2., 3., 4.], [7., 6., 4., 2.]]])
v = tf.constant([[[1., 2., 3., 4.], [7., 6., 4., 2.]], [[1., 2., 3., 4.], [7., 6., 4., 2.]]])

y = attn(q, k, v)
print(y)
print(y.shape)

x = tf.repeat(tf.expand_dims(y, -3), 12, axis=-3)
print(x.shape)

z = tf.reshape(x, shape=[-1, 24, 4])
print(z.shape)

selfattn = EquivariantSelfAttention(ga, 5, 2, heads=10)

a = selfattn(q)

print(a)
print(a.shape)
q2 = tf.constant([[[1., 2., 3., 4.], [7., 6., 4., 2.]]])

b = selfattn(q2)
print(b.shape)

transformer = EquivariantTransformerBlock(ga, 10, 4, 2, heads=3)

y = transformer(q)

print(y)