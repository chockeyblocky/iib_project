"""
This file will test the functioning of the equivariant non-linear layer.
"""

import tensorflow as tf
from layers.layers import *
from tfga import GeometricAlgebra

ga = GeometricAlgebra([1, 1, 1, 1])

model = EquivariantNonLinear(ga, activation='relu')
norm = EquivariantLayerNorm(ga, parameter_initializer='ones')
linear = EquivariantLinear(ga, units=5)

x = tf.constant([[2., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]])

print(model(x))
print(norm(x))

y = norm(x)
print(model(y))

print(linear(x))
