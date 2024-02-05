"""
This contains a test of the equivariant geometric product layer.
"""

import tensorflow as tf
import numpy as np
from layers.layers import EquivariantGP
from tfga import GeometricAlgebra

ga = GeometricAlgebra(metric=[1, 1])

model = EquivariantGP(ga, units=10)

x = tf.constant([[[1., 2., 3., 0.], [7., 6., 4., 0.]], [[1., 2., 3., 0.], [7., 6., 4., 0.]]])

y = model(x)
print(y)
print(y.shape)
