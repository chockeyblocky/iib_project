"""
This will contain experimentation with the Clifford package.
"""

import clifford as cf
import math
import tensorflow as tf
from tfga import GeometricAlgebra

layout, blades = cf.Cl(p=4, q=1)
locals().update(blades)  # lazy way to put entire basis in the namespace

x = e1 + e2 + e3 + e4 - e5
b = x.dual() * e12345

print(x.dual())
print(x * e12345.inv())

R = math.e**(-math.pi * e14 / 4)
y = R * x * ~R

print(y * ~y)
print(R * e12345 * ~R)  # show invariance of pseudoscalar

# compare with ga - dual works incorrectly
ga = GeometricAlgebra(metric=[1, 1, 1, 1, -1])

x = tf.constant([[[1., 1., 1., 1., -1.], [1, 2, 3, 4, 5], [1, -2, 3, 2, 1]]])
ga_x = ga.from_tensor_with_kind(x, 'vector')
print(ga_x)
i_inv = -ga.blade_mvs[-1]
right_dual_x = ga.geom_prod(ga_x, i_inv)
print(right_dual_x)
print(ga.geom_prod(right_dual_x, ga.blade_mvs[-1]))

wrong_dual_x = ga.dual(ga_x)
print(wrong_dual_x)
