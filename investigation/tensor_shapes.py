import tensorflow as tf
from tfga import GeometricAlgebra

a = tf.ones([1, 4, 4, 1, 2])

k = tf.ones([3, 3, 1, 5, 2])

kernel_size = k.shape[0]

a_batch_shape = tf.shape(a)[:-4]

# Reshape a to a 2d image (since that's what the tf op expects)
# [*, S, 1, CI*BI]
a_image_shape = tf.concat(
    [
        a_batch_shape,
        tf.shape(a)[-4:-3],
        [tf.shape(a)[-3], tf.reduce_prod(tf.shape(a)[-2:])],
    ],
    axis=0,
)

print(a.shape)
a_image = tf.reshape(a, a_image_shape)

print(a_image)

print(a_image_shape)

sizes = [1, kernel_size, kernel_size, 1]
strides = [1, 2, 1, 1]

# [*, P1, P2, K*K*CI*BI] where eg. number of patches P = S * K for
# stride=1 and "SAME", (S-K+1) * K for "VALID", ...
a_slices = tf.image.extract_patches(
    a_image, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="SAME"
)

print(a_slices)
print(a_slices.shape)

# [..., P1, P2, K, K, CI, BI]
out_shape = tf.concat(
    [
        a_batch_shape,
        tf.shape(a_slices)[-3:-1],
        tf.shape(k)[:2],
        tf.shape(a)[-2:]
    ],
    axis=0,
)
print(out_shape)

a_slices = tf.reshape(a_slices, out_shape)
print(a_slices)
print(a_slices.shape)

a = tf.constant([[1, 2], [3, 4]])

from einops import rearrange, repeat

b = rearrange(a, 'a b -> (a b)')
print(b)