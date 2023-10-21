import tensorflow as tf
from tfga import GeometricAlgebra

a = tf.constant([[0., 1.], [2., 3.], [4., 5.], [6., 7.]])

b = tf.reshape(a, [-1])

print(b)

c = tf.reshape(b, [1, -1, 2])

print(c)

d = tf.reshape(c, [1, -1, 1, 2])

print(d)

kernel = tf.constant([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.])

kernel = tf.reshape(kernel, [2, 1, 4, 2])

print(kernel)

image_shape = tf.concat([tf.shape(d)[:-3], tf.shape(d)[-3:-2], [1, tf.reduce_prod(tf.shape(d)[-2:])]], axis=0)

print(image_shape)

a_image = tf.reshape(d, image_shape)

sizes = [1, kernel.shape[0], 1, 1]
strides = [1, 1, 1, 1]

# [*, P, 1, K*CI*BI] where eg. number of patches P = S * K for
# stride=1 and "SAME", (S-K+1) * K for "VALID", ...
a_slices = tf.image.extract_patches(
    a_image, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="SAME"
)

print(a_slices)

out_shape = tf.concat(
    [
        tf.shape(d)[:-3],
        tf.shape(a_slices)[-3:-2],
        tf.shape(kernel)[:1],
        tf.shape(d)[-2:],
    ],
    axis=0,
)

a_slices = tf.reshape(a_slices, out_shape)

a_slices = tf.convert_to_tensor(a_slices, dtype_hint=tf.float32)
kernel = tf.convert_to_tensor(kernel, dtype_hint=tf.float32)

print(a_slices)
print(kernel)

ga = GeometricAlgebra(metric=[1])

a_slices = ga(a_slices)

print(a_slices.tensor)