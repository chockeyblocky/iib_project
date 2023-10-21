"""This will contain operations used to support the layers constructed in layers.py."""

from typing import List, Union
import tensorflow as tf

def rotor_conv1d(
    a_blade_values: tf.Tensor,
    k_blade_values: tf.Tensor,
    cayley: tf.Tensor,
    stride: int,
    padding: str,
    dilations: Union[int, None] = None,
) -> tf.Tensor:
    # A: [..., S, CI, BI]
    # K: [K, CI, CO, BK]
    # C: [BI, BK, BO]

    kernel_size = k_blade_values.shape[0]

    a_batch_shape = tf.shape(a_blade_values)[:-3]

    # Reshape a_blade_values to a 2d image (since that's what the tf op expects)
    # [*, S, 1, CI*BI]
    a_image_shape = tf.concat(
        [
            a_batch_shape,
            tf.shape(a_blade_values)[-3:-2],
            [1, tf.reduce_prod(tf.shape(a_blade_values)[-2:])],
        ],
        axis=0,
    )
    a_image = tf.reshape(a_blade_values, a_image_shape)

    sizes = [1, kernel_size, 1, 1]
    strides = [1, stride, 1, 1]

    # [*, P, 1, K*CI*BI] where eg. number of patches P = S * K for
    # stride=1 and "SAME", (S-K+1) * K for "VALID", ...
    a_slices = tf.image.extract_patches(
        a_image, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding=padding
    )

    # [..., P, K, CI, BI]
    out_shape = tf.concat(
        [
            a_batch_shape,
            tf.shape(a_slices)[-3:-2],
            tf.shape(k_blade_values)[:1],
            tf.shape(a_blade_values)[-2:],
        ],
        axis=0,
    )

    a_slices = tf.reshape(a_slices, out_shape)

    # a_...p,k,ci,bi; k_k,ci,co,bk; c_bi,bk,bo -> y_...p,co,bo
    #   ...a b c  d ,   e c  f  g ,   d  g  h  ->   ...a f  h
    x = tf.einsum("...abcd,bcfg,dgh->...afh", algebra, a_slices, k_blade_values, cayley)

    return x