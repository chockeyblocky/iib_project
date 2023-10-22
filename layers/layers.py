"""
This contains the custom layers implemented using tfga infrastructure.
"""
from typing import List, Union

import tensorflow as tf
from tensorflow.keras import (activations, constraints, initializers, layers,
                              regularizers)
from tensorflow.keras.utils import register_keras_serializable

from tfga.blades import BladeKind
from tfga.layers import GeometricAlgebraLayer
from tfga.tfga import GeometricAlgebra


class RotorConv1D(GeometricAlgebraLayer):
    """
    This is a convolution layer as described in "Geometric Clifford Algebra
    Networks" (Ruhe et al.). It uses a weighted sandwich product with rotors in the kernel.


     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        filters: How many channels the output will have
        kernel_size: Size for the convolution kernel
        stride: Stride to use for the convolution
        padding: "SAME" (zero-pad input length so output
            length == input length / stride) or "VALID" (no padding)
        blade_indices_kernel: Blade indices to use for the kernel parameter
        blade_indices_bias: Blade indices to use for the bias parameter (if used)
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            filters: int,
            kernel_size: int,
            stride: int,
            padding: str,
            blade_indices_kernel: List[int] = None,
            blade_indices_bias: Union[None, List[int]] = None,
            dilations: Union[None, int] = None,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=tf.keras.constraints.UnitNorm(axis=-1),
            bias_constraint=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilations = dilations

        # if no blade index specified, default to rotors (only even indices)
        if blade_indices_kernel is not None:
            blade_indices_kernel = self.algebra.get_kind_blade_indices('even')

        self.blade_indices_kernel = tf.convert_to_tensor(
            blade_indices_kernel, dtype_hint=tf.int64
        )
        if use_bias:
            self.blade_indices_bias = tf.convert_to_tensor(
                blade_indices_bias, dtype_hint=tf.int64
            )

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape: tf.TensorShape):
        # I: [..., S, C, B]
        self.num_input_filters = input_shape[-2]

        # K: [K, IC, OC, B]
        shape_kernel = [
            self.kernel_size,
            self.num_input_filters,
            self.filters,
            self.blade_indices_kernel.shape[0],
        ]
        self.kernel = self.add_weight(
            "kernel",
            shape=shape_kernel,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_weights = self.add_weight(
            "kernel_weights",
            shape=shape_kernel[:-1],
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            shape_bias = [self.filters, self.blade_indices_bias.shape[0]]
            self.bias = self.add_weight(
                "bias",
                shape=shape_bias,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def rotor_conv1d(
            self,
            a_blade_values: tf.Tensor,
            k_blade_values: tf.Tensor,
            weights,
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
        x = tf.einsum("...bcf,...bcfi,hij,...abcd,bcfg,dgh->...afj", weights, self.algebra.reversion(k_blade_values),
                      self.algebra._cayley, a_slices, k_blade_values, self.algebra._cayley)

        return x

    def call(self, inputs):
        k_geom = self.algebra.from_tensor(self.kernel, self.blade_indices_kernel)

        inputs = tf.convert_to_tensor(inputs, dtype_hint=tf.float32)
        k_geom = tf.convert_to_tensor(k_geom, dtype_hint=tf.float32)
        weights = tf.convert_to_tensor(self.kernel_weights, dtype_hint=tf.float32)

        result = self.rotor_conv1d(
            inputs,
            k_geom,
            weights,
            stride=self.stride,
            padding=self.padding,
            dilations=self.dilations,
        )

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias)
            result += b_geom

        return self.activation(result)
