"""
This contains the custom layers implemented using tfga infrastructure.
"""
from typing import List, Union

import tensorflow as tf
import keras
from tensorflow.keras import (activations, constraints, initializers, layers,
                              regularizers)
from tensorflow.keras.utils import register_keras_serializable

from tfga.blades import BladeKind
from tfga.layers import GeometricAlgebraLayer
from tfga.tfga import GeometricAlgebra

# TODO: fix serialisation when tfga fixes it


@keras.saving.register_keras_serializable(package="EquiLayers")
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
            if blade_indices_bias is None:
                blade_indices_bias = self.algebra.get_kind_blade_indices('scalar')
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
        x = tf.einsum("...bcf,...bcfi,hij,...abcd,bcfg,gdh->...afj", weights, self.algebra.reversion(k_blade_values),
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "blade_indices_kernel": self.blade_indices_kernel.numpy(),
                "blade_indices_bias": self.blade_indices_bias.numpy(),
                "dilations": self.dilations,
                "padding": self.padding,
                "stride": self.stride,
                "filters": self.filters,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config


@keras.saving.register_keras_serializable(package="EquiLayers")
class RotorConv2D(GeometricAlgebraLayer):
    """
    This is a  2D convolution layer as described in "Geometric Clifford Algebra
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
            stride_horizontal: int,
            stride_vertical: int,
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
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding
        self.dilations = dilations

        # if no blade index specified, default to rotors (only even indices)
        if blade_indices_kernel is None:
            blade_indices_kernel = self.algebra.get_kind_blade_indices('even')

        self.blade_indices_kernel = tf.convert_to_tensor(
            blade_indices_kernel, dtype_hint=tf.int64
        )
        if use_bias:
            if blade_indices_bias is None:
                blade_indices_bias = self.algebra.get_kind_blade_indices('scalar')
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
        # I: [..., S, S, C, B]
        self.num_input_filters = input_shape[-2]

        # K: [K, K, IC, OC, B]
        shape_kernel = [
            self.kernel_size,
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

    def rotor_conv2d(
            self,
            a_blade_values: tf.Tensor,
            k_blade_values: tf.Tensor,
            weights,
            stride_horizontal: int,
            stride_vertical: int,
            padding: str,
            dilations: Union[int, None] = None,
    ) -> tf.Tensor:
        # A: [..., S, S, CI, BI]
        # K: [K, K, CI, CO, BK]
        # C: [BI, BK, BO]

        kernel_size = k_blade_values.shape[0]

        a_batch_shape = tf.shape(a_blade_values)[:-4]

        # Reshape a_blade_values to a 2d image (since that's what the tf op expects)
        # [*, S, S, CI*BI]
        a_image_shape = tf.concat(
            [
                a_batch_shape,
                tf.shape(a_blade_values)[-4:-3],
                [tf.shape(a_blade_values)[-3], tf.reduce_prod(tf.shape(a_blade_values)[-2:])],
            ],
            axis=0,
        )
        a_image = tf.reshape(a_blade_values, a_image_shape)

        sizes = [1, kernel_size, kernel_size, 1]
        strides = [1, stride_vertical, stride_horizontal, 1]

        # [*, P1, P2, K*K*CI*BI] where eg. number of patches P = S * K for
        # stride=1 and "SAME", (S-K+1) * K for "VALID", ...
        a_slices = tf.image.extract_patches(
            a_image, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding=padding
        )

        # [..., P1, P2, K, K, CI, BI]
        out_shape = tf.concat(
            [
                a_batch_shape,
                tf.shape(a_slices)[-3:-1],
                tf.shape(k_blade_values)[:2],
                tf.shape(a_blade_values)[-2:],
            ],
            axis=0,
        )

        a_slices = tf.reshape(a_slices, out_shape)

        # no-sandwich product convolution:
        # a_...p,p,k,k,ci,bi; k,k,ci,co,bk; c_bi,bk,bo -> y_...p,p,co,bo
        #   ...a n b m c  d , b m c  f  g ,   d  g  h  ->   ...a n f  h

        # sandwich product adds additional cayley matrix, otherwise dimensions correspond; thus just need to add extra
        # dimension from 1d case to all kernel elements to maintain correspondence
        x = tf.einsum("...bmcf,...bmcfi,hij,...anbmcd,bmcfg,gdh->...anfj", weights,
                      self.algebra.reversion(k_blade_values), self.algebra._cayley, a_slices, k_blade_values,
                      self.algebra._cayley)

        return x

    def call(self, inputs):
        k_geom = self.algebra.from_tensor(self.kernel, self.blade_indices_kernel)

        inputs = tf.convert_to_tensor(inputs, dtype_hint=tf.float32)
        k_geom = tf.convert_to_tensor(k_geom, dtype_hint=tf.float32)
        weights = tf.convert_to_tensor(self.kernel_weights, dtype_hint=tf.float32)

        result = self.rotor_conv2d(
            inputs,
            k_geom,
            weights,
            stride_horizontal=self.stride_horizontal,
            stride_vertical=self.stride_vertical,
            padding=self.padding,
            dilations=self.dilations,
        )

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias)
            result += b_geom

        return self.activation(result)


@keras.saving.register_keras_serializable(package="EquiLayers")
class EquivariantNonLinear(GeometricAlgebraLayer):
    """
    This is an equivariant multivector layer as described in "Clifford Group Equivariant Neural Networks" (Ruhe et al.).
    It uses an equivariant mapping scheme to apply non-linearities to multivectors.


     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        activation: Activation function to use
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            activation=None,
            activity_regularizer=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )
        # separate blade indices of each grade
        ones = tf.ones(self.algebra.num_blades)
        self.input_grades = tf.stack([self.algebra.keep_blades(ones, self.algebra.get_blade_indices_of_degree(i))
                                      for i in range(self.algebra.max_degree + 1)])

        # get number of bases of each grade
        self.grade_numbers = [len(self.algebra.get_blade_indices_of_degree(i))
                              for i in range(self.algebra.max_degree + 1)]

        # define activation
        self.activation = activations.get(activation)

    def build(self, input_shape: tf.TensorShape):
        self.built = True

    def call(self, inputs):
        # get grade 0 part of inputs_r * ~inputs_r - CAN USE SCALAR PART FROM INPUT WITHOUT SQUARING
        # can do linear transformation - i.e. a*q + b
        graded_inputs = tf.einsum("...j,ij->...ij", inputs, self.input_grades)
        quad_form = self.algebra.geom_prod(graded_inputs, self.algebra.reversion(graded_inputs))[..., 0]

        # repeat grade r parts as required for each basis in that grade
        quad_form_repeated = tf.repeat(quad_form, repeats=self.grade_numbers, axis=-1)

        # apply non-linearity then multiply by inputs and return
        return self.activation(quad_form_repeated) * inputs


@keras.saving.register_keras_serializable(package="EquiLayers")
class EquivariantLayerNorm(GeometricAlgebraLayer):
    """
    This is an equivariant multivector layer as described in "Clifford Group Equivariant Neural Networks" (Ruhe et al.).
    It uses an equivariant mapping scheme to normalise multivectors while preserving grade relative magnitude
    information.


     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        activation: Activation function to use
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            parameter_initializer="ones",
            parameter_regularizer=None,
            parameter_constraint=None,
            activity_regularizer=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )
        # define parameter initialisation
        self.parameter_initializer = initializers.get(parameter_initializer)
        self.parameter_regularizer = regularizers.get(parameter_regularizer)
        self.parameter_constraint = constraints.get(parameter_constraint)
        ones = tf.ones(self.algebra.num_blades)

        # separate blade indices of each grade
        self.input_grades = tf.stack([self.algebra.keep_blades(ones, self.algebra.get_blade_indices_of_degree(i))
                                      for i in range(self.algebra.max_degree + 1)])

        # get number of bases of each grade
        self.grade_numbers = [len(self.algebra.get_blade_indices_of_degree(i))
                              for i in range(self.algebra.max_degree + 1)]


    def build(self, input_shape: tf.TensorShape):
        # initialise normalisation parameter assuming first dimension is batch dim
        shape_parameter = input_shape[1:-1].as_list()
        shape_parameter.append(self.algebra.max_degree.numpy() + 1)
        self.parameter = self.add_weight(
            "parameter",
            shape=shape_parameter,
            initializer=self.parameter_initializer,
            regularizer=self.parameter_regularizer,
            constraint=self.parameter_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        # get norms from quadratic form
        graded_inputs = tf.einsum("...j,ij->...ij", inputs, self.input_grades)
        quad_form = self.algebra.geom_prod(graded_inputs, self.algebra.reversion(graded_inputs))[..., 0]
        norm = tf.math.sqrt(quad_form + 1.0e-12)  # added constant for stability in gradient

        # apply sigmoid
        s_a = tf.sigmoid(self.parameter)

        # interpolate between 1 and the norm
        interpolated_norm = s_a * (norm - 1) + 1

        # repeat for division
        interpolated_norm_repeated = tf.repeat(interpolated_norm, repeats=self.grade_numbers, axis=-1)
        return inputs / (interpolated_norm_repeated + 1e-12)


@keras.saving.register_keras_serializable(package="EquiLayers")
class EquivariantLinear(GeometricAlgebraLayer):
    """
    This is an equivariant multivector layer as described in "Clifford Group Equivariant Neural Networks" (Ruhe et al.).
    It does a weighted sum of grades of input multivectors.


     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        activation: Activation function to use
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            units: int,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )
        # separate blade indices of each grade
        ones = tf.ones(self.algebra.num_blades)
        self.input_grades = tf.stack([self.algebra.keep_blades(ones, self.algebra.get_blade_indices_of_degree(i))
                                      for i in range(self.algebra.max_degree + 1)])

        # get number of bases of each grade
        self.grade_numbers = [len(self.algebra.get_blade_indices_of_degree(i))
                              for i in range(self.algebra.max_degree + 1)]

        # use scalar bias only for equivariance
        if use_bias:
            blade_indices_bias = self.algebra.get_kind_blade_indices('scalar')
            self.blade_indices_bias = tf.convert_to_tensor(
                blade_indices_bias, dtype_hint=tf.int64
            )

        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape: tf.TensorShape):
        self.num_input_units = input_shape[-2]
        shape_kernel = [
            self.units,
            self.num_input_units,
            self.algebra.max_degree + 1,
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
        self.built = True
        if self.use_bias:
            shape_bias = [self.units, 1]
            self.bias = self.add_weight(
                "bias",
                shape=shape_bias,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

    def call(self, inputs):
        # repeat grade r parts as required for each basis in that grade
        kernel_repeated = tf.repeat(self.kernel, repeats=self.grade_numbers, axis=-1)

        # multiply kernel with inputs and sum
        res = tf.einsum("ijk,...jk->...ik", kernel_repeated, inputs)

        # add scalar bias if required
        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias)
            res+= b_geom

        return res