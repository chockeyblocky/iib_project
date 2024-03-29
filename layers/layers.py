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
        method: 'norm' or 'scalar' - choice of which invariant to use
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            activation=None,
            activity_regularizer=None,
            method='norm',
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

        # save method
        self.method = method

    def build(self, input_shape: tf.TensorShape):
        self.built = True

    def call(self, inputs):
        if self.method == 'norm':
            # get grade 0 part of inputs_r * ~inputs_r - CAN USE SCALAR PART FROM INPUT WITHOUT SQUARING
            # can do linear transformation - i.e. a*q + b
            graded_inputs = tf.einsum("...j,ij->...ij", inputs, self.input_grades)
            quad_form = self.algebra.geom_prod(graded_inputs, self.algebra.reversion(graded_inputs))[..., 0]

            # repeat grade r parts as required for each basis in that grade
            quad_form_repeated = tf.repeat(quad_form, repeats=self.grade_numbers, axis=-1)

            # apply non-linearity then multiply by inputs and return
            return self.activation(quad_form_repeated) * inputs

        # if scalar method was selected, apply non-linearity to scalar part of multivector
        return tf.einsum('...i,...ij->...ij', self.activation(inputs[..., 0]), inputs)


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
class EquivariantStableLayerNorm(GeometricAlgebraLayer):
    """
    This layer uses an equivariant mapping scheme to normalise multivectors by dividing all elements by the
    quadratic form.


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
        # get norms from quadratic form - for stable layer norm, take geometric product over entire mv
        quad_form = self.algebra.geom_prod(inputs, self.algebra.reversion(inputs))[..., 0]
        norm = tf.math.sqrt(tf.math.abs(quad_form + 1.0e-12))  # added constant for stability in gradient

        # apply sigmoid
        s_a = tf.sigmoid(self.parameter)

        # interpolate between 1 and the norm
        interpolated_norm = tf.einsum("...ij,...i->...ij", s_a, (norm - 1)) + 1

        # repeat for division
        interpolated_norm_repeated = tf.repeat(interpolated_norm, repeats=self.grade_numbers, axis=-1)
        return inputs / (interpolated_norm_repeated + 1e-12)


@keras.saving.register_keras_serializable(package="EquiLayers")
class EquivariantMeanLayerNorm(GeometricAlgebraLayer):
    """
    This layer uses an equivariant mapping scheme to normalise multivectors by dividing all elements by the average of
    abs(x * ~x) over the layer dimension. Assumes tensor input with shape (batch, num_mv, mv_dim).


     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        activation: Activation function to use
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            activity_regularizer=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )

    def build(self, input_shape: tf.TensorShape):
        self.built = True

    def call(self, inputs):
        # get norms from quadratic form - for stable layer norm, take geometric product over entire mv
        quad_form = self.algebra.geom_prod(inputs, self.algebra.reversion(inputs))[..., 0]
        norm = tf.math.reduce_mean(tf.math.sqrt(tf.math.abs(quad_form) + 1.0e-12), axis=-1)  # added
        # constant for stability in gradient

        # reshape for broadcasted division - assuming input shape (batch, num_mvs, mv_dim)
        reshaped_norm = tf.reshape(norm, [-1, 1, 1])

        # divide inputs by corresponding batches' layer norm mean
        # return tf.einsum("i...,i->i...", inputs, 1 / (norm + 1e-12))
        return inputs / (reshaped_norm + 1.0e-12)


@keras.saving.register_keras_serializable(package="EquiLayers")
class EquivariantLinear(GeometricAlgebraLayer):
    """
    This is an equivariant linear layer as described in "Clifford Group Equivariant Neural Networks" (Ruhe et al.).
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
            res += b_geom

        return res


@keras.saving.register_keras_serializable(package="EquiLayers")
class EquivariantMultiLinear(GeometricAlgebraLayer):
    """
    This is a modified equivariant linear layer as described in "Clifford Group Equivariant Neural Networks"
    (Ruhe et al.), designed to work with mutliple channels.
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
        self.num_channels = input_shape[-3]
        shape_kernel = [
            self.num_channels,
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
        if self.use_bias:
            shape_bias = [self.num_channels, self.units, 1]
            self.bias = self.add_weight(
                "bias",
                shape=shape_bias,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        self.built = True

    def call(self, inputs):
        # repeat grade r parts as required for each basis in that grade
        kernel_repeated = tf.repeat(self.kernel, repeats=self.grade_numbers, axis=-1)

        # multiply kernel with inputs and sum (channels, units, inputs, ga_dim)
        res = tf.einsum("hijk,...hjk->...hik", kernel_repeated, inputs)

        # add scalar bias if required
        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias)
            res += b_geom
        return res


@keras.saving.register_keras_serializable(package="EquiLayers")
class EquivariantAttention(GeometricAlgebraLayer):
    """
    This is an equivariant attention layer as described in "Geometric Algebra Transformer" (Brehmer et al.).
    It uses the inner product between multivectors as an attention mechanism.

     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        key_dimension: dimension of key (in terms of non-zero values in the multivector) - if left at None, defaults to
        multivector length
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            key_dimension=None,
            activity_regularizer=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )
        # assign key dimension
        if not key_dimension:
            self.k_dim = algebra.num_blades
        else:
            self.k_dim = key_dimension

        # square root key dimension for dividing
        self.dividing_constant = self.k_dim ** 0.5

    def build(self, input_shape: tf.TensorShape):
        self.built = True

    def call(self, queries, keys, values):
        # compute inner product between queries and keys and take scalar part
        inner_prod = self.algebra.inner_prod(queries, keys)[..., 0]

        # apply non-linearity (softmax) and divide by constant
        nl_inner_prod = tf.nn.softmax(inner_prod / self.dividing_constant)

        # multiply nl inner prod with values and return
        return tf.einsum("...i,...ij->...ij", nl_inner_prod, values)


@keras.saving.register_keras_serializable(package="EquiLayers")
class EquivariantSelfAttention(GeometricAlgebraLayer):
    """
    This is an equivariant attention layer as described in "Geometric Algebra Transformer" (Brehmer et al.).
    It uses the inner product between multivectors as an attention mechanism.

     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        units_per_head: number of linear units to use in each attention layer (i.e. number of multivectors in each head)
        output_units: number of output units to be used
        key_dimension: dimension of key (in terms of non-zero values in the multivector) - if left at None,
        defaults to multivector length
        heads: number of heads in attention layer
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            units_per_head,
            output_units,
            heads=1,
            key_dimension=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            activity_regularizer=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )

        # define attention layer to be used
        self.attention = EquivariantAttention(algebra, key_dimension, activity_regularizer)

        # define multi linear layers for query, key, and value projection
        self.query_multi_linear = EquivariantMultiLinear(algebra, units=units_per_head, use_bias=use_bias,
                                                         kernel_initializer=kernel_initializer,
                                                         bias_initializer=bias_initializer,
                                                         kernel_regularizer=kernel_regularizer,
                                                         bias_regularizer=bias_regularizer,
                                                         activity_regularizer=activity_regularizer,
                                                         kernel_constraint=kernel_constraint,
                                                         bias_constraint=bias_constraint)

        self.key_multi_linear = EquivariantMultiLinear(algebra, units=units_per_head, use_bias=use_bias,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer,
                                                       kernel_regularizer=kernel_regularizer,
                                                       bias_regularizer=bias_regularizer,
                                                       activity_regularizer=activity_regularizer,
                                                       kernel_constraint=kernel_constraint,
                                                       bias_constraint=bias_constraint)

        self.value_multi_linear = EquivariantMultiLinear(algebra, units=units_per_head, use_bias=use_bias,
                                                         kernel_initializer=kernel_initializer,
                                                         bias_initializer=bias_initializer,
                                                         kernel_regularizer=kernel_regularizer,
                                                         bias_regularizer=bias_regularizer,
                                                         activity_regularizer=activity_regularizer,
                                                         kernel_constraint=kernel_constraint,
                                                         bias_constraint=bias_constraint)

        # define final linear layer
        self.output_linear = EquivariantLinear(algebra, output_units)

        # store number of heads and number of units per head
        self.units_per_head = units_per_head
        self.heads = heads

    def build(self, input_shape: tf.TensorShape):
        self.pre_linear_output_shape = tf.concat([[-1], [self.heads * self.units_per_head], [input_shape[-1]]], axis=0)
        self.built = True

    def call(self, inputs):
        # repeat inputs to expand into number of heads
        inputs_repeated = tf.repeat(tf.expand_dims(inputs, -3), self.heads, axis=-3)

        # use multi-linear layer
        queries = self.query_multi_linear(inputs_repeated)
        keys = self.key_multi_linear(inputs_repeated)
        values = self.value_multi_linear(inputs_repeated)

        # call attention layer on queries, keys and values
        attention_heads = self.attention(queries, keys, values)

        # reshape from (batch, heads, units_per_head, ga_dimension) to (batch, heads x units_per_head, ga_dimension)
        concatenated_heads = tf.reshape(attention_heads, shape=self.pre_linear_output_shape)

        # apply linear layer and return output
        return self.output_linear(concatenated_heads)


class EquivariantGP(GeometricAlgebraLayer):
    """
    This is an equivariant geometric product layer. It takes the multivector inputs, projects them using equilinear
    layers, then takes the geometric product of the result and outputs the result

     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        units: number of linear units to use in the equilinear layer (and number of output multivectors)
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
        # define linear layers
        self.linear1 = EquivariantLinear(algebra, units, use_bias, kernel_initializer, bias_initializer,
                                         kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint,
                                         bias_constraint)
        self.linear2 = EquivariantLinear(algebra, units, use_bias, kernel_initializer, bias_initializer,
                                         kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint,
                                         bias_constraint)

    def build(self, input_shape: tf.TensorShape):
        self.built = True

    def call(self, inputs):
        # apply linear layers to inputs
        x = self.linear1(inputs)
        y = self.linear2(inputs)

        # compute geometric product between x and y and return
        return self.algebra.geom_prod(x, y)


class EquivariantJoin(GeometricAlgebraLayer):
    """
    This is an equivariant join layer. It takes the multivector inputs, projects them using equilinear
    layers, then takes the join (or meet) of the result and outputs the result.

     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        units: number of linear units to use in the equilinear layer (and number of output multivectors)
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
        # define linear layers
        self.linear1 = EquivariantLinear(algebra, units, use_bias, kernel_initializer, bias_initializer,
                                         kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint,
                                         bias_constraint)
        self.linear2 = EquivariantLinear(algebra, units, use_bias, kernel_initializer, bias_initializer,
                                         kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint,
                                         bias_constraint)

        # get pseudoscalar and its inverse for applying joins
        self.pseudoscalar = self.algebra.blade_mvs[-1]
        self.pseudoscalar_inv = self.algebra.inverse(self.algebra.blade_mvs[-1])

    def build(self, input_shape: tf.TensorShape):
        self.built = True

    def call(self, inputs):
        # apply linear layers to inputs
        x = self.linear1(inputs)
        y = self.linear2(inputs)

        # compute join between x and y and return - join is (x I_inv ^ y I_inv)I - tfga does not correctly implement
        # this, necessitating a more complex approach
        return self.algebra.geom_prod(self.algebra.ext_prod(self.algebra.geom_prod(x, self.pseudoscalar_inv),
                                      self.algebra.geom_prod(y, self.pseudoscalar_inv)), self.pseudoscalar)


class EquivariantTransformerBlock(GeometricAlgebraLayer):
    """
    This is an Equivariant Transformer Block designed to use CGA to solve geometric prediction problems. It assumes that
    the desired output shape of the network is equal to the input shape so that residual connections can be used

     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        units_per_head: number of linear units to use in each attention layer (i.e. number of multivectors in each head)
        output_units: output (and input) number of multivectors
        hidden_units: number of multivectors to use in hidden parts of transformer (i.e. after attention and during
        feed-forward network)
        output_units: desired output shape
        key_dimension: dimension of key (in terms of non-zero values in the multivector) - if left at None,
        defaults to multivector length
        heads: number of heads in attention layer
        non_linear_activation: activation function to use in non-linear part of feed-forward network
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            units_per_head,
            hidden_units,
            output_units,
            heads=1,
            non_linear_activation='sigmoid',
            key_dimension=None,
            activity_regularizer=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )
        self.num_multivectors = output_units

        # layer norm before attn block
        self.layer_norm1 = EquivariantMeanLayerNorm(algebra)
        # initial linear layer - geometric product allows mixing before attention layer
        self.gp_linear1 = EquivariantGP(algebra, hidden_units)
        # multi head attention - includes linear layer at the end of attention block
        self.multi_head_attention = EquivariantSelfAttention(algebra, units_per_head, output_units, heads,
                                                             key_dimension=key_dimension)
        # linear layer before residual connection
        self.linear1 = EquivariantLinear(algebra, units=output_units)

        # layer norm (beginning of feed-forward block) - stable layer norm used to avoid /0
        self.layer_norm2 = EquivariantMeanLayerNorm(algebra)
        # geometric product layer
        self.gp_linear2 = EquivariantGP(algebra, hidden_units // 2)
        # join layer
        self.join = EquivariantJoin(algebra, hidden_units // 2)
        # non-linear layer (using scalar part for stability)
        self.non_linear = EquivariantNonLinear(algebra, activation=non_linear_activation, method='scalar')
        # linear layer (end of transformer, return to original shape) - TODO try using rotor convolution layer
        self.linear2 = EquivariantLinear(algebra, output_units)

    def build(self, input_shape: tf.TensorShape):
        if self.num_multivectors != input_shape[-2]:
            raise Exception('Input shape not equal to output_units')
        self.built = True

    def call(self, inputs):
        # initial attention block + residual connection
        residual_1 = inputs
        x = self.layer_norm1(residual_1)
        x = self.gp_linear1(x)
        x = self.multi_head_attention(x)
        x = self.linear1(x)
        residual_2 = x + residual_1

        # feed-forward block + residual connection
        x = self.layer_norm2(residual_2)
        x = tf.concat([self.gp_linear2(x), self.join(x)], axis=-2)
        x = self.non_linear(x)
        x = self.linear2(x)

        return x + residual_2


class ExperimentalEquivariantTransformerBlock(GeometricAlgebraLayer):
    """
    This is an Equivariant Transformer Block designed to use CGA to solve geometric prediction problems. It assumes that
    the desired output shape of the network is equal to the input shape so that residual connections can be used

     Args:
        algebra: GeometricAlgebra instance to use for the parameters
        units_per_head: number of linear units to use in each attention layer (i.e. number of multivectors in each head)
        output_units: output (and input) number of multivectors
        hidden_units: number of multivectors to use in hidden parts of transformer (i.e. after attention and during
        feed-forward network)
        output_units: desired output shape
        key_dimension: dimension of key (in terms of non-zero values in the multivector) - if left at None,
        defaults to multivector length
        heads: number of heads in attention layer
        non_linear_activation: activation function to use in non-linear part of feed-forward network
    """

    def __init__(
            self,
            algebra: GeometricAlgebra,
            units_per_head,
            hidden_units,
            output_units,
            heads=1,
            non_linear_activation='sigmoid',
            key_dimension=None,
            activity_regularizer=None,
            **kwargs
    ):
        super().__init__(
            algebra=algebra, activity_regularizer=activity_regularizer, **kwargs
        )
        self.num_multivectors = output_units

        # layer norm before attn block
        self.layer_norm1 = EquivariantMeanLayerNorm(algebra)
        # initial linear layer - geometric product allows mixing before attention layer
        self.gp_linear1 = EquivariantGP(algebra, hidden_units)
        # multi head attention - includes linear layer at the end of attention block
        self.multi_head_attention = EquivariantSelfAttention(algebra, units_per_head, output_units, heads,
                                                             key_dimension=key_dimension)
        # linear layer before residual connection
        self.linear1 = EquivariantLinear(algebra, units=output_units)

        # layer norm (beginning of feed-forward block) - stable layer norm used to avoid /0
        self.layer_norm2 = EquivariantMeanLayerNorm(algebra)
        # geometric product layer
        self.gp_linear2 = EquivariantGP(algebra, hidden_units // 2)
        # join layer
        self.join = EquivariantJoin(algebra, hidden_units // 2)
        # non-linear layer (using scalar part for stability)
        self.non_linear = EquivariantNonLinear(algebra, activation=non_linear_activation, method='scalar')
        # linear layer (end of transformer, return to original shape) - TODO try using rotor convolution layer
        self.linear2 = EquivariantLinear(algebra, output_units)

    def build(self, input_shape: tf.TensorShape):
        if self.num_multivectors != input_shape[-2]:
            raise Exception('Input shape not equal to output_units')
        self.built = True

    def call(self, inputs):
        # initial attention block + residual connection
        residual_1 = inputs
        x = self.layer_norm1(residual_1)  # ADDED LAYERNORM - EXPERIMENT WITH STABILITY
        x = self.gp_linear1(x)
        x = self.multi_head_attention(x)
        x = self.linear1(x)
        residual_2 = x + residual_1

        # feed-forward block + residual connection
        x = self.layer_norm2(residual_2)
        x = tf.concat([self.gp_linear2(x), self.join(x)], axis=-2)
        x = self.non_linear(x)
        x = self.linear2(x)

        return x + residual_2
