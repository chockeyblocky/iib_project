"""
This file is used for testing custom rotor convolution layer.
"""

from keras.models import Model
from keras.layers import Input, Reshape, Flatten, GlobalAveragePooling2D, \
    Dense, Dropout, BatchNormalization
from keras import regularizers
import tensorflow as tf
import keras
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from tfga import GeometricAlgebra
from tfga.layers import TensorToGeometric, GeometricProductConv1D, GeometricToTensor, GeometricSandwichProductDense
from layers.layers import RotorConv1D

ga = GeometricAlgebra(metric=[1, 1, 1, 1])

# define network

idx = ga.get_kind_blade_indices("even")  # gets a mask of indices for the
# given ga which satisfy the blade kind

# defining base model
# input tensor for RGB (I think) so has additional 3 dimensions
model = InceptionV3(classifier_activation=None, weights="imagenet",
                    input_tensor=Input(shape=(224, 224, 3)))

x2 = Dropout(0.3)(model.layers[-2].output)
x2 = Reshape((-1, 1, 8))(x2)
x2 = TensorToGeometric(ga, blade_indices=idx)(x2)

x2 = RotorConv1D(
    ga, filters=2, kernel_size=8, stride=1, padding='SAME', blade_indices_kernel=idx, blade_indices_bias=idx,
    activation='relu'
)(x2)

x2 = GeometricSandwichProductDense(
    ga, units=1, activation="tanh",
    blade_indices_kernel=idx,
    blade_indices_bias=idx)(x2)
x2 = GeometricToTensor(ga, blade_indices=idx)(x2)
outputs2 = Flatten()(x2)

Model1 = tf.keras.Model(inputs=model.input, outputs=outputs2)
Model1.summary()
CGAPoseNet = Model1
