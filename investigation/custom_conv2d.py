"""
This will test the 2D convolution layer which uses the sandwich product with rotors to maintain grade invariance.
"""

from keras.layers import Input, Reshape, Flatten, GlobalAveragePooling2D, \
    Dense, Dropout, BatchNormalization
from keras.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from tfga import GeometricAlgebra
from tfga.layers import TensorToGeometric, GeometricProductConv1D, GeometricToTensor, GeometricSandwichProductDense
from layers.layers import RotorConv2D
from clifford.g3c import *
from math import sqrt
from layers.operations import q2S, translation_rotor, down1D
import csv
import matplotlib.pyplot as plt
import pandas as pd

ga = GeometricAlgebra(metric=[1, 1, 1, 1])

# define network

idx = ga.get_kind_blade_indices("even")  # gets a mask of indices for the
# given ga which satisfy the blade kind

# defining base model
# input tensor for RGB (I think) so has additional 3 dimensions
model = InceptionV3(classifier_activation=None, weights="imagenet",
                    input_tensor=Input(shape=(224, 224, 3)))

x2 = Dropout(0.3)(model.layers[-2].output)
x2 = Reshape((-1, 8, 1, 8))(x2)
x2 = TensorToGeometric(ga, blade_indices=idx)(x2)

x2 = RotorConv2D(
    ga, filters=4, kernel_size=8, stride_vertical=2, stride_horizontal=2, padding='SAME', blade_indices_kernel=idx, blade_indices_bias=idx,
    activation='relu'
)(x2)
x2 = RotorConv2D(
    ga, filters=1, kernel_size=8, stride_vertical=2, stride_horizontal=2, padding='SAME', blade_indices_kernel=idx, blade_indices_bias=idx,
    activation='relu'
)(x2)

x2 = Reshape((-1, 16))(x2)

x2 = GeometricSandwichProductDense(
    ga, units=8, activation="relu",
    blade_indices_kernel=idx,
    blade_indices_bias=idx)(x2)

x2 = GeometricSandwichProductDense(
    ga, units=1, activation="tanh",
    blade_indices_kernel=idx,
    blade_indices_bias=idx)(x2)

x2 = GeometricToTensor(ga, blade_indices=idx)(x2)
outputs2 = Flatten()(x2)

Model1 = tf.keras.Model(inputs=model.input, outputs=outputs2)
Model1.summary()
CGAPoseNet = Model1