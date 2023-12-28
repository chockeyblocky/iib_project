"""
This tests that loading and saving of model architectures with custom layers works.
"""
import tensorflow as tf
import keras
import numpy as np
from tfga import GeometricAlgebra
from tfga.layers import TensorToGeometric, GeometricToTensor, GeometricSandwichProductDense
from layers.layers import *
from keras.layers import Input, Reshape, Flatten, GlobalAveragePooling2D, \
    Dense, Dropout, BatchNormalization

ga = GeometricAlgebra(metric=[1, 1, 1, 1])
idx = ga.get_kind_blade_indices("even")  # gets a mask of indices for the
# given ga which satisfy the blade kind

# Create the model.
def get_model():
    inputs = keras.Input(shape=(16,))
    x2 = Reshape((-1, 1, 8))(inputs)
    x2 = TensorToGeometric(ga, blade_indices=idx)(x2)
    x2 = GeometricSandwichProductDense(ga, units=8, blade_indices_kernel=idx, blade_indices_bias=idx)(x2)

    x2 = RotorConv1D(
        ga, filters=4, kernel_size=2, stride=1, padding='SAME', blade_indices_kernel=idx, blade_indices_bias=idx,
        activation='relu'
    )(x2)
    x2 = RotorConv1D(
        ga, filters=1, kernel_size=2, stride=1, padding='SAME', blade_indices_kernel=idx, blade_indices_bias=idx,
        activation='relu'
    )(x2)
    x2 = Reshape((-1, 16))(x2)
    x2 = GeometricToTensor(ga, blade_indices=idx)(x2)
    outputs2 = Flatten()(x2)
    outputs = keras.layers.Dense(1, activation='relu')(outputs2)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="mean_squared_error")
    return model


# Train the model.
def train_model(model):
    input = np.random.random((16, 16))
    target = np.random.random((16, 1))
    model.fit(input, target)
    return model


test_input = np.random.random((16, 16))
test_target = np.random.random((16, 1))

model = get_model()
model.summary()
print(model(test_input))
model = train_model(model)
model.save_weights("custom_model.weights.h5")

new_model = get_model()
new_model.load_weights("custom_model.weights.h5")

# Now, we can simply load without worrying about our custom objects.
# reconstructed_model = keras.models.load_model("custom_model.keras")

# Let's check:
print(np.testing.assert_allclose(
    model.predict(test_input), new_model.predict(test_input)
))