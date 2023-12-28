"""
This contains the code used to generate networks for the technical milestone report.
"""

from keras.models import Model
from keras.layers import Input, Reshape, Flatten, GlobalAveragePooling2D, \
    Dense, Dropout, BatchNormalization
from keras.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from tfga import GeometricAlgebra
from tfga.layers import TensorToGeometric, GeometricProductConv1D, GeometricToTensor, GeometricSandwichProductDense
from layers.layers import RotorConv1D, EquivariantNonLinear
from clifford.g3c import *
from math import sqrt
from layers.operations import q2S, translation_rotor, down1D
import csv
import matplotlib.pyplot as plt
import pandas as pd

# set random seed
tf.random.set_seed(0)

# define ga
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
    ga, filters=4, kernel_size=8, stride=2, padding='SAME', blade_indices_kernel=idx, blade_indices_bias=idx,
    activation='relu'
)(x2)
x2 = RotorConv1D(
    ga, filters=1, kernel_size=8, stride=4, padding='SAME', blade_indices_kernel=idx, blade_indices_bias=idx,
    activation='relu'
)(x2)

x2 = Reshape((-1, 16))(x2)

x2 = GeometricSandwichProductDense(
    ga, units=8, activation="relu",
    blade_indices_kernel=idx,
    blade_indices_bias=idx)(x2)
x2 = EquivariantNonLinear(ga, activation='relu')(x2)
x2 = GeometricSandwichProductDense(
    ga, units=1, activation="tanh",
    blade_indices_kernel=idx,
    blade_indices_bias=idx)(x2)

x2 = GeometricToTensor(ga, blade_indices=idx)(x2)
outputs2 = Flatten()(x2)

Model1 = tf.keras.Model(inputs=model.input, outputs=outputs2)
Model1.summary()
CGAPoseNet = Model1

FOLDER = "OldHospital"  # Change the name to change the dataset
PATH = "C:/Users/Christian/Documents/Coursework/iib_project/"

# reading the dataset labels and converting them into motors (Train set)

list_of_lines = open(PATH + FOLDER + "/dataset_train.txt").readlines()
position_train = []
fieldnames = ['filename', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
df_train = pd.DataFrame(columns=fieldnames)

for i in range(3, len(list_of_lines)):
    a = list_of_lines[i].split()

    position_train = np.append(position_train, [float(a[1]), float(a[2]), float(a[3])])

    Ta = translation_rotor(float(a[1]) * e1 + float(a[2]) * e2 + float(a[3]) * e3)
    R = q2S(float(a[4]), float(a[5]), float(a[6]), float(a[7]))

    M = Ta * R

    a[1] = M[0]
    a[2] = M[6]
    a[3] = M[7]
    a[4] = M[8]
    a[5] = M[10]
    a[6] = M[11]
    a[7] = M[13]
    a.append(M[26])

    df_train.loc[len(df_train.index)] = a

position_train = np.reshape(position_train, (-1, 3))

dir = PATH + FOLDER + "/"

print(df_train.head(10))

# reads the dataset labels and converts them into motors (Test Set)

list_of_lines = open(PATH + FOLDER + "/dataset_test.txt").readlines()  # no need for closing, python will do it for you
df_test = pd.DataFrame(columns=fieldnames)
position_test = []
y_test = []

for i in range(3, len(list_of_lines)):

    a = list_of_lines[i].split()
    position_test = np.append(position_test, [float(a[1]), float(a[2]), float(a[3])])
    Ta = translation_rotor(float(a[1]) * e1 + float(a[2]) * e2 + float(a[3]) * e3)
    R = q2S(float(a[4]), float(a[5]), float(a[6]), float(a[7]))

    M = Ta * R

    a[1] = M[0]
    a[2] = M[6]
    a[3] = M[7]
    a[4] = M[8]
    a[5] = M[10]
    a[6] = M[11]
    a[7] = M[13]
    a.append(M[26])
    if i == 3:
        print(a)
        print(list_of_lines[i])

    if i % 100 == 0:
        print(i)
    df_test.loc[len(df_test.index)] = a

    y_test.append(float(a[1]))
    y_test.append(float(a[2]))
    y_test.append(float(a[3]))
    y_test.append(float(a[4]))
    y_test.append(float(a[5]))
    y_test.append(float(a[6]))
    y_test.append(float(a[7]))
    y_test.append(float(a[8]))

position_test = np.reshape(position_test, (-1, 3))

y_test = np.reshape(y_test, (-1, 8))

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
train_generator = train_datagen.flow_from_dataframe(dataframe=df_train, directory=dir,
                                                    x_col="filename", y_col=columns, has_ext=True,
                                                    class_mode="raw", target_size=(224, 224),
                                                    shuffle=True,
                                                    sort=False,
                                                    batch_size=64)

test_generator = test_datagen.flow_from_dataframe(dataframe=df_test, directory=dir,
                                                  x_col="filename", y_col=columns, has_ext=True,
                                                  class_mode="raw", target_size=(224, 224),
                                                  shuffle=False,
                                                  sort=False,
                                                  batch_size=64)

position_test = np.reshape(position_test, (-1, 3))

# defining hyperparameters

nb_epoch = 100  # changed from 100 for speed
batch_size = 64

initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.98,
    staircase=True)

# compiling the model
CGAPoseNet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                   loss=mean_squared_error, run_eagerly=True)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

# training
model_train = CGAPoseNet.fit(train_generator,
                             validation_data=test_generator,
                             epochs=nb_epoch,
                             verbose=1,
                             shuffle=True,
                             callbacks=es_callback,
                             batch_size=batch_size)

# plotting losses
loss = model_train.history['loss']
val_loss = model_train.history['val_loss']
epochs = range(0, np.size(loss))

plt.figure()
plt.plot(epochs, loss, 'b-', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training loss')
plt.legend()
plt.show()

# saving model

CGAPoseNet.save_weights('test.weights.h5')
