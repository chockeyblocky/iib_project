"""
This file is used for testing custom rotor convolution layer.
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
from layers.layers import RotorConv1D
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
newfile = open(PATH + FOLDER + "/new_dataset_train.txt", "w")

position_train = []

for i in range(0, 3):
    newfile.write(list_of_lines[i])
    newfile.write("\n")

for i in range(3, len(list_of_lines)):
    a = []
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

    b = " ".join(map(str, a))
    list_of_lines[i] = b
    newfile.write(list_of_lines[i])
    newfile.write("\n")

position_train = np.reshape(position_train, (-1, 3))

# reads the dataset frames, reshapes them and normalizes them  (Train Set)

list_train = open(
    PATH + FOLDER + "/new_dataset_train.txt").readlines()  # no need for closing, python will do
# it for you

dir = PATH + FOLDER + "/"

with open(dir + 'TRAIN.csv', "w") as csv_file:
    fieldnames = ['filename', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(6, len(list_train)):

        if i % 100 == 0:
            print(i)

        a = list_train[i].split()

        d = {'filename': a[0],
             'a': a[1],
             'b': a[2],
             'c': a[3],
             'd': a[4],
             'e': a[5],
             'f': a[6],
             'g': a[7],
             'h': a[8]}
        writer.writerow(d)

# reads the dataset labels and converts them into motors (Test Set)

list_of_lines = open(PATH + FOLDER + "/dataset_test.txt").readlines()  # no need for closing, python will do it for you
newfile = open(PATH + FOLDER + "/new_dataset_test.txt", "w")

position_test = []
for i in range(0, 3):
    newfile.write(list_of_lines[i])
    newfile.write("\n")

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

    # print(a)

    b = " ".join(map(str, a))
    list_of_lines[i] = b
    newfile.write(list_of_lines[i])
    newfile.write("\n")

    if i == 3:
        print(a)
        print(list_of_lines[i])

position_test = np.reshape(position_test, (-1, 3))

y_test = []

# reads the dataset frames, reshapes them and normalizes them  (Train Set)

list_test = open(PATH + FOLDER + "/new_dataset_test.txt").readlines()

with open(dir + 'TEST.csv', "w") as csv_file:
    fieldnames = ['filename', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(6, len(list_test)):

        if i % 100 == 0:
            print(i)

        a = list_test[i].split()

        # img = cv2.imread("/content/drive/MyDrive/"+ FOLDER + "/" + a[0])

        # resized = cv2.resize(img, (224, 224))
        # normalized = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # x_train = np.append(x_train, normalized)

        y_test.append(float(a[1]))
        y_test.append(float(a[2]))
        y_test.append(float(a[3]))
        y_test.append(float(a[4]))
        y_test.append(float(a[5]))
        y_test.append(float(a[6]))
        y_test.append(float(a[7]))
        y_test.append(float(a[8]))

        d = {'filename': a[0],
             'a': a[1],
             'b': a[2],
             'c': a[3],
             'd': a[4],
             'e': a[5],
             'f': a[6],
             'g': a[7],
             'h': a[8]}
        writer.writerow(d)

y_test = np.reshape(y_test, (-1, 8))

df_train = pd.read_csv(dir + '/TRAIN.csv')
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

# df = pd.read_csv(dir + 'train_df.csv', delimiter=' ', header=None, names=['filename', 'a', 'b', 'c', 'd', 'e', 'f',
# 'g', 'h'])
columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
train_generator = train_datagen.flow_from_dataframe(dataframe=df_train, directory=dir,
                                                    x_col="filename", y_col=columns, has_ext=True,
                                                    class_mode="raw", target_size=(224, 224),
                                                    shuffle=True,
                                                    sort=False,
                                                    batch_size=64)

df_test = pd.read_csv(dir + '/TEST.csv')
test_generator = test_datagen.flow_from_dataframe(dataframe=df_test, directory=dir,
                                                  x_col="filename", y_col=columns, has_ext=True,
                                                  class_mode="raw", target_size=(224, 224),
                                                  shuffle=False,
                                                  sort=False,
                                                  batch_size=64)

position_test = np.reshape(position_test, (-1, 3))

# defining hyperparameters

nb_epoch = 20  # changed from 100 for speed
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

CGAPoseNet.save('model_1.h5')
