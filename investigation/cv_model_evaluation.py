"""
This contains the code used to evaluate models trained on the pose estimation datasets.
"""
import keras.models
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
from layers.operations import q2S, translation_rotor, down1D, up1D
import csv
import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import IPython

# set random seed
tf.random.set_seed(0)

# define ga
ga = GeometricAlgebra(metric=[1, 1, 1, 1])

# define blade indices required
idx = ga.get_kind_blade_indices("even")  # gets a mask of indices for the
# given ga which satisfy the blade kind

# define paths to dataset
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

# construct model architecture - change for each architecture

model = InceptionV3(classifier_activation=None, weights="imagenet",
                    input_tensor=Input(shape=(224, 224, 3)))

x2 = Dropout(0.3)(model.layers[-2].output)
x2 = Reshape((-1, 8))(x2)
x2 = TensorToGeometric(ga, blade_indices=idx)(x2)
x2 = GeometricSandwichProductDense(
    ga, units=128, activation="relu",
    blade_indices_kernel=idx,
    blade_indices_bias=idx)(x2)
x2 = GeometricSandwichProductDense(
    ga, units=64, activation="relu",
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
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.98,
    staircase=True)
CGAPoseNet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                   loss=mean_squared_error, run_eagerly=True)

# loading model
model_path = PATH + "models/basic.weights.h5"
CGAPoseNet.load_weights(model_path)

# make predictions
y_pred = CGAPoseNet.predict(test_generator)

MSE = []

tot = 0
cnt = 0
for i in range(len(y_test)):
    mse = (np.square(y_test[i] - y_pred[i])).mean()

    MSE = np.append(MSE, mse)

    # printing the first 20 motors M, \hat{M} if the MSE between them is close
    if cnt < 20 and mse < 0.0008:
        print("original:", y_test[i])

        X = y_test[i]
        Y = y_pred[i]

        M_real = X[0] + X[1] * e12 + X[2] * e13 + X[3] * e14 + X[4] * e23 + X[5] * e24 + X[6] * e34 + X[7] * e1234
        M_pred = Y[0] + Y[1] * e12 + Y[2] * e13 + Y[3] * e14 + Y[4] * e23 + Y[5] * e24 + Y[6] * e34 + Y[7] * e1234

        print("prediction:", y_pred[i])
        print("****")
        cnt += 1

    tot += mse

print(tot)
np.save("MSE.npy", MSE)

# evaluating positional and rotational error

origin = e4
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
list_of_lines = open(PATH + FOLDER + "/dataset_test.txt").readlines()  # no need for closing, python will do it for you

positional_error = []
rotational_error = []

translation = []
translation_pred = []

rotation = []
rotation_pred = []
for i in range(len(y_test)):
    a = []
    a = list_of_lines[i + 3].split()

    # x is required by the function down1D
    x = float(a[1]) * e1 + float(a[2]) * e2 + float(a[3]) * e3

    X = y_test[i]
    Y = y_pred[i]

    # construct M and \hat{M}
    M_real = X[0] + X[1] * e12 + X[2] * e13 + X[3] * e14 + X[4] * e23 + X[5] * e24 + X[6] * e34 + X[7] * e1234
    M_pred = Y[0] + Y[1] * e12 + Y[2] * e13 + Y[3] * e14 + Y[4] * e23 + Y[5] * e24 + Y[6] * e34 + Y[7] * e1234

    # normalizing
    M_pred = M_pred / sqrt((M_pred * ~M_pred)[0])

    # predicted and real displacement vector \hat{D}, D in spherical space
    S = M_pred * origin * ~M_pred
    T = M_real * origin * ~M_real

    # predicted and real displacement vector \hat{d}, d in Euclidean space
    # s = down1D(S, x)
    # t = down1D(T, x)

    s = down1D(S)
    t = down1D(T)

    # POSITIONAL ERROR
    mae = np.mean(np.abs(np.array([t[1], t[2], t[3]]) - np.array([s[1], s[2], s[3]])))

    positional_error = np.append(positional_error, mae)

    translation = np.append(translation, np.array([t[1], t[2], t[3]]))
    translation_pred = np.append(translation_pred, np.array([s[1], s[2], s[3]]))

    # plotting the camera trace
    ax.scatter(t[1], t[2], t[3], s=20, c="r")
    ax.scatter(s[1], s[2], s[3], s=20, c="b", alpha=0.5)

    Tup = translation_rotor(t[1] * e1 + t[2] * e2 + t[3] * e3)
    Sup = translation_rotor(s[1] * e1 + s[2] * e2 + s[3] * e3)

    # predicted and real rotors \hat{R}, R
    R_pred = ~Sup * M_pred
    R_real = ~Tup * M_real

    if (R_real * ~R_pred)[0] > 1:
        error = (np.arccos(1)) * 360 / (2 * np.pi)
    elif (R_real * ~R_pred)[0] < -1:
        error = (np.arccos(-1)) * 360 / (2 * np.pi)
    else:
        # ROTATIONAL ERROR
        error = (np.arccos((R_real * ~R_pred)[0])) * 360 / (2 * np.pi)

    rotational_error = np.append(rotational_error, error)

    rotation = np.append(rotation, np.array([R_real[0], R_real[6], R_real[7], R_real[10]]))
    rotation_pred = np.append(rotation_pred, np.array([R_pred[0], R_pred[6], R_pred[7], R_pred[10]]))

plt.show()

# storing rotational and translational errors
np.save("translation_error.npy", positional_error)
np.save("rotational_error.npy", rotational_error)

# storing original and predicted translations
np.save("T.npy", translation)
np.save("S.npy", translation_pred)

# storing original and predicted rotations
np.save("R.npy", rotation)
np.save("Q.npy", rotation_pred)

# plotting the camera orientation (coefficients e_{12}, e_{13}, e_{23} of rotor R)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

N = 200
stride = 1

u = np.linspace(0, 2 * np.pi, N)
v = np.linspace(0, np.pi, N)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, linewidth=0.0, alpha=0.1, cstride=stride, rstride=stride)

ax.scatter(0, 0, 0, c="k", marker="s", label="O")

rotation = np.reshape(rotation, (-1, 4))
rotation_pred = np.reshape(rotation_pred, (-1, 4))
ax.scatter(rotation[:, 1], rotation[:, 2], rotation[:, 3], s=15, c="r")
ax.scatter(rotation_pred[:, 1], rotation_pred[:, 2], rotation_pred[:, 3], s=15, c="b")
plt.show()

print("Median positional error: ", np.median(positional_error))
print("Mean positional error: ", np.mean(positional_error))

print("Median rotational error: ", np.median(rotational_error))
print("Mean rotational error: ", np.mean(rotational_error))

# visualise intermediate outputs for a random example
example = next(test_generator)

test = example[0][0]
test = np.reshape(test, (-1, 224, 224, 3))
CGAPoseNet.load_weights("weights.h5")
layer_outs = CGAPoseNet(test)
print(layer_outs)
np.save("predmotor.npy", layer_outs)

m = Model(inputs=CGAPoseNet.input, outputs=CGAPoseNet.layers[-4].output)
layer4 = m(test)

m = Model(inputs=CGAPoseNet.input, outputs=CGAPoseNet.layers[-5].output)
layer5 = m(test)

m = Model(inputs=CGAPoseNet.input, outputs=CGAPoseNet.layers[-6].output)
layer6 = m(test)

x0 = np.multiply([0, 1, 2, 0], 10)
y0 = np.multiply([0, 0, 1, 2], 10)
z0 = np.multiply([0, 2, 0, 1], 10)


def packmotor(coeff):
    N = 0
    N += coeff[0] * 1
    N += coeff[1] * e12
    N += coeff[2] * e13
    N += coeff[3] * e14
    N += coeff[4] * e23
    N += coeff[5] * e24
    N += coeff[6] * e34
    N += coeff[7] * e1234

    N = N / sqrt((N * ~N)[0])
    return N


M = packmotor(example[1][0])
Mp = packmotor(np.array(layer_outs[0]))

xr = []
yr = []
zr = []

for i in range(len(x0)):
    v = float(x0[i]) * e1 + float(y0[i]) * e2 + float(z0[i]) * e3
    V = up1D(v)
    P = M * V * ~M

    p = down1D(P)
    xr.append(p[1])
    yr.append(p[2])
    zr.append(p[3])

xp = []
yp = []
zp = []

for i in range(len(x0)):
    v = float(x0[i]) * e1 + float(y0[i]) * e2 + float(z0[i]) * e3
    V = up1D(v)
    P = Mp * V * ~Mp

    p = down1D(P)
    xp.append(p[1])
    yp.append(p[2])
    zp.append(p[3])

fig = go.Figure(data=[

    go.Mesh3d(
        x=xr,
        y=yr,
        z=zr,
        # colorbar_title='z',
        colorscale=[[0, 'firebrick'],
                    [0.5, 'red'],
                    [1, 'lightcoral']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=np.linspace(0, 1, 8, endpoint=True),
        # i, j and k give the vertices of triangles
        i=[0, 0, 0, 1],
        j=[1, 2, 3, 2],
        k=[2, 3, 1, 3],
        showscale=False,
        name="Ground Truth"
    ),

    go.Mesh3d(
        x=xp,
        y=yp,
        z=zp,
        # colorbar_title='z',
        colorscale=[[0, 'navy'],
                    [0.5, 'blue'],
                    [1, 'dodgerblue']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=np.linspace(0, 1, 8, endpoint=True),
        # i, j and k give the vertices of triangles
        i=[0, 0, 0, 1],
        j=[1, 2, 3, 2],
        k=[2, 3, 1, 3],
        showscale=False,
        name="Predicted"

    ),
])
fig.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', scene_aspectmode='data')
pio.write_html(fig, file="tetrahedra.html", auto_open=True)

M = 0
for i in range(len(layer5[0])):
    c = layer5[0][i]

    xn = []
    yn = []
    zn = []

    coeff = [float(c[0]), float(c[5]), float(c[6]),
             float(c[7]), float(c[8]), float(c[9]),
             float(c[10]), float(c[15])]

    # M += packmotor(coeff)
    M = packmotor(coeff)
    M = M / sqrt((M * ~M)[0])

    for j in range(len(x0)):
        v = float(x0[j]) * e1 + float(y0[j]) * e2 + float(z0[j]) * e3
        V = up1D(v)
        P = M * V * ~M
        p = down1D(P)

        xn.append(p[1])
        yn.append(p[2])
        zn.append(p[3])

    fig.add_trace(go.Mesh3d(
        x=xn,
        y=yn,
        z=zn,
        # colorbar_title='z',
        colorscale=[[0, 'green'],
                    [0.5, 'limegreen'],
                    [1, 'lime']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=np.linspace(0, 1, 8, endpoint=True),
        # i, j and k give the vertices of triangles
        i=[0, 0, 0, 1],
        j=[1, 2, 3, 2],
        k=[2, 3, 1, 3],
        showscale=False,
        opacity=0.5,
        name="128 coeff"
    ))

M = 0
for i in range(len(layer4[0])):
    c = layer6[0][i]

    xn = []
    yn = []
    zn = []
    coeff = [float(c[0]), float(c[5]), float(c[6]),
             float(c[7]), float(c[8]), float(c[9]),
             float(c[10]), float(c[15])]
    # M += packmotor(coeff)

    M = packmotor(coeff)
    M = M / sqrt((M * ~M)[0])
    for j in range(len(x0)):
        v = float(x0[j]) * e1 + float(y0[j]) * e2 + float(z0[j]) * e3
        V = up1D(v)
        P = M * V * ~M
        p = down1D(P)

        xn.append(p[1])
        yn.append(p[2])
        zn.append(p[3])

    fig.add_trace(go.Mesh3d(
        x=xn,
        y=yn,
        z=zn,
        # colorbar_title='z',
        colorscale=[[0, 'orange'],
                    [0.5, 'gold'],
                    [1, 'yellow']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=np.linspace(0, 1, 8, endpoint=True),
        # i, j and k give the vertices of triangles
        i=[0, 0, 0, 1],
        j=[1, 2, 3, 2],
        k=[2, 3, 1, 3],
        showscale=False,
        opacity=0.3,
        name="64 coeff"
    ))

camera = dict(
    eye=dict(x=-2, y=2, z=0.1)
)

fig.update_layout(scene_camera=camera)
pio.write_html(fig, file="tetrahedra_all.html", auto_open=True)
IPython.display.HTML(filename="tetrahedra_all.html")
