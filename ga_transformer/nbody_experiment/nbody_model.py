"""
This contains the functions which define the GA transformer model.
"""

import tensorflow as tf
from tfga import GeometricAlgebra
from nbody_cga_transformer import CGATransformer


def cga_transformer_model(num_bodies=4):
    """
    Creates a transformer which uses CGA to solve geometric problems.
    :param num_bodies: number of bodies used in simulation
    :return: CGA transformer
    """
    # define geometric algebra instance - CGA
    cga = GeometricAlgebra(metric=[1, 1, 1, 1, -1])

    # instantiate model
    model = CGATransformer(geometric_algebra=cga, n_bodies=num_bodies)

    initial_learning_rate = 2e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True)

    # build the model
    model.build(input_shape=(None, num_bodies, 7))

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.mean_squared_error)

    return model


def mlp_model():
    """
    Creates a basic MLP model for use in the n-body modelling problem.
    :return: MLP model
    """
    # define model layers and instantiate model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(4, 7,)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(12),
        tf.keras.layers.Reshape((4, 3))
    ])

    initial_learning_rate = 3e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.99,
        staircase=True)

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.mean_squared_error)

    return model


if __name__ == "__main__":
    # from run_experiment import load_data, PATH
    # full_path = PATH + "datasets/" + "train" + ".npz"
    # x, y, trajectories = load_data(full_path)
    ga = GeometricAlgebra(metric=[1, 1, 1, 1, -1])
    n = ga.e3 + ga.e4
    print(ga.e(["3"]))
    print(n)
    print(ga.blade_mvs[4] + ga.blade_mvs[5])
    # a = embed_points(x[..., 1:4], ga)
    # b = embed_velocities(x[..., 1:4], x[..., 4:7], ga)
    # c = cga_embed_inputs(x, ga)
    # d = cga_extract_outputs(c, ga)
    # print(x.shape, c.shape)
    # print(y.shape, d.shape)
    # print(c[0, 0])
    # print(c[0, 1])
    # print(ga.inner_prod(c[0, 0], c[0, 1]))
    # print(ga.geom_prod(c[0, 0], ga.reversion(c[0, 1])))
    # print(ga.num_blades)
