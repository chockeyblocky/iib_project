"""
This will run the n-body experiment on a defined model.
Original source: 2023 Qualcomm Technologies, Inc.
This has been adapted to use TensorFlow for academic purposes.
"""

import tensorflow as tf
import numpy as np

# define path to dataset
PATH = "C:/Users/Christian/Documents/Coursework/iib_project/ga_transformer/"


def load_data(filename, subsample=None, keep_trajectories=False):
    """Loads data from file and converts to input and output tensors."""
    # Load data from file
    npz = np.load(filename, "r")
    m, x_initial, v_initial, x_final = (
        npz["m"],
        npz["x_initial"],
        npz["v_initial"],
        npz["x_final"],
    )

    # Convert to tensors
    m = tf.expand_dims(tf.convert_to_tensor(m, dtype=tf.float32), 2)
    x_initial = tf.convert_to_tensor(x_initial, dtype=tf.float32)
    v_initial = tf.convert_to_tensor(v_initial, dtype=tf.float32)
    x_final = tf.convert_to_tensor(x_final, dtype=tf.float32)

    # Concatenate into inputs and outputs
    x = tf.concat((m, x_initial, v_initial), axis=2)  # (batchsize, num_objects, 7)
    y = x_final  # (batchsize, num_objects, 3)

    # Optionally, keep raw trajectories around (for plotting)
    if keep_trajectories:
        trajectories = npz["trajectories"]
    else:
        trajectories = None

    # Subsample
    if subsample is not None and subsample < 1.0:
        n_original = len(x)
        n_keep = int(round(subsample * n_original))
        assert 0 < n_keep <= n_original
        x = x[:n_keep]
        y = y[:n_keep]
        if trajectories is not None:
            trajectories = trajectories[:n_keep]

    return x, y, trajectories


def create_dataset(dataset_name):
    """
    Creates tf dataset for training.
    :param dataset_name: type of dataset (e.g. "train")
    :return: tf dataset
    """
    full_path = PATH + "datasets/" + dataset_name + ".npz"
    x, y, trajectories = load_data(full_path)

    return tf.data.Dataset.from_tensor_slices((x, y))


def mlp_model():
    """
    Creates a basic MLP model for use in the n-body modelling problem.
    :return: MLP model
    """
    # define input shape
    x = tf.keras.Input(shape=(4, 7,))

    # define model layers and instantiate model
    layers = tf.keras.Sequential([
        tf.keras.layers.Reshape((-1, 28)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(12),
        tf.keras.layers.Reshape((-1, 4, 3))
    ])

    y = layers(x)
    model = tf.keras.Model(x, y)

    initial_learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.98,
        staircase=True)

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.mean_squared_error, run_eagerly=True)

    return model


def main():
    """
    Main function to run.
    :return:
    """
    ds_train = create_dataset("train")
    ds_val = create_dataset("val")
    model = mlp_model()
    model.summary()

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    batch_size = 100
    num_epochs = 100

    ds_train_batch = ds_train.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)
    ds_val_batch = ds_val.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)

    # training
    model_train = model.fit(ds_train_batch,
                            validation_data=ds_val_batch,
                            epochs=num_epochs,
                            callbacks=es_callback)


if __name__ == "__main__":
    main()
