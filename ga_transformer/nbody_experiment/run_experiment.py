"""
This will run the n-body experiment on a defined model.
Original source: 2023 Qualcomm Technologies, Inc.
This has been adapted to use TensorFlow for academic purposes.
"""

import tensorflow as tf
import numpy as np
import pickle
import tensorflow_datasets as tfds

# set random seed
tf.random.set_seed(0)

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

    learning_rate = 1e-4

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.mean_squared_error)

    return model


def save_model(name, model):
    """
    Saves model weights as .pkl files
    :param name: String under which to save model
    :param model: TensorFlow model to save
    :return:
    """
    with open('{}.pkl'.format(name), 'wb') as f:
        pickle.dump(model.get_weights(), f)


def get_grads(model, x, y):
    """
    Gets the training gradients from a single sample.
    :param model: model to apply
    :param x: input data
    :param y: output data
    :return: gradients of trainable weights
    """
    with tf.GradientTape() as tape:
        # Trainable variables are automatically tracked by GradientTape
        loss = tf.keras.losses.mean_squared_error(y, model(x))

    # Use GradientTape to calculate the gradients with respect to weights
    grads = tape.gradient(loss, model.trainable_weights)

    return grads


def train(model, x, y, optimizer):
    """
    Applies the training gradients from a single batch.
    :param model: model to apply
    :param x: input data
    :param y: output data
    :param optimizer: optimizer to be used
    :return:
    """
    with tf.GradientTape() as tape:
        # Trainable variables are automatically tracked by GradientTape
        loss = tf.keras.losses.mean_squared_error(y, model(x))

    # Use GradientTape to calculate the gradients with respect to weights
    grads = tape.gradient(loss, model.trainable_weights)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))


def training_loop(model, dataset):
    """
    Full training loop
    :param model: model to train
    :param dataset: training dataset
    :return:
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    epochs = 100

    for epoch in range(epochs):
        for element in dataset:
            x = element[0]
            y = element[1]
            print(x.shape)
            print(y.shape)
            print(model(x).shape)
            # Update the model with the single giant batch
            train(model, x, y, optimizer)

            # Track this before I update
            # weights.append(model.w.numpy())
            # biases.append(model.b.numpy())
            mse = custom_mse(y, model(x))
            print(mse)

        print("Epoch", epoch)
        print(mse)


def custom_mse(y_true, y_pred):
    """
    Define custom mean squared error loss function.
    :return: mse
    """
    return tf.reduce_mean(tf.reduce_mean(tf.square(y_true - y_pred), axis=-1), axis=-1)


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
    num_epochs = 10

    ds_train_batch = ds_train.shuffle(1000, reshuffle_each_iteration=False).batch(batch_size)
    ds_val_batch = ds_val.shuffle(1000, reshuffle_each_iteration=False).batch(batch_size)

    # training
    model_train = model.fit(ds_train_batch,
                            validation_data=ds_val_batch,
                            epochs=num_epochs,
                            callbacks=es_callback)
    # model_train = model.fit(x=x, y=y,
    #                         validation_data=(xval, yval),
    #                         epochs=num_epochs,
    #                         batch_size=100,
    #                         callbacks=es_callback)

    # debugging training loop
    # for element in ds_train_batch.take(1):
    #     print(model(element[0]))
    #     print(get_grads(model, element[0], element[1]))

    # training_loop(model, ds_train_batch)


if __name__ == "__main__":
    main()
