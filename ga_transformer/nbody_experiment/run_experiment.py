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


def main():
    """
    Main function to run.
    :return:
    """
    load_data(PATH + 'datasets/train.npz')


if __name__ == "__main__":
    main()
