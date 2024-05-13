"""
This script evaluates models trained using the protein structure prediction dataset.
"""

import tensorflow as tf
import pickle
import numpy as np
from run_experiment import custom_norm, get_nodes_and_edges, orient_coords
import os
from proteins_model import cga_transformer_model, mlp_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# set random seed
tf.random.set_seed(0)

# set path to data
PATH = "C:/Users/Christian/Documents/Coursework/iib_project/ga_transformer/"
DATA_PATH = PATH + "datasets/psp_dataset"


def load_weights(model_name, model):
    """
    Assigns weights to created model.
    :param model_name: name of pkl file containing model weights
    :param model: model architecture corresponding to weights
    :return:
    """
    model_path = PATH + "models/{}.pkl".format(model_name)
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)


def evaluate(model, filenames_list, features_path, distances_path, coords_path):
    """
    Finds and outputs mae and ssim for a trained model.
    :param model: model to assess
    :param filenames_list: list of protein filenames to assess
    :param features_path: path to features
    :param distances_path: path to distances
    :param coords_path: path to coordinates
    :return:
    """
    avg_mae = 0
    avg_ssim = 0
    avg_mse = 0
    count = 0

    mae = tf.keras.losses.MeanAbsoluteError()
    mse = tf.keras.losses.MeanSquaredError()

    for filename in filenames_list:
        count += 1
        filename = os.path.splitext(filename)[0]

        nodes, edges, mask, l = get_nodes_and_edges(filename, [features_path], 27, 4, distances_path)
        distance = tf.convert_to_tensor(np.load(distances_path + filename + '-ca.npy', allow_pickle=True))
        true_coords = tf.convert_to_tensor(np.load(coords_path + filename + ".npy", allow_pickle=True),
                                           dtype=tf.float32)

        # feed forward into model
        coord = model(nodes, edges, mask=mask)

        # re-orient coordinates of output from model
        oriented_coords = orient_coords(coord, true_coords)

        # compute mse between oriented and true coordinates
        avg_mse += tf.reduce_mean(mse(oriented_coords, true_coords))

        # convert to predicted distances
        coord = tf.repeat(coord, repeats=coord.shape[1], axis=0)
        pred_dist = tf.expand_dims(tf.expand_dims(custom_norm(coord - tf.transpose(coord, perm=[1, 0, 2])), 0),
                                   -1)  # could do a reshape instead

        # compute loss
        actual_dist = tf.expand_dims(tf.expand_dims(tf.cast(distance, tf.float32), -1), 0)

        avg_mae += tf.reduce_mean(mae(pred_dist, actual_dist))
        avg_ssim += tf.image.ssim(img1=pred_dist, img2=actual_dist, max_val=255)

        if count % 20 == 0:
            print(count, "/", len(filenames_list))
            print("Current avg MAE: ", avg_mae / count)

        del nodes, edges, mask, l, coord, distance, actual_dist, pred_dist

    avg_mae /= len(filenames_list)
    avg_ssim /= len(filenames_list)
    avg_mse /= len(filenames_list)

    print("MAE:", avg_mae)
    print("Avg SSIM:", avg_ssim)
    print("MSE:", avg_mse)


def visualise_output(cga_model, mlp, filename, features_path, distances_path, coords_path):
    """
    Visualises a single output from a given PSP model corresponding to the given filename's protein.
    :param cga_model: cga psp model to use
    :param mlp: mlp psp model to use
    :param filename: filename of given protein
    :param features_path: path to features
    :param distances_path: path to distances
    :param coords_path: path to coordinates
    :return:
    """
    filename = os.path.splitext(filename)[0]
    nodes, edges, mask, l = get_nodes_and_edges(filename, [features_path], 27, 4, distances_path)
    true_coords = tf.convert_to_tensor(np.load(coords_path + filename + ".npy", allow_pickle=True),
                                       dtype=tf.float32)

    # feed forward into model
    coord = cga_model(nodes, edges, mask=mask)
    mlp_coord = mlp(nodes, edges, mask=mask)

    # re-orient coordinates of output from model
    oriented_coords = orient_coords(coord, true_coords)
    mlp_oriented_coords = orient_coords(mlp_coord, true_coords)

    # convert oriented and predicted coords to numpy arrays
    oriented_np = tf.reshape(oriented_coords, [-1, 3]).numpy()
    actual_np = tf.reshape(true_coords, [-1, 3]).numpy()
    mlp_np = tf.reshape(mlp_oriented_coords, [-1, 3]).numpy()

    # plot coordinates
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(actual_np[:, 0], actual_np[:, 1], actual_np[:, 2], c='b')
    ax.plot(actual_np[:, 0], actual_np[:, 1], actual_np[:, 2], c='b', label="actual")
    ax.scatter(oriented_np[:, 0], oriented_np[:, 1], oriented_np[:, 2], c='r')
    ax.plot(oriented_np[:, 0], oriented_np[:, 1], oriented_np[:, 2], c='r', label="CGATr predicted")
    ax.scatter(mlp_np[:, 0], mlp_np[:, 1], mlp_np[:, 2], c='g')
    ax.plot(mlp_np[:, 0], mlp_np[:, 1], mlp_np[:, 2], c='g', label="MLP predicted")

    ax.legend()
    plt.show()


def main():
    """
    Main function to run.
    :return:
    """

    model = cga_transformer_model(num_blocks=2)
    mlp = mlp_model()
    load_weights("psp_cgatr_2_block", model)  # demonstrates loading of model
    load_weights("psp_mlp", mlp)

    deepcov_features_path = DATA_PATH + '/data/deepcov/features/'
    deepcov_distances_path = DATA_PATH + '/data/deepcov/ca_distance/'
    deepcov_coords_path = DATA_PATH + '/data/deepcov/ca_coords/'
    psicov_features_path = DATA_PATH + '/data/psicov/features/'
    psicov_distances_path = DATA_PATH + '/data/psicov/ca_distance/'
    psicov_coords_path = DATA_PATH + '/data/psicov/ca_coords/'

    lst = os.listdir(deepcov_features_path) + os.listdir(psicov_features_path)
    lst.sort()

    lst_train = []
    for filename in lst:
        pdb = os.path.splitext(filename)[0]
        if os.path.exists(deepcov_features_path + pdb + '.pkl'):
            lst_train = np.append(lst_train, filename)

    # lst_test is separate from evaluation set used during training
    lst_test = []
    for filename in lst:
        pdb = os.path.splitext(filename)[0]
        if os.path.exists(psicov_features_path + pdb + '.pkl'):
            lst_test = np.append(lst_test, filename)

    lsttrain, lstval = train_test_split(lst_train, test_size=0.20, random_state=42)

    model.summary()
    print(lsttrain[0])

    visualise_output(model, mlp, lsttrain[1], deepcov_features_path, deepcov_distances_path, deepcov_coords_path)
    visualise_output(model, mlp, lsttrain[2], deepcov_features_path, deepcov_distances_path, deepcov_coords_path)
    del mlp

    evaluate(model, lstval, deepcov_features_path, deepcov_distances_path, deepcov_coords_path)
    evaluate(model, lst_test, psicov_features_path, psicov_distances_path, psicov_coords_path)
    evaluate(model, lsttrain, deepcov_features_path, deepcov_distances_path, deepcov_coords_path)


if __name__ == "__main__":
    main()
