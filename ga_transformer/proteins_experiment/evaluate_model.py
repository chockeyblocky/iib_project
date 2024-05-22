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


def save_statistics(stat_name, data):
    """
    Saves computed model evaluation statistics for plotting.
    :param stat_name: name under which to save stats
    :param data: tuple containing evaluated information
    :return:
    """

    path = PATH + "model_histories/{}.pkl".format(stat_name)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_statistics(stat_name):
    """
    Loads computed statistics for plotting.
    :param stat_name: name under which stats are saved
    :return: (mse_list, mae_list)
    """
    path = PATH + "model_histories/{}.pkl".format(stat_name)
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def evaluate(model, filenames_list, features_path, distances_path, coords_path, stat_name):
    """
    Finds and outputs mae and ssim for a trained model.
    :param model: model to assess
    :param filenames_list: list of protein filenames to assess
    :param features_path: path to features
    :param distances_path: path to distances
    :param coords_path: path to coordinates
    :param stat_name: name under which to save statistics
    :return:
    """
    avg_mae = 0
    avg_ssim = 0
    avg_mse = 0
    count = 0

    mse_list = []
    mae_list = []

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
        mse_list.append(tf.reduce_mean(mse(oriented_coords, true_coords)).numpy())
        avg_mse += mse_list[-1]

        # convert to predicted distances
        coord = tf.repeat(coord, repeats=coord.shape[1], axis=0)
        pred_dist = tf.expand_dims(tf.expand_dims(custom_norm(coord - tf.transpose(coord, perm=[1, 0, 2])), 0),
                                   -1)  # could do a reshape instead

        # compute loss
        actual_dist = tf.expand_dims(tf.expand_dims(tf.cast(distance, tf.float32), -1), 0)

        mae_list.append(tf.reduce_mean(mae(pred_dist, actual_dist)).numpy())
        avg_mae += mae_list[-1]
        avg_ssim += tf.image.ssim(img1=pred_dist, img2=actual_dist, max_val=255).numpy()

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

    save_statistics(stat_name, (mse_list, mae_list))


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
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(actual_np[:, 0], actual_np[:, 1], actual_np[:, 2], c='b')
    ax.plot(actual_np[:, 0], actual_np[:, 1], actual_np[:, 2], c='b', label="actual")
    ax.scatter(oriented_np[:, 0], oriented_np[:, 1], oriented_np[:, 2], c='r')
    ax.plot(oriented_np[:, 0], oriented_np[:, 1], oriented_np[:, 2], c='r', label="CGATr predicted")
    ax.scatter(mlp_np[:, 0], mlp_np[:, 1], mlp_np[:, 2], c='g')
    ax.plot(mlp_np[:, 0], mlp_np[:, 1], mlp_np[:, 2], c='g', label="MLP predicted")

    ax.legend()
    plt.title("Plot of predicted and actual protein structure")
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.set_zlabel("z")
    plt.show()


def load_mse_mae(model_name):
    """
    Loads mse and mae lists for a given model
    :param model_name: name of model to load statistics of
    :return: train, validation and test stats
    """

    train = load_statistics(model_name + "_train")
    val = load_statistics(model_name + "_val")
    test = load_statistics(model_name + "_test")

    return (train, val, test)


def plot_histogram(stats, metric='MSE', blocks=1):
    """
    Plots a histogram of given summary statistics.
    :param stats: array for which to plot the histogram
    :param metric: name of metric being plotted
    :param blocks: number of blocks used in projector
    :return:
    """
    bins = np.linspace(0, 100, 100)
    plt.figure(figsize=(8, 4))
    plt.hist(stats, bins, rwidth=1)
    plt.ylim((0, 450))
    plt.xlabel(metric)
    plt.ylabel('Number of occurrences')
    plt.title('Histogram of {} for {}-block CGATr projector'.format(metric, str(blocks)))
    plt.show()


def main():
    """
    Main function to run.
    :return:
    """

    model = cga_transformer_model(num_blocks=2)
    mlp = mlp_model()
    model_name = "psp_cgatr_2_block"

    # lst = load_mse_mae(model_name)
    #
    # for t in lst[:-2]:
    #     print("///////////")
    #     print(sorted(t[0])[-20:])
    #     plot_histogram(t[0], 'MSE', 2)
    #
    # model_name = "psp_cgatr_1_block"  # "psp_cgatr_1_block"  # "psp_cgatr_new_5e-4_2_block"
    #
    # lst = load_mse_mae(model_name)
    #
    # for t in lst[:-2]:
    #     print("///////////")
    #     print(sorted(t[0])[-20:])
    #     plot_histogram(t[0], 'MSE', 1)

    load_weights(model_name, model)  # demonstrates loading of model
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

    evaluate(model, lst_test, psicov_features_path, psicov_distances_path, psicov_coords_path, model_name + "_test")
    evaluate(model, lstval, deepcov_features_path, deepcov_distances_path, deepcov_coords_path, model_name + "_val")
    evaluate(model, lsttrain, deepcov_features_path, deepcov_distances_path, deepcov_coords_path, model_name +
    "_train")



if __name__ == "__main__":
    main()
