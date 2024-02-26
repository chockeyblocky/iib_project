"""
This script will evaluate a given model using a stored .pkl file containing the model.
"""

import tensorflow as tf
import numpy as np
import pickle
from nbody_model import mlp_model, cga_transformer_model
from run_experiment import create_dataset, custom_mse
import matplotlib.pyplot as plt

# set random seed
tf.random.set_seed(0)

# define path
PATH = "C:/Users/Christian/Documents/Coursework/iib_project/ga_transformer/"


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


def load_train_val_data(num_seconds):
    """
    Loads training and validation data for a given set
    :param num_seconds: string containing either 01 (meaning 0.1s) or 1 (meaning 1s)
    :return: ds_train, ds_val
    """
    ds_train = create_dataset("{}_seconds_100_steps/train".format(num_seconds))
    ds_val = create_dataset("{}_seconds_100_steps/val".format(num_seconds))

    return ds_train, ds_val


def validation_loop(model, dataset):
    """
    Full validation loop used for debugging.
    :param model: model to train
    :param dataset: training dataset
    :return:
    """

    for element in dataset:
        x = element[0]
        y = element[1]

        y_pred = model(x)

        # outputting useful information
        print(y, y_pred)
        print(y - y_pred)

        # output mse
        mse = custom_mse(y, y_pred)
        print(mse)


def load_history(model_name):
    hist_path = PATH + "model_histories/{}_history.pkl".format(model_name)
    with open(hist_path, 'rb') as f:
        history = pickle.load(f)
    return history


def plot_training_losses(model_name):
    """
    Plots losses in training using model history.
    :param model_name: string containing name of model
    :return:
    """
    # load model history
    history = load_history(model_name)

    # get loss and val loss from history
    loss = history['loss']
    val_loss = history['val_loss']

    # plot against epoch
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.yscale('log')
    plt.legend()

    plt.show()


def plot_all_val_losses():
    """
    Plots all validation losses from model histories for all trained models on the 0.1s dataset on the same axes.
    :return:
    """
    # load model histories
    base_model_name = "{}_block_nbody_cga_transformer"
    history1 = load_history(base_model_name.format("1"))
    history2 = load_history(base_model_name.format("2"))
    history3 = load_history(base_model_name.format("3"))

    # get loss and val loss from history
    val_loss1 = history1['val_loss']
    val_loss2 = history2['val_loss']
    val_loss3 = history3['val_loss']

    # plot against epoch
    epochs = range(1, len(val_loss1) + 1)
    plt.plot(epochs, val_loss1, 'b', label='1 block')
    epochs = range(1, len(val_loss2) + 1)
    plt.plot(epochs, val_loss2, 'r', label='2 block')
    epochs = range(1, len(val_loss3) + 1)
    plt.plot(epochs, val_loss3, 'g', label='3 block')

    plt.title('Validation loss for different CGA transformers')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.yscale('log')
    plt.legend()

    plt.show()


def main():
    """
    Main function to run.
    :return:
    """
    model_name = "2_block_nbody_cga_transformer"

    # create model
    model = cga_transformer_model(num_blocks=2)

    # load weights
    load_weights(model_name, model)

    # load datasets
    ds_train, ds_val = load_train_val_data("01")

    # batch datasets
    ds_train_batch = ds_train.shuffle(1000, reshuffle_each_iteration=False).batch(1)
    ds_val_batch = ds_val.shuffle(1000, reshuffle_each_iteration=False).batch(1)

    # run validation loop on first 5 elements of validation set
    validation_loop(model, ds_val_batch.take(5))

    # plot training losses
    plot_training_losses(model_name)

    # plot all val losses
    plot_all_val_losses()


if __name__ == "__main__":
    main()
