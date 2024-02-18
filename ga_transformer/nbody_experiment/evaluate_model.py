"""
This script will evaluate a given model using a stored .pkl file containing the model.
"""

import tensorflow as tf
import numpy as np
import pickle
from nbody_model import mlp_model, cga_transformer_model
from run_experiment import create_dataset, custom_mse

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


def main():
    """
    Main function to run.
    :return:
    """
    model_name = "nbody_cga_transformer"

    # create model
    model = cga_transformer_model()

    # load weights
    load_weights(model_name, model)

    # load datasets
    ds_train, ds_val = load_train_val_data("01")

    # batch datasets
    ds_train_batch = ds_train.shuffle(1000, reshuffle_each_iteration=False).batch(1)
    ds_val_batch = ds_val.shuffle(1000, reshuffle_each_iteration=False).batch(1)

    # run validation loop on first 5 elements of validation set
    validation_loop(model, ds_val_batch.take(5))


if __name__ == "__main__":
    main()
