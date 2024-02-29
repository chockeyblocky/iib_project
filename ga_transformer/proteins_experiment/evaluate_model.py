"""
This script evaluates models trained using the protein structure prediction dataset.
"""

import tensorflow as tf
import pickle
from proteins_model import cga_transformer_model

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


def main():
    """
    Main function to run.
    :return:
    """

    model = cga_transformer_model()
    load_weights("test", model)  # demonstrates loading of model


if __name__ == "__main__":
    main()
