"""
This script will evaluate a given model using a stored .pkl file containing the model.
"""

import tensorflow as tf
import numpy as np
import pickle

# set random seed
tf.random.set_seed(0)