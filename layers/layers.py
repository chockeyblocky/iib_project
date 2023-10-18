"""
This contains the custom layers implemented using tfga infrastructure.
"""
from typing import List, Union

import tensorflow as tf
from tensorflow.keras import (activations, constraints, initializers, layers,
                              regularizers)
from tensorflow.keras.utils import register_keras_serializable

from tfga.blades import BladeKind
from tfga.layers import GeometricAlgebraLayer


class RotorConv1D(GeometricAlgebraLayer):
    """
    This is a convolution layer as described in "Geometric Clifford Algebra
    Networks". It uses a weighted sandwich product with rotors in the kernel.


    """

    pass
