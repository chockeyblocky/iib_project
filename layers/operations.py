"""This will contain operations used to support the layers constructed in layers.py."""

from typing import List, Union
import tensorflow as tf
from clifford.g3c import *
from math import sqrt
import numpy as np


# functions to convert quaternions coefficients into to GA rotors
def q2S(*args):
    '''
    convert tuple of quaternion coefficients to a spinor'''
    q = args
    return q[0] + q[1] * e13 + q[2] * e23 + q[3] * e12


# From Euclidean to 1D Up CGA.
# function implementing Eq. 6 (convert a vector in Euclidean space into a rotor
# in spherical space)
def translation_rotor(a, L=200):
    Ta = (L + a * e4) / (sqrt(L ** 2 + a ** 2))
    return Ta


# From Euclidean to 1D Up CGA. function implementing the Eq. 10 (X = f(x))
def up1D(x, L=200):
    Y = (2 * L / (L ** 2 + x ** 2)) * x + ((L ** 2 - x ** 2) / (L ** 2 + x ** 2)) * e4
    return Y


# From 1D Up CGA to Euclidean. function implementing the inverse of Eq. 10 (x = f^{-1}(X))
def down1D(Y, L=200):
    x = (L / (1 + Y * e4)) * ((Y | e1) * e1 + (Y | e2) * e2 + (Y | e3) * e3)
    return x
