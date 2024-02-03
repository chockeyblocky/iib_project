"""
This contains the functions which define the GA transformer model.
"""

import tensorflow as tf
import numpy as np
import tfga
from tfga import GeometricAlgebra
from run_experiment import load_data, PATH


def embed_points(points, ga):
    """
    Embeds points into CGA using the scheme described in "Guide to Geometric Algebra in Practice" (Dorst, Lasenby).
    :param points: points tensor to be embedded.
    :param ga: Geometric Algebra to be used
    :return: points as tensors in CGA
    """
    # get squared sum of vector elements for each point
    points_norm = tf.reduce_sum(points ** 2, axis=-1)

    # get n and n_bar from ga instance - expecting signature [1, 1, 1, 1, -1]
    n = ga.e3 + ga.e4
    n_bar = ga.e3 - ga.e4

    # get basic vector embedding
    cga_vecs = ga.from_tensor(points, ga.get_blade_indices_of_degree(1)[:3])

    # sum and return tf tensor corresponding to final embedding
    return cga_vecs + 0.5 * tf.einsum("j,...i->...ij", n, points_norm) \
        - 0.5 * tf.reshape(tf.tile(n_bar, tf.reshape(tf.size(points_norm), [1])), tf.shape(cga_vecs))


def embed_velocities(points, vels, ga):
    """
    Embeds points into CGA using the scheme described in "Guide to Geometric Algebra in Practice" (Dorst, Lasenby).
    :param points: points tensor corresponding to velocities vector
    :param vels: velocities tensor to be embedded.
    :param ga: Geometric Algebra to be used
    :return: points as tensors in CGA
    """
    # get inner product of vectors and points
    inner_prod = tf.reduce_sum(points * vels, axis=-1)

    # get n from ga instance - expecting signature [1, 1, 1, 1, -1]
    n = ga.e3 + ga.e4

    # get basic vector embedding
    cga_vecs = ga.from_tensor(vels, ga.get_blade_indices_of_degree(1)[:3])

    # sum and return tf tensor corresponding to final embedding
    return cga_vecs + 0.5 * tf.einsum("j,...i->...ij", n, inner_prod)


def cga_embed_inputs(x, ga):
    """
    Embeds inputs as tfga tensors in CGA as in "Guide to Geometric Algebra in Practice" (Dorst, Lasenby).
    :param x: input tensor containing (mass, point, velocity) along final dimension
    :param ga: Geometric Algebra to be used
    :return: tensor embedded into CGA
    """
    mass_mv = ga.from_scalar(x[..., 0])
    points_mv = embed_points(x[..., 1:4], ga)
    vel_mv = embed_velocities(x[..., 1:4], x[..., 4:7], ga)

    # TODO: MAYBE INSTEAD OF ADDING THESE TOGETHER, PASS VELS AND POINTS AS KEYS AND QUERIES INTO FIRST ATTN. LAYER
    return mass_mv + points_mv + vel_mv


def cga_extract_outputs(y, ga):
    """
    Extracts 3D vector outputs corresponding to training dataset (x, y, z).
    :param y: output multivector
    :param ga: Geometric Algebra to be used
    :return: tensor extracted from CGA
    """
    # extract tensor from tfga by getting e0, e1, e2 parts
    return tf.gather(y, ga.get_blade_indices_of_degree(1)[:3], axis=-1)


if __name__ == "__main__":
    full_path = PATH + "datasets/" + "train" + ".npz"
    x, y, trajectories = load_data(full_path)
    ga = GeometricAlgebra(metric=[1, 1, 1, 1, -1])
    a = embed_points(x[..., 1:4], ga)
    b = embed_velocities(x[..., 1:4], x[..., 4:7], ga)
    c = cga_embed_inputs(x, ga)
    d = cga_extract_outputs(c, ga)
    print(x.shape, c.shape)
    print(y.shape, d.shape)
    print(c[0, 0])
    print(c[0, 1])
    print(ga.inner_prod(c[0, 0], c[0, 1]))
    print(ga.geom_prod(c[0, 0], ga.reversion(c[0, 1])))
    print(ga.num_blades)
