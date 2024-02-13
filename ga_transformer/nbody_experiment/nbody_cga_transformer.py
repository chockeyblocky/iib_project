"""
This will contain the class implementation of the CGA transformer used in the NBody problem
"""

import tensorflow as tf
from layers.layers import EquivariantTransformerBlock


class CGATransformer(tf.keras.Model):
    """
    This will define a transformer which uses CGA to make geometric predictions for an n-body dynamics problem.
    """
    def __init__(self, geometric_algebra, n_bodies):
        """
        :param geometric_algebra: Instance of CGA
        :param n_bodies: number of bodies to make predictions about
        """
        super().__init__()
        # define transformer block
        self.transformer_block = EquivariantTransformerBlock(algebra=geometric_algebra, units_per_head=5,
                                                             hidden_units=10, heads=5, output_units=n_bodies)
        # define algebra instance
        self.ga = geometric_algebra

    def embed_points(self, points):
        """
        Embeds points into CGA using the scheme described in "Guide to Geometric Algebra in Practice" (Dorst, Lasenby).
        :param points: points tensor to be embedded.
        :param ga: Geometric Algebra to be used
        :return: points as tensors in CGA
        """
        # get squared sum of vector elements for each point
        points_norm = tf.reduce_sum(points ** 2, axis=-1)

        # get n and n_bar from ga instance - expecting signature [1, 1, 1, 1, -1]
        # bug in tfga - assertion causes failure upon trying to train when doing ga.ex - this is why method is
        # convoluted
        n = self.ga.blade_mvs[4] + self.ga.blade_mvs[5]
        n_bar = self.ga.blade_mvs[4] - self.ga.blade_mvs[5]

        # get basic vector embedding
        cga_vecs = self.ga.from_tensor(points, self.ga.get_blade_indices_of_degree(1)[:3])

        # sum and return tf tensor corresponding to final embedding
        return cga_vecs + 0.5 * tf.einsum("j,...i->...ij", n, points_norm) \
            - 0.5 * tf.reshape(tf.tile(n_bar, tf.reshape(tf.size(points_norm), [1])), tf.shape(cga_vecs))

    def embed_velocities(self, points, vels):
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
        n = self.ga.blade_mvs[4] + self.ga.blade_mvs[5]

        # get basic vector embedding
        cga_vecs = self.ga.from_tensor(vels, self.ga.get_blade_indices_of_degree(1)[:3])

        # sum and return tf tensor corresponding to final embedding
        return self.ga.dual(cga_vecs + 0.5 * tf.einsum("j,...i->...ij", n, inner_prod))  # TODO CHECK THIS WORKS

    def cga_embed_inputs(self, x):
        """
        Embeds inputs as tfga tensors in CGA as in "Guide to Geometric Algebra in Practice" (Dorst, Lasenby).
        :param x: input tensor containing (mass, point, velocity) along final dimension
        :param ga: Geometric Algebra to be used
        :return: tensor embedded into CGA
        """
        mass_mv = self.ga.from_scalar(x[..., 0])
        points_mv = self.embed_points(x[..., 1:4])
        vel_mv = self.embed_velocities(x[..., 1:4], x[..., 4:7])

        return mass_mv + points_mv + vel_mv

    def cga_extract_outputs(self, y):
        """
        Extracts 3D vector outputs corresponding to training dataset (x, y, z).
        :param y: output multivector
        :param ga: Geometric Algebra to be used
        :return: tensor extracted from CGA
        """
        # extract tensor from tfga by getting e0, e1, e2 parts
        return tf.gather(y, self.ga.get_blade_indices_of_degree(1)[:3], axis=-1)

    def call(self, inputs):
        embedded_inputs = self.cga_embed_inputs(inputs)
        transformed_inputs = self.transformer_block(embedded_inputs)
        return self.cga_extract_outputs(transformed_inputs)