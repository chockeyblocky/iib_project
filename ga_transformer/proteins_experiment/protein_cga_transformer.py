"""
This will contain the class implementation of the CGA transformer used in the NBody problem
"""

import tensorflow as tf
from layers.layers import EquivariantTransformerBlock, EquivariantLinear
from layers.graph_transformer import GraphTransformer


class ProteinCGATransformer(tf.keras.Model):
    """
    This will define a model which uses a graph transformer and CGATr to make geometric predictions for an n-body
    dynamics problem.
    """

    def __init__(self, geometric_algebra, n_hidden_multivectors, n_output_multivectors, n_blocks, n_edges):
        """
        :param geometric_algebra: Instance of CGA
        :param n_hidden_multivectors: number of hidden multivectors to process with CGATr
        :param n_output_multivectors: desired number of output multivectors
        :param n_blocks: number of transformer blocks to add to the network
        :param n_edges: number of layers of edges connecting each node
        """
        super().__init__()
        # define graph transformer
        self.gt = GraphTransformer(
            depth=3,
            heads=4,
            edge_dim=n_edges,
            with_feedforwards=True,
            rel_pos_emb=True)

        # define CGATr net
        self.transformer_net = tf.keras.Sequential(name='transformer_blocks')
        for i in range(n_blocks):
            self.transformer_net.add(
                EquivariantTransformerBlock(algebra=geometric_algebra, units_per_head=2,
                                            hidden_units=10, heads=5, output_units=n_hidden_multivectors,
                                            non_linear_activation='sigmoid'))

        self.output_linear = EquivariantLinear(algebra=geometric_algebra, units=n_output_multivectors)

        # define algebra instance
        self.ga = geometric_algebra

        # record number of hidden multivectors
        self.n_hidden_multivectors = n_hidden_multivectors

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

    def cga_embed_inputs(self, x):
        """
        Embeds inputs as tfga tensors in CGA as in "Guide to Geometric Algebra in Practice" (Dorst, Lasenby).
        :param x: input tensor containing (mass, point, velocity) along final dimension
        :param ga: Geometric Algebra to be used
        :return: tensor embedded into CGA
        """
        points_mv = self.embed_points(x)

        return points_mv

    def cga_extract_outputs(self, y):
        """
        Extracts 3D vector outputs corresponding to training dataset (x, y, z).
        :param y: output multivector
        :param ga: Geometric Algebra to be used
        :return: tensor extracted from CGA
        """
        # extract tensor from tfga by getting e0, e1, e2 parts
        return tf.gather(y, self.ga.get_blade_indices_of_degree(1)[:3], axis=-1)

    def call(self, nodes, edges, mask):
        x, edges_new = self.gt(nodes, edges, mask=mask)
        x = tf.reshape(x, shape=[-1, self.n_hidden_multivectors, 3])
        x = self.cga_embed_inputs(x)
        x = self.transformer_net(x)
        x = self.output_linear(x)
        x = self.cga_extract_outputs(x)
        return tf.reshape(x, shape=[1, -1, 3])