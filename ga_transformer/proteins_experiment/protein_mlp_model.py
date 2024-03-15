"""
This contains the class defining an MLP for protein structure prediction with a graph transformer backbone.
"""

import tensorflow as tf
from layers.graph_transformer import GraphTransformer


class MLPModel(tf.keras.Model):
    def __init__(self, edge_n):
        super().__init__()
        self.gt = GraphTransformer(
            depth=3,
            heads=4,
            edge_dim=edge_n,
            with_feedforwards=True,
            rel_pos_emb=True)
        self.dense_1 = tf.keras.layers.Dense(9, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(3, activation=None)
        # add in shape reduction layers to make shapes match

    def call(self, nodes, edges, mask):
        x, edges_new = self.gt(nodes, edges, mask=mask)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x - tf.reduce_mean(x, axis=1)
