"""
This contains the functions used for creating models for protein structure prediction.
"""

import tensorflow as tf
from tfga import GeometricAlgebra
from protein_cga_transformer import ProteinCGATransformer


def cga_transformer_model(num_blocks=1, num_edge_layers=3, num_features=27):
    """
    Creates a transformer which uses CGA to solve geometric problems.
    :param num_blocks: number of transformer blocks to use
    :param num_edge_layers: number of edge layers in GT input
    :param num_features: number of features for each node in GT input
    :return: CGA transformer
    """
    # define geometric algebra instance - CGA
    cga = GeometricAlgebra(metric=[1, 1, 1, 1, -1])

    # instantiate model
    model = ProteinCGATransformer(geometric_algebra=cga, n_hidden_multivectors=9, n_output_multivectors=1,
                                  n_blocks=num_blocks, n_edges=num_edge_layers)

    # call model on dummy input - effectively builds all layers - allows saving/loading
    dummy_nodes = tf.ones([1, 100, num_features])
    dummy_edges = tf.ones([1, 100, 100, num_edge_layers])
    dummy_mask = tf.cast(tf.ones([1, 100]), tf.bool)

    output = model(dummy_nodes, dummy_edges, dummy_mask)

    return model
