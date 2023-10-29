"""
This file will test the graph transformer informally to ensure it works.
"""

from layers.graph_transformer import GraphTransformer as gt1
import tensorflow as tf

model = gt1(
    depth = 6,
    edge_dim = 512,             # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
    with_feedforwards = True,   # whether to add a feedforward after each attention layer, suggested by literature to be needed
    rel_pos_emb = True          # set to True if the nodes are ordered, default to False
)

nodes = tf.random.normal([1, 128, 256])
edges = tf.random.normal([1, 128, 128, 512])
mask = tf.cast(tf.ones([1, 128]), tf.bool)

nodes, edges = model(nodes, edges, mask=mask)

print(nodes.shape)  # (1, 128, 256) - project to R^3 for coordinate

model = gt1(
    depth = 6,
    edge_dim = 512,
    with_feedforwards = True,
    rel_pos_emb = True,
    accept_adjacency_matrix = True  # set this to True
)

nodes = tf.random.normal([2, 128, 256])
adj_mat = tf.random.uniform((2, 128, 128), 0, 2)
mask = tf.cast(tf.ones([2, 128]), tf.bool)

nodes, edges = model(nodes, adj_mat = adj_mat, mask = mask)

print(nodes.shape) # (2, 128, 256) - project to R^3 for coordinates