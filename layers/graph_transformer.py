"""
This will contain an implementation of a graph transformer in TensorFlow.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, Identity, Embedding
from tensorflow import einsum
from einops import rearrange, repeat  # if einops incompatible, use functions written in rotary_embedding_tensorflow
from rotary_embedding_tensorflow import RotaryEmbedding, apply_rotary_emb
import numpy as np


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# normalizations


class PreNorm(Layer):
    def __init__(
            self,
            fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = LayerNormalization()

    def call(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# gated residual


class GatedResidual(Layer):
    def __init__(self):
        super().__init__()
        self.proj = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, x, res):
        gate_input = tf.concat((x, res, x - res), axis=-1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)


# attention

class Attention(Layer):
    def __init__(
            self,
            pos_emb=None,
            dim_head=64,
            heads=8,
            edge_dim=None
    ):
        super().__init__()
        self.edge_dim = edge_dim

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_emb = pos_emb

        self.to_q = Dense(inner_dim, activation=None)
        self.to_kv = Dense(inner_dim * 2, activation=None)
        self.edges_to_kv = Dense(inner_dim, activation=None)
        self.to_out = None

    def build(self, input_shape):
        self.edge_dim = default(self.edge_dim, input_shape)
        self.to_out = Dense(input_shape[-1], activation=None)

    def call(self, inputs, edges, mask=None):
        nodes = inputs

        h = self.heads

        q = self.to_q(nodes)

        temp = self.to_kv(nodes)
        k, v = tf.split(temp, [(temp.shape[-1]+1) // 2, -1], axis=-1)

        e_kv = self.edges_to_kv(edges)

        # separating out each of the heads
        q, k, v, e_kv = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h=h), (q, k, v, e_kv))

        if exists(self.pos_emb):
            freqs = self.pos_emb(tf.range(nodes.shape[1]))
            freqs = rearrange(freqs, 'n d -> () n d')
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        ek, ev = e_kv, e_kv
        k, v = map(lambda t: rearrange(t, 'b j d -> b () j d '), (k, v))
        k = k + ek
        v = v + ev

        sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') & rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            max_neg_value = -tf.experimental.numpy.finfo(sim.dtype).max
            sim = tf.where(~mask, max_neg_value, sim)

        attn = tf.nn.softmax(sim, axis=-1)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# optional feedforward

class FeedForward(Layer):
    def __init__(self, ff_mult=4):
        super().__init__()
        self.ff_mult = ff_mult

    def build(self, input_shape):
        self.dense_1 = Dense(input_shape[-1] * self.ff_mult, activation='gelu')
        self.dense_2 = Dense(input_shape[-1], activation=None)


    def call(self, inputs, *args, **kwargs):
        out = self.dense_1(inputs)
        out = self.dense_2(out)

        return out


# classes

class GraphTransformer(Layer):
    def __init__(
            self,
            depth,
            dim_head=64,
            edge_dim=None,
            heads=8,
            with_feedforwards=False,
            norm_edges=False,
            rel_pos_emb=False,
            accept_adjacency_matrix=False
    ):
        super().__init__()
        self.layers = []
        self.edge_dim = edge_dim
        self.norm = LayerNormalization()
        self.norm_edges = LayerNormalization() if norm_edges else Identity()

        self.adj_emb = Embedding(2, edge_dim) if accept_adjacency_matrix else None

        pos_emb = RotaryEmbedding(dim_head) if rel_pos_emb else None

        for _ in range(depth):
            self.layers.append([
                [
                    PreNorm(Attention(pos_emb=pos_emb, edge_dim=edge_dim, dim_head=dim_head, heads=heads)),
                    GatedResidual()
                ],
                [
                    PreNorm(FeedForward()),
                    GatedResidual()
                ] if with_feedforwards else None
            ])

    def build(self, input_shape):
        self.edge_dim = default(self.edge_dim, input_shape)

    def call(self, inputs, edges=None, mask=None, adj_mat=None):
        nodes = inputs

        batch, seq, _ = nodes.shape

        if exists(edges):
            edges = self.norm_edges(edges)

        if exists(adj_mat):
            assert adj_mat.shape == (batch, seq, seq)
            assert exists(self.adj_emb), 'accept_adjacency_matrix must be set to True'
            adj_mat = self.adj_emb(tf.cast(adj_mat, tf.int64))

        all_edges = default(edges, 0) + default(adj_mat, 0)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, all_edges, mask=mask), nodes)
            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)

        return nodes, edges
