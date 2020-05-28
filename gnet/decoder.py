"""Single-shot graph reconstruction from fixed-dimensional representation by
directly predicting adjacency matrix and edge feature.
"""

from typing import Dict
import code

import tensorflow as tf
import tensorflow_addons as tfa

from ml_utils import dense_regularization


class GlobalAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads:int, hidden_size:int):
    super(GlobalAttention, self).__init__()
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.scale = 1. / tf.math.sqrt(tf.cast(hidden_size, tf.float32))
    # global attention
    self.q_ws = [tf.keras.layers.Dense(hidden_size, **dense_regularization) for _ in range(num_heads)]
    self.k_ws = [tf.keras.layers.Dense(hidden_size, **dense_regularization) for _ in range(num_heads)]
    self.v_ws = [tf.keras.layers.Dense(hidden_size, **dense_regularization) for _ in range(num_heads)]
    # out
    self.w_out_1 = tf.keras.layers.Dense(hidden_size, **dense_regularization)
    self.layer_norm_1 = tf.keras.layers.LayerNormalization()
    self.w_out_2 = tf.keras.layers.Dense(hidden_size, **dense_regularization)
    self.layer_norm_2 = tf.keras.layers.LayerNormalization()
    self.w_out_3 = tf.keras.layers.Dense(hidden_size, **dense_regularization)
    self.layer_norm_3 = tf.keras.layers.LayerNormalization()

  def call(self, x):
    start_x = tf.nn.dropout(x, 0.1)
    
    # ==================== GLOBAL ATTENTION ====================

    # linear transformation of input embeddings
    Qs = [q_w(x) for q_w in self.q_ws]
    Ks = [k_w(x) for k_w in self.k_ws]
    Vs = [v_w(x) for v_w in self.v_ws]

    # find alignment per attention head
    Es = [tf.matmul(q, k, transpose_b=True) for q, k in zip(Qs, Ks)]
    Es = [self.scale * e for e in Es]

    # calculate alignment scores and aggregate contexts
    scores = [tf.nn.softmax(e) for e in Es]
    context = [tf.matmul(score, v) for score, v in zip(scores, Vs)]
    context = tf.concat(context, axis=-1)

    # produce new features from full context
    x = self.w_out_1(context)
    x = self.layer_norm_1(x)
    x = start_x + x
    x = tfa.activations.gelu(x)
    res_x = x
    res_x_2 = x

    x = self.w_out_2(x)
    x = self.layer_norm_2(x)
    x = res_x + x
    x = tfa.activations.gelu(x)
    res_x = x

    x = self.w_out_3(x)
    x = self.layer_norm_3(x)
    x = res_x_2 + res_x + x
    x = tfa.activations.gelu(x)
    return x


class GraphDecoder(tf.keras.Model):
  def __init__(self, num_heads:int, hidden_size:int, max_nodes:int,
               node_feature_specs:Dict[str, int], decoder_attention_layers:int, **kwargs):
    """Simple graph reconstruction with dense feed-forward neural network based
    generally on the GraphVAE paper. I also added global self-attention as a
    refining step which improves accuracy.

    ---
    GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders,
    Simonovsky et al.
    https://arxiv.org/abs/1802.03480
    ---

    Args:
      num_heads
      hidden_size
      max_nodes
      node_feature_specs: a dict of integers, each mapping a node feature name
        to its dimensionality. Example: {'ord': 4}
    """
    super(GraphDecoder, self).__init__()
    self.max_nodes = max_nodes

    # expanding from hidden state
    self.expand_w = tf.keras.layers.Dense(max_nodes * hidden_size, **dense_regularization)
    # pos embeds
    self.pos_embeds = [tf.keras.layers.Dense(128, **dense_regularization) for _ in range(3)]
    self.pos_norms = [tf.keras.layers.LayerNormalization() for _ in range(3)]
    self.combine_pos = tf.keras.layers.Dense(hidden_size, **dense_regularization)
    self.combine_pos_norm = tf.keras.layers.LayerNormalization()
    # attention
    self.global_attns = [GlobalAttention(num_heads, hidden_size) for _ in range(decoder_attention_layers)]

    self.adj_w = tf.keras.layers.Dense(max_nodes, name='adjacency', **dense_regularization)
    self.nf_w = {name: tf.keras.layers.Dense(size + 1, name=f'feature_{name}', **dense_regularization) for
      name, size in node_feature_specs.items()}

    self.scale = 1. / tf.math.sqrt(tf.cast(hidden_size, tf.float32))


  def call(self, Z):
    """
    Inputs:
      Z: tensor of shape [batch_size, node_embedding], which is a
    fixed-dimensional representation of a graph that will be reconstructed to
    its nodes.
    """
    batch_size = Z.shape[0]

    # expand fixed-dimensional representation
    expanded_x = self.expand_w(Z)
    expanded_x = tfa.activations.gelu(expanded_x)
    x = tf.reshape(expanded_x, [batch_size, self.max_nodes, -1])

    # Positional embedding, simply as a hint so that nodes know
    # the index of other nodes when passing messages
    pos = tf.tile(tf.range(x.shape[1])[tf.newaxis], [x.shape[0], 1])
    pos = tf.cast(pos, tf.float32)[..., tf.newaxis] / x.shape[1]
    for pos_embed, pos_norm in zip(self.pos_embeds, self.pos_norms):
      pos = pos_embed(pos)
      pos = pos_norm(pos)

    x = tf.concat([pos, x], axis=-1)
    x = self.combine_pos(x)
    x = self.combine_pos_norm(x)

    # local and global attention
    for attn_layer in self.global_attns:
      x = attn_layer(x)

    # predict adjacencies
    adj_out = self.adj_w(x)
    adj_out = self.scale * adj_out
    adj_out = tf.nn.sigmoid(adj_out)

    # predict node features
    nf_out = {}
    for name in self.nf_w.keys():
      w_layer = self.nf_w[name]
      nf_pred = w_layer(x)
      nf_pred = self.scale * nf_pred
      nf_pred = tf.nn.softmax(nf_pred)
      nf_out[name] = nf_pred

    return adj_out, nf_out
