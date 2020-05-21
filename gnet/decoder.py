"""Single-shot graph reconstruction from fixed-dimensional representation by
directly predicting adjacency matrix and edge feature.
"""

from typing import Dict
import code

import tensorflow as tf
import tensorflow_addons as tfa

class GlobalAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads:int, hidden_size:int):
    super(GlobalAttention, self).__init__()
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.scale = 1. / tf.math.sqrt(tf.cast(hidden_size, tf.float32))
    # global attention
    self.q_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    self.k_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    self.v_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    # out
    self.w_out_1 = tf.keras.layers.Dense(hidden_size)
    self.layer_norm_1 = tf.keras.layers.LayerNormalization()
    self.w_out_2 = tf.keras.layers.Dense(hidden_size)
    self.layer_norm_2 = tf.keras.layers.LayerNormalization()


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
    pre_linear_x = tf.nn.dropout(x, 0.1)

    x = self.w_out_2(x)
    x = self.layer_norm_2(x)
    x = pre_linear_x + x
    x = tfa.activations.gelu(x)
    return x


class GraphDecoder(tf.keras.layers.Layer):
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

    self.expand_w = tf.keras.layers.Dense(max_nodes * hidden_size)
    self.global_attns = [GlobalAttention(num_heads, hidden_size) for _ in range(decoder_attention_layers)]

    self.adj_w = tf.keras.layers.Dense(max_nodes, name='adjacency')    
    self.nf_w = {name: tf.keras.layers.Dense(size + 1, name=f'feature_{name}') for
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
