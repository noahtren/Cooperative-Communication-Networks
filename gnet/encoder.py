"""Graph attention with global neighorhoods (every node sees all other nodes.)
Because the graphs are very small, each node can attend to all other nodes.
This behaves very similarly to a language transformer with multi-head attention,
where each node is treated as a token.
"""

import code
from typing import Dict

import tensorflow as tf
import tensorflow_addons as tfa


class NodeFeatureEmbed(tf.keras.layers.Layer):
  """Aggregate and embed node features before doing global attention between
  nodes.
  """
  def __init__(self, hidden_size:int, node_feature_specs:Dict[str, int]):
    super(NodeFeatureEmbed, self).__init__()
    self.nf_w = {name: tf.keras.layers.Dense(hidden_size) for name in
      node_feature_specs.keys()}
    self.w = tf.keras.layers.Dense(hidden_size)
    self.layer_norm_1 = tf.keras.layers.LayerNormalization()
    self.w_out = tf.keras.layers.Dense(hidden_size)
    self.layer_norm_2 = tf.keras.layers.LayerNormalization()


  def call(self, inputs):
    nf = inputs['node_features']

    feature_reps = []
    for name, layer in self.nf_w.items():
      nf_rep = layer(nf[name])
      nf_rep = tfa.activations.gelu(nf_rep)
      feature_reps.append(nf_rep)
    
    feature_reps = tf.concat(feature_reps, axis=-1)
    x = self.w(feature_reps)
    x = self.layer_norm_1(x)
    pre_linear_x = x
    x = tfa.activations.gelu(x)

    x = self.w_out(x)
    x = self.layer_norm_2(x)
    x = pre_linear_x + x
    x = tfa.activations.gelu(x)
    return x


class GlobalLocalAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads:int, hidden_size:int):
    """Multi-head global self-attention (based generally on the original
    Transformer paper) combined with local graph attention (Graph Attention
    Networks.)

    ---
    Attention Is All You Need by Vaswani et al.
    https://arxiv.org/abs/1706.03762
    ---
    Graph Attention Networks by Veličković et al.
    https://arxiv.org/abs/1710.10903
    ---


    This layer incorporates a multi-head self-attention module as well as
    a feed-forward layer with the gelu activation function.

    Args:
      num_heads
      hidden_size
    """
    super(GlobalLocalAttention, self).__init__()
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.scale = 1. / tf.math.sqrt(tf.cast(hidden_size, tf.float32))
    # local
    self.local_q_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    self.local_k_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    self.local_v_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    # global
    self.global_q_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    self.global_k_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    self.global_v_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    # out
    self.w_out_1 = tf.keras.layers.Dense(hidden_size)
    self.layer_norm_1 = tf.keras.layers.LayerNormalization()
    self.w_out_2 = tf.keras.layers.Dense(hidden_size)
    self.layer_norm_2 = tf.keras.layers.LayerNormalization()
  
  def call(self, inputs):
    """
    Inputs:
      x: tensor of shape [batch_size, max_nodes, node_embedding]
      num_nodes: a tensor of shape [batch_size] stating the number of nodes per
        input graph
    """
    # TODO: may need to apply masking to bottom rows as they are used in dot-product,
    # but they shouldn't be. it seems like the only way to fix this is
    # by masking right *before* the dot product, and then again before softmaxing.

    x = inputs['x']
    start_x = x
    num_nodes = inputs['num_nodes']
    adj = inputs['adj']

    # ==================== GLOBAL ATTENTION ====================

    # linear transformation of input embeddings
    Qs = [q_w(x) for q_w in self.global_q_ws]
    Ks = [k_w(x) for k_w in self.global_k_ws]
    Vs = [v_w(x) for v_w in self.global_v_ws]

    # find alignment per attention head
    Es = [tf.matmul(q, k, transpose_b=True) for q, k in zip(Qs, Ks)]
    Es = [self.scale * e for e in Es]

    # apply masking for if num_nodes < max_nodes
    mask = tf.sequence_mask(num_nodes, maxlen=x.shape[1])[:, tf.newaxis]
    # a bunch of gross syntax that creates a broadcastable 2D mask:
    grid_max_idxs = tf.tile(tf.expand_dims(tf.math.reduce_max(
      tf.cast(tf.reshape(tf.where(
        tf.ones((x.shape[1], x.shape[1]))
      ), (x.shape[1], x.shape[1], 2)), tf.int32), axis=-1), 0),
      [num_nodes.shape[0], 1, 1])
    mask = grid_max_idxs < num_nodes[:, tf.newaxis, tf.newaxis]
    Es = [tf.where(mask, e, tf.ones_like(e) * -1e9) for e in Es]

    # calculate alignment scores and aggregate contexts
    scores = [tf.nn.softmax(e) for e in Es]
    global_context = [tf.matmul(score, v) for score, v in zip(scores, Vs)]
    global_context = tf.concat(global_context, axis=-1)

    # ==================== LOCAL ATTENTION ====================

    # linear transformation of input embeddings
    Qs = [q_w(x) for q_w in self.local_q_ws]
    Ks = [k_w(x) for k_w in self.local_k_ws]
    Vs = [v_w(x) for v_w in self.local_v_ws]

    # find alignment per attention head
    Es = [tf.matmul(q, k, transpose_b=True) for q, k in zip(Qs, Ks)]
    Es = [self.scale * e for e in Es]

    # apply masking wherever adj = 0, but also attend to self
    # (we can do this by adding the identity matrix to the adjacency matrix)
    adj = adj + tf.eye(adj.shape[1], adj.shape[1], batch_shape=[adj.shape[0]], dtype=tf.int32)
    Es = [tf.where(adj == 1, e, tf.ones_like(e) * -1e9) for e in Es]

    # calculate alignment scores and aggregate contexts
    scores = [tf.nn.softmax(e) for e in Es]
    local_context = [tf.matmul(score, v) for score, v in zip(scores, Vs)]
    local_context = tf.concat(local_context, axis=-1)

    # produce new features from full context
    context = tf.concat([global_context, local_context], axis=-1)

    x = self.w_out_1(context)
    x = self.layer_norm_1(x)
    x = start_x + x
    x = tfa.activations.gelu(x)
    pre_linear_x = x

    x = self.w_out_2(x)
    x = self.layer_norm_2(x)
    x = pre_linear_x + x
    x = tfa.activations.gelu(x)
    return x


class Encoder(tf.keras.Model):
  def __init__(self, max_nodes:int, node_feature_specs:Dict[str, int],
               hidden_size:int, attention_layers:int, num_heads:int, **kwargs):
    super(Encoder, self).__init__()
    self.embed = NodeFeatureEmbed(hidden_size, node_feature_specs)
    self.attns = [GlobalLocalAttention(num_heads, hidden_size) for _ in range(attention_layers)]


  def call(self, inputs):
    node_features = inputs['node_features']
    num_nodes = inputs['num_nodes']
    adj = inputs['adj']

    x = self.embed({'node_features': node_features})
    for attn_layer in self.attns:
      x = attn_layer({'x': x, 'num_nodes': num_nodes, 'adj': adj})
    x = tf.math.reduce_mean(x, axis=1)
    return x
