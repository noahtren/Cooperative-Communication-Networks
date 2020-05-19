"""Graph attention with global neighorhoods (every node sees all other nodes.)
Because the graphs are very small, each node can attend to all other nodes.
This behaves very similarly to a language transformer with multi-head attention,
where each node is treated as a token.
"""

import code

import tensorflow as tf


class GraphFeatureEmbed(tf.keras.layers.Layer):
  """Aggregate and embed node features
  """
  def __init__(self, hidden_size:int, num_feature_types:int):
    self.ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_feature_types)]


  def call(self):
    raise NotImplementedError


class GlobalAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads:int, hidden_size:int):
    super(GlobalAttention, self).__init__()
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.v_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    self.k_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    self.q_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    self.w_out = tf.keras.layers.Dense(hidden_size)
    self.scale = 1. / tf.math.sqrt(tf.cast(hidden_size, tf.float32))
  

  def call(self, inputs):
    """
    Inputs:
      x: tensor of shape [batch_size, max_nodes, node_embedding]
      num_nodes: a tensor of shape [batch_size] stating the number of nodes per
        input graph
    """
    x = inputs['x']
    num_nodes = inputs['num_nodes']

    # TODO: where to apply activations and batch normalization?

    # linear transformation of input embeddings
    Vs = [v_w(x) for v_w in self.v_ws]
    Ks = [k_w(x) for k_w in self.k_ws]
    Qs = [q_w(x) for q_w in self.q_ws]

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
    contexts = [tf.matmul(score, v) for score, v in zip(scores, Vs)]
    contexts = tf.concat(contexts, axis=-1)

    # produce new features from full context
    x = self.w_out(contexts)
    return x

