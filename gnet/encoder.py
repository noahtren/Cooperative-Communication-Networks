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

    code.interact(local={**locals(), **globals()})

    # find alignment per attention head
    Es = [tf.matmul(q, k, transpose_b=True) for q, k in zip(Qs, Ks)]
    Es = [self.scale * e for e in Es]

    # apply masking for if num_nodes < max_nodes
    mask = tf.sequence_mask(num_nodes, maxlen=x.shape[1])[:, tf.newaxis]
    grid_max_idxs = tf.math.reduce_max(
      tf.cast(tf.where(
        tf.ones((x.shape[1], x.shape[1]))
      ), tf.int32), axis=-1)
    mask = tf.where(grid_max_idxs < num_nodes,
      tf.ones((x.shape[1], x.shape[1]), dtype=tf.bool),
      tf.zeros((x.shape[1], x.shape[1]), dtype=tf.bool)
    )
    mask = tf.ones((num_nodes,))
    Es = [tf.where(mask, e, tf.ones_like(e) * -1e9) for e in Es]

    # calculate alignment scores and aggregate contexts
    scores = [tf.nn.softmax(e) for e in Es]
    contexts = [tf.matmul(score, q) for score, q in zip(scores, Qs)]
    contexts = tf.concat(contexts, axis=-1)

    # produce new features from full context
    x = self.w_out(contexts)
    return x


if __name__ == "__main__":
  x = tf.random.normal((2, 7, 512))
  num_nodes = tf.convert_to_tensor([4, 6])
  global_attn = GlobalAttention(num_heads=5, hidden_size=512)
  y = global_attn(inputs={
    'x': x,
    'num_nodes': num_nodes
  })
