from typing import Dict
import code

import tensorflow as tf
import tensorflow_addons as tfa

from .ml_utils import dense_regularization


# TODO: consider adding auxiliary "certainty" prediction when using
# Wassertein loss training. Otherwise, MSE loss seems like a good idea


class NodeFeatureEmbed(tf.keras.Model):
  """Aggregate and embed node features before doing global attention between
  nodes.
  """
  def __init__(self, graph_hidden_size:int, node_feature_specs:Dict[str, int]):
    super(NodeFeatureEmbed, self).__init__()
    self.nf_w = {name: tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for name in
      node_feature_specs.keys()}
    self.w = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
    self.layer_norm_1 = tf.keras.layers.LayerNormalization()
    self.w_out = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
    self.layer_norm_2 = tf.keras.layers.LayerNormalization()


  def call(self, inputs):
    nf = inputs['node_features']

    feature_reps = []
    for name, layer in self.nf_w.items():
      nf_rep = layer(nf[name])
      nf_rep = tf.nn.swish(nf_rep)
      feature_reps.append(nf_rep)
    
    feature_reps = tf.concat(feature_reps, axis=-1)
    x = self.w(feature_reps)
    x = self.layer_norm_1(x)
    pre_linear_x = x
    x = tf.nn.swish(x)

    x = self.w_out(x)
    x = self.layer_norm_2(x)
    x = pre_linear_x + x
    x = tf.nn.swish(x)
    return x


class GlobalLocalAttention(tf.keras.Model):
  def __init__(self, num_heads:int, graph_hidden_size:int, max_nodes:int):
    """Multi-head global self-attention (based generally on the original
    Transformer paper) combined with local graph attention (Graph Attention
    Networks.)

    Attention Is All You Need by Vaswani et al.
    https://arxiv.org/abs/1706.03762

    Graph Attention Networks by Veličković et al.
    https://arxiv.org/abs/1710.10903

    This layer incorporates a multi-head self-attention module as well as
    a feed-forward layer with the swish activation function.
    """
    super(GlobalLocalAttention, self).__init__()
    self.max_nodes = max_nodes
    self.num_heads = num_heads
    self.graph_hidden_size = graph_hidden_size
    self.scale = 1. / tf.math.sqrt(tf.cast(graph_hidden_size, tf.float32))
    # local
    self.local_q_ws = [tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for _ in range(num_heads)]
    self.local_k_ws = [tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for _ in range(num_heads)]
    self.local_v_ws = [tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for _ in range(num_heads)]
    # global
    self.global_q_ws = [tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for _ in range(num_heads)]
    self.global_k_ws = [tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for _ in range(num_heads)]
    self.global_v_ws = [tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for _ in range(num_heads)]
    # out
    self.w_out_1 = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
    self.layer_norm_1 = tf.keras.layers.LayerNormalization()
    self.w_out_2 = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
    self.layer_norm_2 = tf.keras.layers.LayerNormalization()
    self.w_out_3 = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
    self.layer_norm_3 = tf.keras.layers.LayerNormalization()

  def call(self, inputs):
    """
    Inputs:
      x: tensor of shape [batch_size, max_nodes, node_embedding]
      num_nodes: a tensor of shape [batch_size] stating the number of nodes per
        input graph
    """
    x = inputs['x']
    start_x = tf.nn.dropout(x, 0.1)
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
    mask = tf.sequence_mask(num_nodes, maxlen=self.max_nodes)[:, tf.newaxis]
    # a bunch of gross syntax that creates a broadcastable 2D mask:
    r = tf.range(self.max_nodes)
    grid_max_idxs = tf.tile(
      tf.expand_dims(
        tf.math.maximum(r[tf.newaxis], r[:, tf.newaxis]), 0),
      [num_nodes.shape[0], 1, 1]
    )
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
    x = tf.nn.swish(x)
    res_x = x

    x = self.w_out_2(x)
    x = self.layer_norm_2(x)
    x = res_x + x
    x = tf.nn.swish(x)
    res_x = x

    x = self.w_out_3(x)
    x = self.layer_norm_3(x)
    x = res_x + x
    x = tf.nn.swish(x)
    return x


class GraphEncoder(tf.keras.Model):
  def __init__(self, max_nodes:int, node_feature_specs:Dict[str, int],
               graph_hidden_size:int, encoder_attention_layers:int, num_heads:int, **kwargs):
    super(GraphEncoder, self).__init__()
    self._name = 'g_encoder'
    self.max_nodes = max_nodes
    self.embed = NodeFeatureEmbed(graph_hidden_size, node_feature_specs)
    self.attns = [GlobalLocalAttention(num_heads, graph_hidden_size, max_nodes) for _ in range(encoder_attention_layers)]


  def call(self, inputs, debug=False):
    node_features = inputs['node_features']
    num_nodes = inputs['num_nodes']
    adj = inputs['adj']

    x = self.embed({'node_features': node_features})
    for attn_layer in self.attns:
      x = attn_layer({'x': x, 'num_nodes': num_nodes, 'adj': adj})
    # pad based on num_nodes
    output_mask = tf.sequence_mask(num_nodes, maxlen=self.max_nodes)
    x = tf.where(output_mask[..., tf.newaxis], x, tf.zeros_like(x))
    x = tf.math.reduce_mean(x, axis=1)
    # now scale up based on how many nodes were set to 0
    x = x * tf.cast(5 / num_nodes, tf.float32)[:, tf.newaxis]
    return x


class GlobalAttention(tf.keras.Model):
  def __init__(self, num_heads:int, graph_hidden_size:int):
    super(GlobalAttention, self).__init__()
    self.num_heads = num_heads
    self.graph_hidden_size = graph_hidden_size
    self.scale = 1. / tf.math.sqrt(tf.cast(graph_hidden_size, tf.float32))
    # global attention
    self.q_ws = [tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for _ in range(num_heads)]
    self.k_ws = [tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for _ in range(num_heads)]
    self.v_ws = [tf.keras.layers.Dense(graph_hidden_size, **dense_regularization) for _ in range(num_heads)]
    # out
    self.w_out_1 = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
    self.layer_norm_1 = tf.keras.layers.LayerNormalization()
    self.w_out_2 = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
    self.layer_norm_2 = tf.keras.layers.LayerNormalization()
    self.w_out_3 = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
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
    x = tf.nn.swish(x)
    res_x = x
    res_x_2 = x

    x = self.w_out_2(x)
    x = self.layer_norm_2(x)
    x = res_x + x
    x = tf.nn.swish(x)
    res_x = x

    x = self.w_out_3(x)
    x = self.layer_norm_3(x)
    x = res_x_2 + res_x + x
    x = tf.nn.swish(x)
    return x


class GraphDecoder(tf.keras.Model):
  def __init__(self, num_heads:int, graph_hidden_size:int, max_nodes:int,
               node_feature_specs:Dict[str, int], decoder_attention_layers:int, **kwargs):
    """Simple graph reconstruction with dense feed-forward neural network based
    generally on the GraphVAE paper. Added global self-attention as a refining
    step which improves accuracy.

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
    self._name = 'g_decoder'
    self.max_nodes = max_nodes

    # expanding from hidden state
    self.expand_w = tf.keras.layers.Dense(max_nodes * graph_hidden_size, **dense_regularization)
    # pos embeds
    self.pos_embeds = [tf.keras.layers.Dense(128, **dense_regularization) for _ in range(3)]
    self.pos_norms = [tf.keras.layers.LayerNormalization() for _ in range(3)]
    self.combine_pos = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
    self.combine_pos_norm = tf.keras.layers.LayerNormalization()
    # attention
    self.global_attns = [GlobalAttention(num_heads, graph_hidden_size) for _ in range(decoder_attention_layers)]

    self.adj_w = tf.keras.layers.Dense(max_nodes, name='adjacency', **dense_regularization)
    self.nf_w = {name: tf.keras.layers.Dense(size + 1, name=f'feature_{name}', **dense_regularization) for
      name, size in node_feature_specs.items()}

    self.scale = 1. / tf.math.sqrt(tf.cast(graph_hidden_size, tf.float32))
    self.graph_hidden_size = graph_hidden_size

  def call(self, Z, debug=False):
    """
    Inputs:
      Z: tensor of shape [batch_size, node_embedding], which is a
    fixed-dimensional representation of a graph that will be reconstructed to
    its nodes.
    """
    batch_size = Z.shape[0]

    # expand fixed-dimensional representation
    expanded_x = self.expand_w(Z)
    expanded_x = tf.nn.swish(expanded_x)
    x = tf.reshape(expanded_x, [batch_size, self.max_nodes, self.graph_hidden_size])

    # Positional embedding, simply as a hint so that nodes know the index of
    # other nodes when passing messages
    # (don't have to use sine for positional embeddings since number of nodes
    # is always constant, unlike number of words in a sentence)
    pos = tf.tile(tf.range(self.max_nodes)[tf.newaxis], [x.shape[0], 1])
    pos = tf.cast(pos, tf.float32)[..., tf.newaxis] / self.max_nodes
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
