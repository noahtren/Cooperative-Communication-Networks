"""Single-shot graph reconstruction from fixed-dimensional representation by
directly predicting adjacency matrix and edge feature.
"""

from typing import Dict

import tensorflow as tf

class GraphDecoder(tf.keras.layers.Layer):
  def __init__(self, hidden_size:int, max_nodes:int,
               node_feature_specs:Dict[str, int], **kwargs):
    """Simple graph reconstruction with dense feed-forward neural network based
    generally on the GraphVAE paper.

    ---
    GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders,
    Simonovsky et al.
    https://arxiv.org/abs/1802.03480
    ---

    Args:
      hidden_size
      max_nodes
      node_feature_specs: a dict of integers, each mapping a node feature name
        to its dimensionality. Example: {'ord': 4}
    """
    super(GraphDecoder, self).__init__()
    self.adj_w = tf.keras.layers.Dense(max_nodes)
    self.nf_w = {name: tf.keras.layers.Dense(size) for
      name, size in node_feature_specs.items()}
    self.scale = 1. / tf.math.sqrt(tf.cast(hidden_size, tf.float32))

  def call(self, inputs):
    """
    Inputs:
      x: tensor of shape [batch_size, max_nodes, node_embedding]
    """
    x = inputs['x']

    # predict adjacencies
    adj = self.adj_w(x)
    adj = self.scale * adj
    adj = tf.nn.softmax(adj)

    # predict node features
    nf = {}
    for name, layer in self.nf_w.items():
      nf_pred = layer(x)
      nf_pred = self.scale * nf_pred
      nf_pred = tf.nn.softmax(nf_pred)
      nf[name] = nf_pred
    
    return adj, nf_pred
