"""Graph attention with global neighorhoods (every node sees all other nodes.)
Because the graphs are very small, each node can attend to all other nodes.
This behaves very similarly to a language transformer with multi-head attention,
where each node is treated as a token.
"""

import tensorflow as tf


class GraphAttention:
  pass
