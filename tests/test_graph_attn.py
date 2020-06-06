import tensorflow as tf

from ccn.graph_models import GlobalAttention

def test_global_attn():
  x = tf.random.normal((2, 7, 512))
  global_attn = GlobalAttention(num_heads=5, graph_hidden_size=512)
  y = global_attn(x)
