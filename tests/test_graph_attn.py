import tensorflow as tf

from gnet.encoder import GlobalAttention

def test_global_attn():
  x = tf.random.normal((2, 7, 512))
  num_nodes = tf.convert_to_tensor([4, 6])
  global_attn = GlobalAttention(num_heads=5, hidden_size=512)
  y = global_attn(inputs={
    'x': x,
    'num_nodes': num_nodes
  })
