import code

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
  def __init__(self, num_heads:int=1, embedding_size:int=1024, max_query_len=12):
    super(Attention, self).__init__()
    self.num_heads = num_heads
    self.max_query_len = max_query_len
    self.attn_ws = [tf.keras.layers.Dense(embedding_size) for _ in range(num_heads)]
    self.out_w = tf.keras.layers.Dense(embedding_size)


  def call(self, inputs):
    """Perform general dot-product attention
    Inputs:
      query: all preceding vectors to attend to when calculating new embedding.
        size: [batch_size, max_query_len, embedding_size]
      value: current value being used to find context.
        size: [batch_size, 1, embedding_size]
      num_to_attend:
        size: []
    """
    query = inputs['query']
    value = inputs['value']
    num_to_attend = inputs['num_to_attend']

    assert query.shape[0] == value.shape[0], f'query and value must have same number of batches'
    assert num_to_attend.shape[0] == query.shape[0], f'num_to_attend should have same number of batches as query/value tensors'

    contexts = []
    for i in range(self.num_heads):
      query = self.attn_ws[i](query)
      e = tf.matmul(value, query, transpose_b=True)
      e = tf.nn.swish(e)
      mask = tf.sequence_mask(num_to_attend, maxlen=self.max_query_len)[:, tf.newaxis]
      e = tf.where(mask, e, tf.ones_like(e) * -1e9)
      scores = tf.nn.softmax(e)
      # sum all query embeddings according to attention scores
      context = tf.matmul(scores, query)
      contexts.append(context)

    context = tf.concat(contexts + [value], axis=-1)
    x = self.out_w(context)
    x = tf.nn.swish(x)
    return x


if __name__ == "__main__":
  value = tf.random.normal([8, 1, 512])
  query = tf.random.normal([8, 12, 512])
  num_to_attend = tf.convert_to_tensor([5] * 8)
  attn = Attention(num_heads=4, embedding_size=512, max_query_len=12)
  x = attn({
      'query': query,
      'value': value,
      'num_to_attend': num_to_attend
  })
  print(f'Input shape: {value.shape}')
  print(f'Output shape: {x.shape}')
