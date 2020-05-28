import tensorflow as tf

from cfg import CFG


class GraphModel(tf.keras.Model):
  def __init__(self, encoder, decoder):
    super(GraphModel, self).__init__()
    self.encoder = encoder
    self.decoder = decoder


  def call(self, batch):
    x = self.encoder(batch)
    adj_out, nf_out = self.decoder()
    return adj_out, nf_out


def make_model(models):
  if CFG['VISION']:
    pass
  elif CFG['JUST_VISION']
    pass
  else:
    print("Graph only model")
    model = GraphModel(
      encoder=models['encoder'],
      decoder=models['decoder'],
    )
