import code

import tensorflow as tf
import tensorflow_addons as tfa
from cfg import CFG


# NOTE: the inception decoder requires an input with 3 channels.
# if I really want to do grayscale, I can still start with
# pre-trained weights, but just tile the output of the CPPN

class CPPN(tf.keras.Model):
  """Compositional Pattern-Producing Network

  Embeds each (x,y,r) -- where r is radius from center -- pair triple via a
  series of dense layers, combines with graph representation (z) and regresses
  pixel values directly.
  """
  def __init__(self, y_dim:int, x_dim:int, G_hidden_size:int, G_num_layers:int,
               cppn_loc_embed_dim:int=64, c_out:int=1, **kwargs):
    super(CPPN, self).__init__()
    self.loc_embed = tf.keras.layers.Dense(cppn_loc_embed_dim)
    self.ws = [tf.keras.layers.Dense(G_hidden_size) for _ in range(G_num_layers)]
    self.out_w = tf.keras.layers.Dense(c_out)
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.scale = 1 / max([y_dim, x_dim])


  def call(self, Z):
    batch_size = Z.shape[0]

    # get pixel locations and embed pixels
    coords = tf.where(tf.ones((self.y_dim, self.x_dim)))
    coords = tf.cast(tf.reshape(coords, (self.y_dim, self.x_dim, 2)), tf.float32)
    coords = tf.stack([coords[:, :, 0] * self.scale,
                       coords[:, :, 1] * self.scale], axis=-1)
    dists = tf.stack([coords[:, :, 0] - 0.5,
                      coords[:, :, 1] - 0.5], axis=-1)
    r = tf.sqrt(tf.math.reduce_sum(dists ** 2, axis=-1))[..., tf.newaxis]
    loc = tf.concat([coords, r], axis=-1)
    loc = self.loc_embed(loc)
    loc = tf.tile(loc[tf.newaxis], [batch_size, 1, 1, 1])

    # concatenate Z to locations
    Z = tf.tile(Z[:, tf.newaxis, tf.newaxis], [1, self.y_dim, self.x_dim, 1])
    x = tf.concat([loc, Z], axis=-1)

    # encode
    for layer in self.ws:
      x = layer(x)
      x = tfa.activations.gelu(x)

    x = self.out_w(x)
    x = tf.nn.tanh(x)
    return x


ImageDecoder = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=((CFG['y_dim'], CFG['x_dim'], 3)),
)

if __name__ == "__main__":
  cppn = CPPN(64, 64, 512, 3)
  Z = tf.random.normal((8, 512))
  img = cppn(Z)