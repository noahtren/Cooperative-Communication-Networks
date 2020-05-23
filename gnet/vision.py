import code

import tensorflow as tf
import tensorflow_addons as tfa
from cfg import CFG


class CPPN(tf.keras.Model):
  """Compositional Pattern-Producing Network

  Embeds each (x,y,r) -- where r is radius from center -- pair triple via a
  series of dense layers, combines with graph representation (z) and regresses
  pixel values directly.
  """
  def __init__(self, y_dim:int, x_dim:int, G_hidden_size:int, G_num_layers:int,
               cppn_loc_embed_dim, cppn_Z_embed_dim:int, c_out:int=1,
               **kwargs):
    super(CPPN, self).__init__()
    self.loc_embed = tf.keras.layers.Dense(cppn_loc_embed_dim)
    self.Z_embed = tf.keras.layers.Dense(cppn_Z_embed_dim)
    self.in_w = tf.keras.layers.Dense(G_hidden_size)
    self.ws = [tf.keras.layers.Dense(G_hidden_size) for _ in range(G_num_layers)]
    self.out_w = tf.keras.layers.Dense(c_out)
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.spatial_scale = 1 / max([y_dim, x_dim])
    self.output_scale = 1. / tf.math.sqrt(tf.cast(G_hidden_size, tf.float32))
    self.norm_1 = tf.keras.layers.LayerNormalization()
    self.norm_2 = tf.keras.layers.LayerNormalization()

  def call(self, Z):
    batch_size = Z.shape[0]
    Z = self.norm_1(Z)

    # get pixel locations and embed pixels
    coords = tf.where(tf.ones((self.y_dim, self.x_dim)))
    coords = tf.cast(tf.reshape(coords, (self.y_dim, self.x_dim, 2)), tf.float32)
    coords = tf.stack([coords[:, :, 0] * self.spatial_scale,
                       coords[:, :, 1] * self.spatial_scale], axis=-1)
    dists = tf.stack([coords[:, :, 0] - 0.5,
                      coords[:, :, 1] - 0.5], axis=-1)
    r = tf.sqrt(tf.math.reduce_sum(dists ** 2, axis=-1))[..., tf.newaxis]
    loc = tf.concat([coords, r], axis=-1)
    loc = self.loc_embed(loc)
    # loc = tfa.activations.gelu(loc)
    loc = tf.tile(loc[tf.newaxis], [batch_size, 1, 1, 1])

    # concatenate Z to locations
    Z = tf.tile(Z[:, tf.newaxis, tf.newaxis], [1, self.y_dim, self.x_dim, 1])
    x = tf.concat([loc, Z], axis=-1)
    x = self.in_w(x)

    # encode
    for layer in self.ws:
      start_x = x
      x = layer(x)
      x = x + start_x
      x = tfa.activations.gelu(x)

    x = self.norm_2(x)
    x = self.out_w(x)
    x = self.output_scale * x
    x = tf.nn.tanh(x)
    if x.shape[-1] == 1:
      # Copy grayscale along RGB axes for easy input into pre-trained, color-based models
      x = tf.tile(x, [1, 1, 1, 3])
    return x


def modify_decoder(decoder, just_GAP=True, NUM_SYMBOLS=None):
  """Takes an image decoder and adds a final classification
  layer with as many output classes as the number of symbols
  in the toy problem.
  """
  inputs = decoder.inputs
  x = decoder.outputs[0]
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  if not just_GAP:
    scale = 1. / tf.math.sqrt(tf.cast(x.shape[-1], tf.float32))
    x = tf.keras.layers.Dense(NUM_SYMBOLS)(x)
    x = tf.keras.layers.Lambda(lambda x: x * scale)(x)
    x = tf.keras.layers.Activation(tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=[x])
  return model


ImageDecoder = tf.keras.applications.InceptionV3(
  include_top=False,
  weights="imagenet",
  input_shape=((CFG['y_dim'], CFG['x_dim'], 3)),
)


# ImageDecoder = tf.keras.applications.ResNet50V2(
#     include_top=False,
#     weights="imagenet",
#     input_shape=((CFG['y_dim'], CFG['x_dim'], 3)),
# )

if __name__ == "__main__":
  cppn = CPPN(64, 64, 512, 3)
  Z = tf.random.normal((8, 512))
  img = cppn(Z)
