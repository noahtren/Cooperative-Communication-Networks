# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=""  # specify which GPU(s) to be used


import code

import tensorflow as tf
import tensorflow_addons as tfa
from cfg import CFG

from ml_utils import dense_regularization


class CPPN(tf.keras.Model):
  """Compositional Pattern-Producing Network

  Embeds each (x,y,r) -- where r is radius from center -- pair triple via a
  series of dense layers, combines with graph representation (z) and regresses
  pixel values directly.
  """
  def __init__(self, y_dim:int, x_dim:int, G_hidden_size:int, R:int,
               cppn_loc_embed_dim, cppn_Z_embed_dim:int, c_out:int=1,
               **kwargs):
    super(CPPN, self).__init__()
    self.loc_embed = tf.keras.layers.Dense(cppn_loc_embed_dim, **dense_regularization)
    self.Z_embed = tf.keras.layers.Dense(cppn_Z_embed_dim, **dense_regularization)
    self.in_w = tf.keras.layers.Dense(G_hidden_size, **dense_regularization)
    self.ws = [tf.keras.layers.Dense(G_hidden_size, **dense_regularization) for _ in range(R)]
    self.out_w = tf.keras.layers.Dense(c_out, **dense_regularization)
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.spatial_scale = 1 / max([y_dim, x_dim])
    self.output_scale = 1. / tf.math.sqrt(tf.cast(G_hidden_size, tf.float32))

  def call(self, Z):
    batch_size = Z.shape[0]

    # get pixel locations and embed pixels
    coords = tf.where(tf.ones((self.y_dim, self.x_dim)))
    coord_ints = tf.reshape(coords, (self.y_dim, self.x_dim, 2))
    coords = tf.cast(coord_ints, tf.float32)
    coords = tf.stack([coords[:, :, 0] * self.spatial_scale,
                       coords[:, :, 1] * self.spatial_scale], axis=-1)
    dists = tf.stack([coords[:, :, 0] - 0.5,
                      coords[:, :, 1] - 0.5], axis=-1)
    r = tf.sqrt(tf.math.reduce_sum(dists ** 2, axis=-1))[..., tf.newaxis]
    loc = tf.concat([coords, r], axis=-1)
    loc = self.loc_embed(loc)
    loc = tf.tile(loc[tf.newaxis], [batch_size, 1, 1, 1])

    # concatenate Z to locations
    Z = self.Z_embed(Z)
    Z = tf.tile(Z[:, tf.newaxis, tf.newaxis], [1, self.y_dim, self.x_dim, 1])
    x = tf.concat([loc, Z], axis=-1)
    x = self.in_w(x)

    # encode
    for layer in self.ws:
      start_x = x
      x = layer(x)
      x = x + start_x
      x = tf.nn.swish(x)

    x = self.out_w(x)
    x = tf.nn.tanh(x)
    if x.shape[-1] == 1:
      # Copy grayscale along RGB axes for easy input into pre-trained, color-based models
      x = tf.tile(x, [1, 1, 1, 3])
    return x


class ConvGenerator(tf.keras.Model):
  def __init__(self, y_dim:int, x_dim:int, G_hidden_size:int, R:int,
               cppn_loc_embed_dim, gen_Z_embed_dim:int, c_out:int=1,
               **kwargs):
    super(ConvGenerator, self).__init__()
    self.Z_embed = tf.keras.layers.Dense(gen_Z_embed_dim, **dense_regularization)
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.spatial_scale = 1 / max([y_dim, x_dim])
    self.output_scale = 1. / tf.math.sqrt(tf.cast(G_hidden_size, tf.float32))
    self.upconvs = []
    self.convs = []
    self.upconv_norms = []
    self.conv_norms = []
    filters = G_hidden_size
    for r in range(R):
      upconv = tf.keras.layers.Conv2DTranspose(
        filters,
        kernel_size=3,
        strides=2,
        padding='same',
        **dense_regularization
      )
      conv = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding='same',
        **dense_regularization
      )
      self.upconvs = [upconv] + self.upconvs
      self.convs = [conv] + self.convs
      self.upconv_norms.append(tf.keras.layers.LayerNormalization())
      self.conv_norms.append(tf.keras.layers.LayerNormalization())
      filters = filters // 2
    self.out_conv = tf.keras.layers.Conv2D(
      CFG['c_out'],
      kernel_size=1,
      strides=1,
      padding='same',
      **dense_regularization
    )

  def call(self, Z):
    batch_size = Z.shape[0]
    Z = self.Z_embed(Z)
    x = Z[:, tf.newaxis, tf.newaxis]
    for upconv, conv, upconv_norm, conv_norm in \
      zip(self.upconvs, self.convs, self.upconv_norms, self.conv_norms):
      x = upconv(x)
      x = upconv_norm(x)
      start_x = x
      x = tf.nn.swish(x)

      x = conv(x)
      x = conv_norm(x)
      x = x + start_x
      x = tf.nn.swish(x)

    x = self.out_conv(x)
    x = tf.nn.tanh(x)
    if x.shape[-1] == 1:
      # Copy grayscale along RGB axes for easy input into pre-trained, color-based models
      x = tf.tile(x, [1, 1, 1, 3])
    return x


def Generator():
  if CFG['vision_model'] == 'conv':
    return ConvGenerator(**CFG)
  elif CFG['vision_model'] == 'cppn':
    return CPPN(**CFG)


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
    x = tf.keras.layers.Dense(NUM_SYMBOLS, **dense_regularization)(x)
    x = tf.keras.layers.Lambda(lambda x: x * scale)(x)
    x = tf.keras.layers.Activation(tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=[x])
  return model


DISC_MODEL = 'ResNet50V2'
ImageDecoder = None
Perceptor = None
PerceptorLayerName = None
ModelFunc = None
if DISC_MODEL == 'Xception':
  ModelFunc = tf.keras.applications.Xception
  PerceptorLayerName = 'block3_sepconv2_bn' # 4x downscale
  # PerceptorLayerName = 'block4_sepconv2_bn' # 8x downscale
elif DISC_MODEL == 'ResNet50V2':
  ModelFunc = tf.keras.applications.ResNet50V2
  # PerceptorLayerName = 'conv2_block2_out' # 4x downscale
  PerceptorLayerName = 'conv3_block3_out' # 8x downscale

ImageDecoder = ModelFunc(
  include_top=False,
  weights="imagenet",
  input_shape=((CFG['y_dim'], CFG['x_dim'], 3)),
)
Perceptor = ModelFunc(
  include_top=False,
  weights="imagenet",
  input_shape=((CFG['y_dim'], CFG['x_dim'], 3)),
)
# PerceptorLayer = Perceptor.get_layer(PerceptorLayerName)
# Perceptor = tf.keras.Model(inputs=Perceptor.inputs, outputs=[PerceptorLayer.output])


def perceptual_loss(imgs, max_pairs=100):
  """Returns a negative value, where higher magnitudes describe further distances.
  This is to encourage samples to be perceptually more distant from each other.
  i.e., they are repelled from each other.
  """
  num_pairs = min([imgs.shape[0] * imgs.shape[0], max_pairs])
  pair_idxs = tf.where(tf.ones((imgs.shape[0], imgs.shape[0])))
  use_pair_idxs = tf.random.uniform([num_pairs], minval=0, maxval=num_pairs, dtype=tf.int32)
  pair_idxs = tf.gather(pair_idxs, use_pair_idxs)

  features = Perceptor(imgs)
  feature_pairs = tf.gather(features, pair_idxs)
  diffs = feature_pairs[:, 0] - feature_pairs[:, 1]
  diffs = tf.math.reduce_mean(tf.abs(diffs), axis=[1, 2, 3])
  repel_loss = tf.sqrt(tf.math.reduce_mean(diffs) + 0.001) * -1
  return repel_loss


if __name__ == "__main__":
  # ex = tf.random.normal([8, 64, 64, 3])
  # perceptual_loss(ex)
  pass