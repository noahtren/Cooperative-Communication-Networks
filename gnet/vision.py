# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=""  # specify which GPU(s) to be used


import code

import tensorflow as tf
import tensorflow_addons as tfa
from cfg import CFG

from ml_utils import dense_regularization, cnn_regularization


def generate_scaled_coordinate_hints(batch_size, y_dim, x_dim):
  """Generally used as the input to a CPPN, but can also augment each layer
  of a ConvNet with location hints
  """
  spatial_scale = 1. / max([y_dim, x_dim])
  coords = tf.where(tf.ones((y_dim, x_dim)))
  coord_ints = tf.reshape(coords, (y_dim, x_dim, 2))
  coords = tf.cast(coord_ints, tf.float32)
  coords = tf.stack([coords[:, :, 0] * spatial_scale,
                      coords[:, :, 1] * spatial_scale], axis=-1)
  dists = tf.stack([coords[:, :, 0] - 0.5,
                    coords[:, :, 1] - 0.5], axis=-1)
  r = tf.sqrt(tf.math.reduce_sum(dists ** 2, axis=-1))[..., tf.newaxis]
  loc = tf.concat([coords, r], axis=-1)
  loc = tf.tile(loc[tf.newaxis], [batch_size, 1, 1, 1])
  return loc


class CPPN(tf.keras.Model):
  """Compositional Pattern-Producing Network

  Embeds each (x,y,r) -- where r is radius from center -- pair triple via a
  series of dense layers, combines with graph representation (z) and regresses
  pixel values directly.
  """
  def __init__(self, y_dim:int, x_dim:int, vision_hidden_size:int, R:int,
               cppn_loc_embed_dim, cppn_Z_embed_dim:int, c_out:int=1,
               **kwargs):
    super(CPPN, self).__init__()
    self.loc_embed = tf.keras.layers.Dense(cppn_loc_embed_dim, **dense_regularization)
    self.Z_embed = tf.keras.layers.Dense(cppn_Z_embed_dim, **dense_regularization)
    self.in_w = tf.keras.layers.Dense(vision_hidden_size, **dense_regularization)
    self.ws = [tf.keras.layers.Dense(vision_hidden_size, **dense_regularization) for _ in range(R)]
    self.out_w = tf.keras.layers.Dense(c_out, **dense_regularization)
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.spatial_scale = 1 / max([y_dim, x_dim])
    self.output_scale = 1. / tf.math.sqrt(tf.cast(vision_hidden_size, tf.float32))

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
  def __init__(self, y_dim:int, x_dim:int, vision_hidden_size:int, R:int,
               c_out:int, Z_embed_num:int, minimum_filters:int,
               **kwargs):
    super(ConvGenerator, self).__init__()
    self.init_Z_embed = tf.keras.layers.Dense(vision_hidden_size, **dense_regularization)
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.upconvs = []
    self.convs = []
    self.one_convs = []
    self.upconv_norms = []
    self.one_conv_norms = []
    self.conv_norms = []
    self.Z_embeds = []
    self.Z_norms = []
    self.loc_convs = []
    self.loc_norms = []
    for r in range(R):
      filters = vision_hidden_size // (2 ** (R-r))
      filters = max([filters, minimum_filters])
      upconv = tf.keras.layers.Conv2DTranspose(
        filters,
        kernel_size=3,
        strides=2,
        padding='same',
        **cnn_regularization
      )
      one_conv = tf.keras.layers.Conv2D(
        filters * 2,
        kernel_size=1,
        strides=1,
        padding='same',
        **cnn_regularization
      )
      conv = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding='same',
        **cnn_regularization
      )
      Z_filters = filters // 4
      loc_conv = tf.keras.layers.Conv2D(
        Z_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        # no regularization because this takes raw pixel locations as inputs
      )
      Z_embeds = [tf.keras.layers.Dense(256, **dense_regularization) for _ in range(Z_embed_num - 1)] + \
        [tf.keras.layers.Dense(Z_filters, **dense_regularization)]
      Z_norms = [tf.keras.layers.LayerNormalization() for _ in range(Z_embed_num)]
      self.upconvs = [upconv] + self.upconvs
      self.one_convs = [one_conv] + self.one_convs
      self.convs = [conv] + self.convs
      self.Z_embeds = [Z_embeds] + self.Z_embeds
      self.Z_norms = [Z_norms] + self.Z_norms
      self.loc_convs = [loc_conv] + self.loc_convs
      self.upconv_norms.append(tf.keras.layers.LayerNormalization())
      self.one_conv_norms.append(tf.keras.layers.LayerNormalization())
      self.conv_norms.append(tf.keras.layers.LayerNormalization())
      self.loc_norms.append(tf.keras.layers.LayerNormalization())
    self.out_conv = tf.keras.layers.Conv2D(
      c_out,
      kernel_size=1,
      strides=1,
      padding='same',
      **cnn_regularization
    )

  def call(self, Z, debug=False):
    batch_size = Z.shape[0]
    if CFG['JUST_VISION']:
      x = self.init_Z_embed(Z)
    else:
      x = Z
    x = x[:, tf.newaxis, tf.newaxis]
    if debug: tf.print('GENERATOR')
    if debug: tf.print(x.shape)
    for li, _ in enumerate(self.convs):
      if debug: tf.print(x.shape)
      upconv = self.upconvs[li]
      upconv_norm = self.upconv_norms[li]
      Z_embeds = self.Z_embeds[li]
      Z_norms = self.Z_norms[li]
      loc_conv = self.loc_convs[li]
      loc_norm = self.loc_norms[li]
      one_conv = self.one_convs[li]
      one_conv_norm = self.one_conv_norms[li]
      conv = self.convs[li]
      conv_norm = self.conv_norms[li]

      x = upconv(x)
      x = upconv_norm(x)
      x = tf.nn.dropout(x, 0.1)
      x = tf.nn.swish(x)

      start_x = x
      y_dim, x_dim = x.shape[1], x.shape[2]
      loc = generate_scaled_coordinate_hints(batch_size, y_dim, x_dim)
      loc = loc_conv(loc)
      loc = loc_norm(loc)
      _z = Z
      for Z_embed, Z_norm in zip(Z_embeds, Z_norms):
        _z = Z_embed(_z)
        _z = tf.nn.swish(_z)
        if debug: tf.print(f"Z shape: {_z.shape}")
      _z = tf.tile(_z[:, tf.newaxis, tf.newaxis],
        [1, y_dim, x_dim, 1])

      if debug: tf.print(f"loc: {loc.shape}, z: {_z.shape}")
      x = tf.concat([x, loc, _z], axis=-1)
      if debug: tf.print(f"joined shape: {x.shape}")

      x = one_conv(x)
      x = one_conv_norm(x)
      x = tf.nn.swish(x)

      x = conv(x)
      x = conv_norm(x)
      x = tf.nn.dropout(x, 0.1)
      x = x + start_x
      x = tf.nn.swish(x)
    
    if debug: tf.print(x.shape)
    x = self.out_conv(x)
    x = tf.nn.tanh(x)
    if x.shape[-1] == 1:
      x = tf.tile(x, [1, 1, 1, 3])
    return x


class ConvDiscriminator(tf.keras.Model):
  def __init__(self, y_dim:int, x_dim:int, vision_hidden_size:int, R:int,
               c_out:int, NUM_SYMBOLS:int, minimum_filters:int,
               graph_hidden_size:int,
               **kwargs):
    super(ConvDiscriminator, self).__init__()
    self.downconvs = []
    self.convs = []
    self.one_convs = []
    self.downconv_norms = []
    self.one_conv_norms = []
    self.conv_norms = []
    for r in range(R):
      filters = vision_hidden_size // (2 ** (R-r-1))
      filters = max([filters, minimum_filters])
      downconv = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=2,
        padding='same',
        **cnn_regularization
      )
      one_conv = tf.keras.layers.Conv2D(
        filters * 2,
        kernel_size=1,
        strides=1,
        padding='same',
        **cnn_regularization
      )
      conv = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding='same',
        **cnn_regularization
      )
      self.downconvs.append(downconv)
      self.one_convs.append(one_conv)
      self.convs.append(conv)
      self.downconv_norms.append(tf.keras.layers.LayerNormalization())
      self.one_conv_norms.append(tf.keras.layers.LayerNormalization())
      self.conv_norms.append(tf.keras.layers.LayerNormalization())
    self.out_conv = tf.keras.layers.Conv2D(
      vision_hidden_size,
      kernel_size=1,
      strides=1,
      padding='same',
      **dense_regularization
    )
    self.pred = tf.keras.layers.Dense(graph_hidden_size, **dense_regularization)
    self.gap = tf.keras.layers.GlobalAveragePooling2D()

  def call(self, x, debug=False):
    if debug: tf.print('DECODER')
    for li, _ in enumerate(self.convs):
      downconv = self.downconvs[li]
      downconv_norm = self.downconv_norms[li]
      one_conv = self.one_convs[li]
      one_conv_norm = self.one_conv_norms[li]
      conv = self.convs[li]
      conv_norm = self.conv_norms[li]
      if debug: tf.print(x.shape)
      x = downconv(x)
      x = downconv_norm(x)
      x = tf.nn.dropout(x, 0.1)
      x = tf.nn.swish(x)
      start_x = x

      x = one_conv(x)
      x = one_conv_norm(x)
      x = tf.nn.swish(x)

      x = conv(x)
      x = conv_norm(x)
      x = tf.nn.dropout(x, 0.1)
      x = x + start_x
      x = tf.nn.swish(x)
    
    if debug: tf.print(x.shape)
    x = self.out_conv(x)
    x = self.gap(x)
    # remove this when training on graph task
    if CFG['JUST_VISION']:
      x = self.pred(x)
      x = tf.nn.softmax(x)
    return x


def Generator():
  if CFG['vision_model'] == 'conv':
    Generator = ConvGenerator(**CFG)
  elif CFG['vision_model'] == 'cppn':
    Generator = CPPN(**CFG)
  Generator._name = 'generator'
  return Generator


def get_pretrained_info():
  if CFG['pretrained_disc_name'] == 'Xception':
    ModelFunc = tf.keras.applications.Xception
    # PerceptorLayerName = 'block3_sepconv2_bn' # 4x downscale
    PerceptorLayerName = 'block4_sepconv2_bn' # 8x downscale
  elif CFG['pretrained_disc_name'] == 'ResNet50V2':
    ModelFunc = tf.keras.applications.ResNet50V2
    # PerceptorLayerName = 'conv2_block3_out' # 4x downscale
    PerceptorLayerName = 'conv3_block4_out' # 8x downscale
    # PerceptorLayerName = 'conv4_block6_out' # 16x downscale
    # PerceptorLayerName = 'conv5_block2_out' # 32x downscale
  elif CFG['pretrained_disc_name'] == 'VGG16':
    ModelFunc = tf.keras.applications.VGG16
    # PerceptorLayerName = 'block3_conv3' # 4x downscale
    # PerceptorLayerName = 'block4_conv3' # 8x downscale
    PerceptorLayerName = 'block5_conv3' # 16x downscale
  return ModelFunc, PerceptorLayerName


def Decoder():
  if CFG['use_custom_disc']:
    ImageDecoder = ConvDiscriminator(**CFG)
  else:
    ModelFunc, _ = get_pretrained_info()
    ImageDecoder = ModelFunc(
      include_top=False,
      weights="imagenet",
      input_shape=((CFG['y_dim'], CFG['x_dim'], 3)),
    )
  ImageDecoder._name = 'decoder'
  return ImageDecoder


def Perceptor():
  """Use a pretrained model to use in calculating a perceptual loss score
  """
  ModelFunc, PerceptorLayerName = get_pretrained_info()
  Perceptor = ModelFunc(
    include_top=False,
    weights="imagenet",
    input_shape=((CFG['y_dim'], CFG['x_dim'], 3))
  )
  PerceptorLayer = Perceptor.get_layer(PerceptorLayerName)
  Perceptor = tf.keras.Model(inputs=Perceptor.inputs, outputs=[PerceptorLayer.output])
  print(f"Perceptor output: {Perceptor.layers[-1].output}")
  return Perceptor


def vector_distance_loss(rep1, rep2, max_pairs=1_000):
  """Experimental loss score based on trying to match the distance between
  corresponding pairs of representations, using logistic loss between distances
  normalized from 0 to 1. No idea how stable this will be!

  This use case: match the distances between pairs of images perceptions with
  the distances between pairs of symbol labels.
  """
  n = rep1.shape[0]
  assert n >= 4
  num_pairs = (rep1.shape[0] * (rep1.shape[0] - 1)) // 2
  pairs = tf.where(tf.ones((n, n)))

  # get diagonal of adjacency matrix indices
  unique_pairs = tf.squeeze(tf.where(pairs[:, 0] < pairs[:, 1]))
  pairs = tf.gather(pairs, unique_pairs)
  num_pairs = min([num_pairs, max_pairs])
  use_pair_idxs = tf.random.uniform([num_pairs], minval=0, maxval=num_pairs, dtype=tf.int32)
  pairs = tf.gather(pairs, use_pair_idxs)
  
  # DISTANCES 1
  pairs1 = tf.gather(rep1, pairs)
  diffs1 = tf.abs(pairs1[:, 0] - pairs1[:, 1])
  # get average feature distance
  diffs1 = tf.math.reduce_mean(diffs1, axis=tf.range(tf.rank(diffs1) - 1) + 1)

  # DISTANCES 2
  pairs2 = tf.gather(rep2, pairs)
  diffs2 = tf.abs(pairs2[:, 0] - pairs2[:, 1])
  # get average feature distance
  diffs2 = tf.math.reduce_mean(diffs2, axis=tf.range(tf.rank(diffs2) - 1) + 1)

  # NORMALIZE DISTANCES
  def zero_one_normalize(tensor):
    # dont subtract minimum because this fails with all samples being equal
    # tensor = tensor - tf.math.reduce_min(tensor)
    tensor = tensor / tf.math.reduce_max(tensor)
    return tensor

  diffs1 = zero_one_normalize(diffs1)
  diffs2 = zero_one_normalize(diffs2)
  
  # FIND THE DISTANCE BETWEEN DISTANCES (HAHA)
  diffs1 = diffs1[:, tf.newaxis]
  diffs2 = diffs2[:, tf.newaxis]
  error = tf.keras.losses.binary_crossentropy(diffs1, diffs2)
  error = tf.math.reduce_mean(error)
  return error


def perceptual_loss(features, max_pairs=1_000, MULTIPLIER=-1):
  """Returns a negative value, where higher magnitudes describe further distances.
  This is to encourage samples to be perceptually more distant from each other.
  i.e., they are repelled from each other.
  """
  b_s = features.shape[0]
  num_pairs = (b_s * (b_s - 1)) // 2
  pair_idxs = tf.where(tf.ones((b_s, b_s)))
  non_matching_pairs = tf.squeeze(tf.where(pair_idxs[:, 0] < pair_idxs[:, 1]))
  pair_idxs = tf.gather(pair_idxs, non_matching_pairs)

  num_pairs = min([num_pairs, max_pairs])
  use_pair_idxs = tf.random.uniform([num_pairs], minval=0, maxval=num_pairs, dtype=tf.int32)
  pair_idxs = tf.gather(pair_idxs, use_pair_idxs)

  features = tf.math.reduce_mean(features, [1, 2])
  feature_pairs = tf.gather(features, pair_idxs)
  diffs = feature_pairs[:, 0] - feature_pairs[:, 1]
  diffs = tf.math.reduce_mean(tf.abs(diffs), axis=-1)

  percept_loss = tf.math.reduce_mean(diffs)
  percept_loss = tf.sqrt(percept_loss)
  percept_loss = percept_loss * MULTIPLIER
  return percept_loss


def make_symbol_data(num_samples):
  x = tf.random.uniform((num_samples,), 0, CFG['NUM_SYMBOLS'], dtype=tf.int32)
  x = tf.one_hot(x, depth=CFG['NUM_SYMBOLS'])
  samples = tf.range(CFG['NUM_SYMBOLS'])
  samples = tf.one_hot(samples, depth=CFG['NUM_SYMBOLS'])
  return x, samples


if __name__ == "__main__":
  # ex = tf.random.normal([8, 64, 64, 3])
  # perceptual_loss(ex)
  generator = ConvGenerator(**CFG)
