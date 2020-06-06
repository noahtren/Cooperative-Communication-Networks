"""Generic utilities for machine learning.
"""

import code
import os

import tensorflow as tf

from .cfg import get_config; CFG = get_config()


def shuffle_together(*tensors):
  idxs = tf.range(tensors[0].shape[0])
  out = []
  for tensor in tensors:
    out.append(tf.gather(tensor, idxs))
  return tuple(out)


def update_data_dict(data_dict, batch_dict):
  for name in batch_dict.keys():
    if name not in data_dict:
      data_dict[name] = batch_dict[name]
    else:
      data_dict[name] += batch_dict[name]
  return data_dict


def normalize_data_dict(data_dict, num_batches):
  for name, value in data_dict.items():
    data_dict[name] = (value / num_batches).numpy().item()
  return data_dict


def gaussian_k(height, width, y, x, sigma, normalized=True):
  """Make a square gaussian kernel centered at (x, y) with sigma as standard deviation.
  Returns:
      A 2D array of size [height, width] with a Gaussian kernel centered at (x, y)
  """
  # cast arguments used in calculations
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.float32)
  sigma = tf.cast(sigma, tf.float32)
  # create indices
  xs = tf.range(0, width, delta=1., dtype=tf.float32)
  ys = tf.range(0, height, delta=1., dtype=tf.float32)
  ys = tf.expand_dims(ys, 1)
  # apply gaussian function to indices based on distance from x, y
  gaussian = tf.math.exp(-((xs - x)**2 + (ys - y)**2) / (2 * (sigma**2)))
  if normalized:
      gaussian = gaussian / tf.math.reduce_sum(gaussian) # all values will sum to 1
  return gaussian


# ============================== REGULARIZATION ==============================
dense_regularization = {
  'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  'bias_regularizer': tf.keras.regularizers.l2(1e-4),
  'activity_regularizer': tf.keras.regularizers.l2(1e-6)
}


# some notes on regularizing CNNs: https://cs231n.github.io/neural-networks-2/#reg
cnn_regularization = {
  'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  'bias_regularizer': tf.keras.regularizers.l2(1e-4),
}

