"""Optimizer utilities. Found on Phil Culliton's Kaggle page.
https://www.kaggle.com/philculliton/bert-optimization
"""

import code
import os

import tensorflow as tf
try:
  CFG
except NameError:
  from cfg import CFG


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

