"""Brute force graph matching with TensorFlow code. This works for small graphs.
(For graphs with >6 max nodes, it becomes pretty much unfeasible.)
"""

import code
import itertools

import numpy as np
import tensorflow as tf
from cfg import CFG


permutations = {
  3: np.array(list(itertools.permutations(range(3)))),
  4: np.array(list(itertools.permutations(range(4)))),
  5: np.array(list(itertools.permutations(range(5)))),
  6: np.array(list(itertools.permutations(range(6)))),
}


def loss_fn(adj, nf, possible_adjs, possible_nfs):
  acc = {}
  permute_dim = possible_adjs.shape[1]

  # calculate losses along last axis (per node)
  lfn = tf.keras.losses.mean_squared_error if CFG['mse_loss_only'] else \
    lambda true, pred: tf.keras.losses.binary_crossentropy(true, pred, label_smoothing=CFG['label_smoothing'])

  adj_label = tf.tile(adj[:, tf.newaxis], [1, permute_dim, 1, 1])
  loss = lfn(adj_label, possible_adjs)

  lfn = tf.keras.losses.mean_squared_error if CFG['mse_loss_only'] else \
    lambda true, pred: tf.keras.losses.categorical_crossentropy(true, pred, label_smoothing=CFG['label_smoothing'])
  for name, pred_nf in possible_nfs.items():
    nf_label = tf.tile(nf[name][:, tf.newaxis], [1, permute_dim, 1, 1])
    loss += lfn(nf_label, pred_nf)

  # sum losses along second to last axis (per graph permutation)
  loss = tf.math.reduce_sum(loss, axis=-1)

  # argmin losses along second axis (per graph)
  min_idxs = tf.argmin(loss, axis=-1)
  min_idxs = tf.stack([tf.range(min_idxs.shape[0], dtype=tf.int64), min_idxs], axis=1)
  loss = tf.gather_nd(loss, min_idxs)

  # calculate metrics
  selected_adjs = tf.gather_nd(possible_adjs, min_idxs)
  acc['adj'] = tf.math.reduce_mean(
    tf.keras.metrics.binary_accuracy(tf.cast(adj, tf.float32), selected_adjs)
  )
  for name, pred_nf in possible_nfs.items():
    selected_nf = tf.gather_nd(pred_nf, min_idxs)
    acc[name] = tf.math.reduce_mean(
      tf.keras.metrics.categorical_accuracy(nf[name], selected_nf)
    )

  # sum loss for each graph in batch
  loss = tf.math.reduce_sum(loss, axis=-1)
  return loss, acc


def minimum_loss_permutation(adj, nf, adj_pred, nf_pred):
  perms = permutations[adj.shape[1]]
  perm_mats = tf.one_hot(perms, depth=adj.shape[1])
  # produce possible permutations of predictions
  possible_adjs = tf.matmul(
    tf.matmul(tf.linalg.matrix_transpose(perm_mats), adj_pred[:, tf.newaxis]),
    perm_mats
  )
  possible_nfs = {
    name: tf.matmul(tf.linalg.matrix_transpose(perm_mats), tensor[:, tf.newaxis]) for
      name, tensor in nf_pred.items()
  }
  min_loss, acc = loss_fn(adj, nf, possible_adjs, possible_nfs)
  return min_loss, acc
