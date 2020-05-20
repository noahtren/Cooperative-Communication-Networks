"""Brute force graph matching with TensorFlow code
"""

import code
import itertools

import numpy as np
import tensorflow as tf


permutations = {
  3: np.array(list(itertools.permutations(range(3)))),
  4: np.array(list(itertools.permutations(range(4)))),
  5: np.array(list(itertools.permutations(range(5)))),
  6: np.array(list(itertools.permutations(range(6)))),
}


def loss_fn(adj, nf, possible_adjs, possible_nfs):
  # calculate losses along last axis (per node)

  # sum losses along second to last axis (per graph permutation)

  # argmax losses along second axis (per graph)



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
  min_loss = loss_fn(adj, nf, possible_adjs, possible_nfs)
  return min_loss
