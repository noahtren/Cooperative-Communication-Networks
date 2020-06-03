"""Interpolate between different inputs to produce smooth animations that
illustrate the differences between patterns.
"""

# don't use GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import code
from typing import List

import tensorflow as tf
import numpy as np
import imageio
from tqdm import tqdm

from encoder import Encoder
from decoder import GraphDecoder
from vision import Generator, Decoder
from train_vision import make_data
from cfg import CFG
from ml_utils import load_ckpts
from graph_data import TensorGraph, get_dataset, label_data
from graph_match import minimum_loss_permutation


def make_sliding_window_symbols(points:List[int], steps_between:int):
  """Given a list of points to interpolate between and the number
  of steps to use in interpolation, return a 2D tensor of inputs
  that follow the interpolation order -- in order to produce a
  smooth animation.
  """
  assert len(points) >= 2
  data = []
  for i in range(len(points) - 1):
    pt_1 = points[i]
    pt_2 = points[i + 1]
    for step in range(steps_between):
      ratio = 1 - (step / (steps_between))
      val = np.zeros((CFG['NUM_SYMBOLS'],), dtype=np.float32)
      val[pt_1] = ratio
      val[pt_2] = 1 - ratio
      data.append(val)
  data = tf.convert_to_tensor(data, dtype=tf.float32)
  return data


def get_tree_results(steps_between:int, models):
  # generate a list of distinct expression trees
  adj_lists = [
    [[], [], [0, 1]],
    [[], [], [0, 1]],
    [[], [], [0, 1]],
    [[], [], [0, 1]],
  ]
  value_tokens_lists = [
    ['a', 'a', '+'],
    ['a', 'b', '+'],
    ['a', 'c', '+'],
    ['a', 'd', '+'],
  ]
  order_lists = [
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
  ]
  # convert to TensorGraphs
  tgs = []
  for i in range(len(adj_lists)):
    tree_args = TensorGraph.parse_tree(CFG['language_spec'], adj_lists[i],
      value_tokens_lists[i], order_lists[i], CFG['max_nodes'])
    tg = TensorGraph.tree_to_instance(*tree_args, CFG['max_nodes'], CFG['language_spec'])
    tgs.append(tg)
  # debugging with a random tree from other algo
  tg = None
  while tg is None:
    tree_args = TensorGraph.random_tree(**CFG)
    tg = TensorGraph.tree_to_instance(*tree_args, CFG['max_nodes'], CFG['language_spec'])
  tgs.append(tg)

  adj, node_features, node_feature_specs, num_nodes = TensorGraph.instances_to_tensors(tgs)

  _ = models['encoder'][0]({'node_features': node_features, 'num_nodes': num_nodes, 'adj': adj})
  Z = models['encoder'][0]({'node_features': node_features, 'num_nodes': num_nodes, 'adj': adj})
  x = Z

  states = []
  for i in range(x.shape[0] - 1):
    diff = x[i + 1] - x[i]
    for j in range(steps_between):
      ratio = j / steps_between
      state = x[i] + diff * ratio
      states.append(state)
  states = tf.stack(states, axis=0)
  # generate images

  _ = models['generator'][0](Z)
  imgs = models['generator'][0](Z)

  # evaluate performance of models
  nf_labels, adj_labels = label_data(node_features, adj)

  _ = models['discriminator'][0](imgs)
  Z = models['discriminator'][0](imgs)

  _ = models['decoder'][0](x)
  adj_pred, nf_pred = models['decoder'][0](Z)

  batch_loss, acc = minimum_loss_permutation(
    adj_labels,
    nf_labels,
    adj_pred,
    nf_pred
  )
  imgs = models['generator'][0](states)
  return states, imgs, acc



def circle_crop(img):
  """Zeroes all pixels that don't fall within a radius of half the length of the
  square image. This adds a nice visual effect and reflects all information in
  the image if the images have perfect accuracy in the noisiest channel setting.
  """
  y_dim = img.shape[0]
  x_dim = img.shape[1]
  coords = tf.where(tf.ones((y_dim, x_dim)))
  coords = tf.cast(tf.reshape(coords, (y_dim, x_dim, 2)), tf.float32)
  dists = tf.stack([coords[:, :, 0] - y_dim / 2,
                    coords[:, :, 1] - x_dim / 2], axis=-1)
  r = tf.sqrt(tf.math.reduce_sum(dists ** 2, axis=-1))
  use_idxs = r < y_dim / 2
  img = tf.where(use_idxs[..., tf.newaxis], img, tf.zeros_like(img))
  return img


if __name__ == "__main__":
  CFG['num_samples'] = 1
  _, _, node_feature_specs, _, \
  _, _ = get_dataset(**CFG, test=False)
  CFG['node_feature_specs'] = node_feature_specs
  encoder = Encoder(**CFG)
  generator = Generator()
  discriminator = Decoder()
  decoder = GraphDecoder(**CFG)
  if CFG['JUST_VISION']:
    data = make_sliding_window_symbols([3,8,9,3], 30)
    generator(tf.expand_dims(data[0], 0))
    models = {'generator': [generator, None]}
    load_ckpts(models, CFG['load_vision_name'], 'latest')
    writer = imageio.get_writer(f"gallery/{CFG['load_name']}_animation.mp4", format='FFMPEG', fps=20)
    for i, val in tqdm(enumerate(data)):
      val = tf.expand_dims(val, 0)
      img = generator(val)
      img = tf.squeeze((img + 1) / 2)
      img = tf.cast(img * 255, tf.uint8)
      img = circle_crop(img)
      writer.append_data(np.array(img))
    writer.close()
  else:
    models = {
      'encoder': [encoder, None],
      'generator': [generator, None],
      'discriminator': [discriminator, None],
      'decoder': [decoder, None]
    }
    load_ckpts(models, CFG['load_graph_name'], 'best')
    states, imgs, acc = get_tree_results(30, models)
    writer = imageio.get_writer(f"gallery/{CFG['load_graph_name']}_animation.mp4", format='FFMPEG', fps=20)
    for i, img in tqdm(enumerate(imgs)):
      img = tf.squeeze((img + 1) / 2)
      img = tf.cast(img * 255, tf.uint8)
      img = circle_crop(img)
      writer.append_data(np.array(img))
    writer.close()
