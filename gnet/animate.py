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

# load custom config from cloud
from cfg import read_config, read_config_from_string, set_config
from upload import gs_download_blob_as_string

run_name = 'cloud_vision_only_newaug_test'
# run_name = 'cloud_vision_only_newaug_test_night'


def get_loaded_model_config(run_name=None):
  if run_name is None:
    local_cfg = read_config()
    run_name = local_cfg['load_name']
    assert run_name is not None
  blob_name = f"logs/{run_name}/config.json"
  cfg_str = gs_download_blob_as_string(blob_name)
  cfg = read_config_from_string(cfg_str)
  cfg['load_name'] = run_name
  return cfg

CFG = get_loaded_model_config(run_name)
set_config(CFG)

from models import get_model, get_optim, run_dummy_batch, load_weights, \
  save_weights


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
  """Currently (and probably always) all visualizations are made locally
  """
  path_prefix = CFG['root_filepath']
  model = get_model()
  run_dummy_batch(model)
  load_weights(model, path_prefix)
  optim = get_optim()
  difficulty = tf.convert_to_tensor(1)
  
  if CFG['JUST_VISION']:
    symbols = make_sliding_window_symbols([3,8,9,3], 30)
    writer = imageio.get_writer(f"gallery/{CFG['load_name']}_animation.mp4", format='FFMPEG', fps=20)
    for i, symbol in tqdm(enumerate(symbols)):
      symbol = tf.expand_dims(symbol, 0)
      predictions, img, aug_img = model(symbol, difficulty)
      img = tf.squeeze((img + 1) / 2)
      img = tf.cast(img * 255, tf.uint8)
      img = circle_crop(img)
      writer.append_data(np.array(img))
    writer.close()
  elif CFG['FULL']:
    raise NotImplementedError
  else:
    assert RuntimeError(f"Nothing to animate for a model ({CFG['run_name']}) that is graph only!")
