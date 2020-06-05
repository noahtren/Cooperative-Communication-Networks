"""Interpolate between different inputs to produce smooth animations that
illustrate the differences between patterns.
"""

# don't use GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import code
import math
from typing import List

import tensorflow as tf
import numpy as np
import imageio
from tqdm import tqdm

# load custom config from cloud
from cfg import read_config, read_config_from_string, set_config
from upload import gs_download_blob_as_string

# see https://www.notion.so/tftnotes/Visual-Programming-Autoencoder-b487383f80834898928be7b2c7e45d63
# for gallery of best results

# NOTE: there are some black and white graph examples that I haven't visualized yet

run_name = 'cloud_full_test'
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
from ml_utils import gaussian_k
from graph_data import TensorGraph, label_data


def make_sliding_window_symbols(points:List[int], steps_between:int):
  """Given a list of points to interpolate between and the number
  of steps to use in interpolation, return a 2D tensor of inputs
  that follow the interpolation order -- in order to produce a
  smooth animation.
  """
  assert len(points) >= 2
  data = []
  texts = []
  for i in range(len(points) - 1):
    pt_1 = points[i]
    pt_2 = points[i + 1]
    for step in range(steps_between):
      ratio = 1 - (step / (steps_between))
      val = np.zeros((CFG['NUM_SYMBOLS'],), dtype=np.float32)
      val[pt_1] = ratio
      val[pt_2] = 1 - ratio
      text = {
        'top': {
          'text': str(i + 1) if i < len(points) - 2 else '0',
          'amount': 1 - ratio,
        },
        'bottom': {
          'text': str(i),
          'amount': ratio,
        }
      }
      data.append(val)
      texts.append(text)
  data = tf.convert_to_tensor(data, dtype=tf.float32)
  return data, texts


def get_tree_results(steps_between:int,
                     model,
                     adj_lists,
                     value_tokens_lists,
                     order_lists,
                     token_strings,
                     debug=False):
  # convert to TensorGraphs
  tgs = []
  for i in range(len(adj_lists)):
    tree_args = TensorGraph.parse_tree(CFG['language_spec'], adj_lists[i],
      value_tokens_lists[i], order_lists[i], CFG['max_nodes'])
    tg = TensorGraph.tree_to_instance(*tree_args, CFG['max_nodes'], CFG['language_spec'])
    tgs.append(tg)
  # debugging with a random tree from other algo
  if debug:
    tg = None
    while tg is None:
      tree_args = TensorGraph.random_tree(**CFG)
      tg = TensorGraph.tree_to_instance(*tree_args, CFG['max_nodes'], CFG['language_spec'])
    tgs.append(tg)

  adj, node_features, node_feature_specs, num_nodes = TensorGraph.instances_to_tensors(tgs)
  nf_labels, adj_labels = label_data(node_features, adj, num_nodes)

  batch = {
    'adj': adj,
    'node_features': node_features,
    'num_nodes': num_nodes
  }
  difficulty = tf.convert_to_tensor(1)
  adj_out, nf_out, imgs, aug_imgs, Z, Z_pred = model(batch, difficulty)

  states = []
  texts = []
  for i in range(Z.shape[0] - 1):
    diff = Z[i + 1] - Z[i]
    for j in range(steps_between):
      ratio = j / steps_between
      state = Z[i] + diff * ratio
      text = {
        'top': {
          'text': token_strings[i + 1] if i < Z.shape[0] - 2 else token_strings[0],
          'amount': ratio,
        },
        'bottom': {
          'text': token_strings[i],
          'amount': 1 - ratio,
        }
      }
      states.append(state)
      texts.append(text)
  states = tf.stack(states, axis=0)
  # generate images

  # _ = models['generator'][0](Z)
  # imgs = models['generator'][0](Z)

  # evaluate performance of models
  # nf_labels, adj_labels = label_data(node_features, adj)

  # _ = models['discriminator'][0](imgs)
  # Z = models['discriminator'][0](imgs)

  # _ = models['decoder'][0](x)
  # adj_pred, nf_pred = models['decoder'][0](Z)

  # batch_loss, acc = minimum_loss_permutation(
  #   adj_labels,
  #   nf_labels,
  #   adj_pred,
  #   nf_pred
  # )
  imgs = []
  for state in tqdm(states):
    img = model.generator(tf.expand_dims(state, 0))
    imgs.append(img[0])
  imgs = tf.stack(imgs, axis=0)
  return states, imgs, texts # acc


def annotate_image(img, text, use_side=64):
  """Annotate an image with the given text
  """
  from PIL import Image, ImageDraw, ImageFont
  orig_width = img.shape[1]
  fs = 16 # font size
  if use_side:
    zeros = tf.zeros((img.shape[0], use_side, img.shape[2]))
    img = tf.concat([img, zeros], axis=1)
  img = np.array(img).astype(np.uint8)
  image = Image.fromarray(img)
  draw = ImageDraw.Draw(image)
  font = ImageFont.truetype('OpenSans.ttf', fs)
  max_height = img.shape[0] // 4 - fs // 2
  min_height = 3 * img.shape[0] // 4 - fs // 2
  dist = (min_height - max_height) // 2

  def get_height(ratio, path='top'):
    """sinusoidal interpolation

    1 returns center
    top 0 returns top
    bot 0 returns bot
    """
    shift = int(dist * math.sin(ratio / 2 * math.pi))
    if path == 'top':
      return max_height + shift
    else:
      return min_height - shift

  if use_side:
    # top
    top_text = text['top']['text']
    top_color = int(text['top']['amount'] * 255)
    top_color = f"rgb({top_color},{top_color},{top_color})"
    top_height = get_height(text['top']['amount'], path='top')
    draw.text((orig_width + 8, top_height), top_text, fill=top_color, font=font)
    # bottom
    bot_text = text['bottom']['text']
    bot_color = int(text['bottom']['amount'] * 255)
    bot_color = f"rgb({bot_color},{bot_color},{bot_color})"
    bot_height = get_height(text['bottom']['amount'], path='bot')
    draw.text((orig_width + 8, bot_height), bot_text, fill=bot_color, font=font)

  else:
    # TODO
    draw.text((0, 0), text, fill='rgb(255,255,255)', font=font)
  return np.array(image)


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


def blur(imgs):
  """Apply blur via a Gaussian convolutional kernel
  """
  if tf.rank(imgs) == 3:
    imgs = imgs[tf.newaxis]
  img_shape = imgs[0].shape
  c = img_shape[2]
  gauss_kernel = gaussian_k(7, 7, 3, 3, 1.25)

  # expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature
  gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
  
  # convolve
  out_channels = []
  for c_ in range(c):
    in_channel = tf.expand_dims(imgs[..., c_], -1)
    out_channel = tf.nn.conv2d(in_channel, gauss_kernel, strides=1, padding="SAME")
    out_channel = out_channel[..., 0]
    out_channels.append(out_channel)
  imgs = tf.stack(out_channels, axis=-1)
  return imgs


def write_images(imgs, scale_down=1, circle_crop=False, do_blur=True, texts:str=None):
  """Texts is a list of strings equal to the number of imgs
  """
  assert len(texts) == len(imgs)

  writer = imageio.get_writer(f"gallery/{CFG['load_name']}_animation.mp4", format='FFMPEG', fps=20)
  for i, img in tqdm(enumerate(imgs)):
    img = (img + 1) / 2.
    img = tf.cast(img * 255, tf.uint8)
    img = tf.image.resize(img, [int(img.shape[0] / scale_down),
                                int(img.shape[1] / scale_down)],
                                'lanczos3',
                                antialias=True)
    if circle_crop:
      img = circle_crop(img)
    if do_blur:
      img = blur(img)[0]
    if texts is not None:
      img = annotate_image(img, texts[i])
    writer.append_data(np.array(img))
  writer.close()


if __name__ == "__main__":
  """Currently (and probably always) all visualizations are made locally
  """
  # TODO: consider cacheing so that multiple experiments can be run on same weights
  path_prefix = CFG['root_filepath']
  model = get_model()
  run_dummy_batch(model)
  load_weights(model, path_prefix)
  optim = get_optim()
  
  if CFG['JUST_VISION']:
    symbols, texts = make_sliding_window_symbols([0,1,2,3,4,5,6,7,8,9,0], 20)
    imgs = []
    difficulty = tf.convert_to_tensor(1)
    for symbol in tqdm(symbols):
      symbol = tf.expand_dims(symbol, 0)
      predictions, out_img, aug_img = model(symbol, difficulty)
      imgs.append(out_img[0])
    write_images(imgs, texts=texts)

  elif CFG['FULL']:
    adj_lists = [
      [[]],
      [[]],
      [[]],
      [[]],

      [[], [], [0, 1]],
      [[], [], [0, 1]],
      [[], [], [0, 1]],

      [[], [], [0, 1]],
      [[], [], [0, 1]],
      [[], [], [0, 1]],

      [[], [], [0, 1]],
      [[], [], [0, 1]],
      [[], [], [0, 1]],

      [[]],
    ]
    value_tokens_lists = [
      ['a'],
      ['b'],
      ['c'],
      ['d'],

      ['a', 'b', '*'],
      ['a', 'b', '-'],
      ['a', 'b', '-'],

      ['c', 'd', '*'],
      ['c', 'd', '-'],
      ['c', 'd', '-'],

      ['b', 'd', '*'],
      ['b', 'd', '-'],
      ['b', 'd', '-'],

      ['a'],
    ]

    order_lists = [
      [-1],
      [-1],
      [-1],
      [-1],

      [-1, -1, -1],
      [0, 1, -1],
      [1, 0, -1],

      [-1, -1, -1],
      [0, 1, -1],
      [1, 0, -1],

      [-1, -1, -1],
      [0, 1, -1],
      [1, 0, -1],

      [-1],
    ]
    token_strings = [
      'a', 'b', 'c', 'd',
      'a+b', 'a-b', 'b-a',
      'c+d', 'c-d', 'd-c',
      'b+d', 'b-d', 'd-b',
      'a'
    ]
    states, imgs, texts = get_tree_results(
      40, model, adj_lists, value_tokens_lists, order_lists, token_strings
    )
    write_images(imgs, scale_down=1, texts=texts)
    code.interact(local={**locals(), **globals()})
  else:
    assert RuntimeError(f"Nothing to animate for a model ({CFG['run_name']}) that is graph only!")
