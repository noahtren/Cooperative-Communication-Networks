import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""  # specify which GPU(s) to be used


import code
from typing import List

import tensorflow as tf
import numpy as np
import imageio
from tqdm import tqdm

from vision import Generator, Decoder
from train_vision import make_data
from cfg import CFG
from ml_utils import load_ckpts


def make_sliding_window_data(points:List[int], steps_between:int):
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


if __name__ == "__main__":
  data = make_sliding_window_data([6,17,18,6], 30)
  generator = Generator()
  generator(tf.expand_dims(data[0], 0))
  models = {'generator': [generator, 0]}
  load_ckpts(models, CFG['load_name'], 'latest')
  writer = imageio.get_writer(f"gallery/{CFG['load_name']}_animation.mp4", format='FFMPEG', fps=10)
  for i, val in tqdm(enumerate(data)):
    val = tf.expand_dims(val, 0)
    img = generator(val)
    img = tf.squeeze((img + 1) / 2)
    img = tf.cast(img * 255, tf.uint8)
    writer.append_data(np.array(img))
  writer.close()
