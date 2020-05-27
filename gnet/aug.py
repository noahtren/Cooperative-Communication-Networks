import code
import random

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt


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


class DifferentiableAugment:
  """Collection of image augmentation functions implmented in pure TensorFlow,
  so that they are fully differentiable (gradients are not lost when applied.)
  
  Each augmentation function takes a tensor of images and a difficulty value
  from 0 to 15. For curriculum learning, the difficulty can be increased slowly
  over time.
  """


  @staticmethod
  def static(imgs, DIFFICULTY, randomize):
    """Gaussian noise, or "static"
    """
    STATIC_STDDEVS = {
      0: 0.00,
      1: 0.03,
      2: 0.06,
      3: 0.1,
      4: 0.13,
      5: 0.16,
      6: 0.2,
      7: 0.23,
      8: 0.26,
      9: 0.3,
      10: 0.33,
      11: 0.36,
      12: 0.4,
      13: 0.43,
      14: 0.46,
      15: 0.50,
    }
    img_shape = imgs[0].shape
    batch_size = imgs.shape[0]
    stddev = STATIC_STDDEVS[DIFFICULTY]
    stddev = tf.random.uniform([], 0, stddev)
    noise = tf.random.normal((batch_size, *img_shape), mean=0, stddev=stddev)
    imgs = imgs + noise
    return imgs


  @staticmethod
  def translate(imgs, DIFFICULTY, randomize):
    """Shift each img a pixel distance from minval to maxval, using zero padding
    For efficiency, each batch is augmented in the same way, but randomized
    between batches.
    """
    SHIFT_PERCENTS = {
      0: 0.00,
      1: 0.025,
      2: 0.05,
      3: 0.07,
      4: 0.09,
      5: 0.1,
      6: 0.11,
      7: 0.12,
      8: 0.13,
      9: 0.14,
      10: 0.15,
      11: 0.16,
      12: 0.17,
      13: 0.18,
      14: 0.19,
      15: 0.20
    }
    img_shape = imgs[0].shape
    batch_size = imgs.shape[0]
    max_shift_percent = SHIFT_PERCENTS[DIFFICULTY]
    average_img_dim = (img_shape[0] + img_shape[1]) / 2
    minval = int(round(max_shift_percent * average_img_dim * -1))
    maxval = int(round(max_shift_percent * average_img_dim))
    shifts = tf.random.uniform([imgs.shape[0], 2], minval=minval, maxval=maxval + 1, dtype=tf.int32)
    shifts = tf.cast(shifts, tf.float32)
    imgs = tfa.image.translate(imgs, shifts, interpolation='BILINEAR')
    return imgs
  

  @staticmethod
  def resize(imgs, DIFFICULTY, randomize):
    """Resize an image, either shrinking it or growing it. This will cause some
    regions of the image to be occluded, in the case that it is grown.
    """
    RESIZE_SCALES = {
      0: [1, 1],
      1: [0.9, 1.1],
      2: [0.85, 1.15],
      3: [0.8, 1.2],
      4: [0.75, 1.25],
      5: [0.7, 1.3],
      6: [0.65, 1.325],
      7: [0.6, 1.35],
      8: [0.55, 1.375],
      9: [0.50, 1.4],
      10: [0.48, 1.42],
      11: [0.46, 1.44],
      12: [0.44, 1.46],
      13: [0.42, 1.48],
      14: [0.4, 1.5],
      15: [0.4, 1.5],
    }
    img_shape = imgs[0].shape
    minscale, maxscale = RESIZE_SCALES[DIFFICULTY]
    scales = tf.random.uniform([2], minval=minscale, maxval=maxscale)
    target_width = tf.cast(scales[0] * img_shape[1], tf.int32)
    target_height = tf.cast(scales[1] * img_shape[0], tf.int32)
    imgs = tf.image.resize(imgs, (target_height, target_width))
    imgs = tf.image.resize_with_crop_or_pad(imgs, img_shape[0], img_shape[1])
    return imgs
  

  @staticmethod
  def fractional_rotate(imgs, DIFFICULTY, randomize):
    """Rotate images, each with a unique number of radians selected uniformly
    from a range.
    Note: this function is not TPU-friendly
    """
    pi = 3.14159265
    RADIANS = {
      0: 0,
      1: 1 * pi / 30,
      2: 2 * pi / 30,
      3: 3 * pi / 30,
      4: 4 * pi / 30,
      5: 5 * pi / 30,
      6: 6 * pi / 30,
      7: 7 * pi / 30,
      8: 8 * pi / 30,
      9: 9 * pi / 30,
      10: 9.5 * pi / 30,
      11: 10 * pi / 30,
      12: 10.5 * pi / 30,
      13: 11 * pi / 30,
      14: 11.5 * pi / 30,
      15: 12 * pi / 30,
    }
    min_angle = RADIANS[DIFFICULTY] * -1
    max_angle = RADIANS[DIFFICULTY]
    batch_size = imgs.shape[0]
    angles = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)
    imgs = tfa.image.rotate(imgs, angles, interpolation='BILINEAR')
    return imgs


  @staticmethod
  def blur(imgs, DIFFICULTY, randomize):
    """Apply blur via a Gaussian convolutional kernel
    """
    STDDEVS = {
      0: 0.01,
      1: 0.2,
      2: 0.4,
      3: 0.6,
      4: 0.8,
      5: 0.9,
      6: 1.0,
      7: 1.25,
      8: 1.5,
      9: 1.6,
      10: 1.7,
      11: 1.8,
      12: 1.9,
      13: 2.0,
      14: 2.1,
      15: 2.3
    }
    img_shape = imgs[0].shape
    c = img_shape[2]
    stddev = STDDEVS[DIFFICULTY]
    stddev = tf.random.uniform([], 0, stddev)
    gauss_kernel = gaussian_k(7, 7, 3, 3, stddev)

    # expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    
    # convolve
    out_channels = []
    for c_ in range(c):
      in_channel = tf.expand_dims(imgs[:, :, :, c_], -1)
      out_channel = tf.nn.conv2d(in_channel, gauss_kernel, strides=1, padding="SAME")
      out_channel = tf.squeeze(out_channel)
      out_channels.append(out_channel)
    imgs = out_channels[0] if len(out_channels) == 1 else tf.stack(out_channels, axis=-1)
    return imgs


  @staticmethod
  def random_scale(imgs, DIFFICULTY, randomize):
    """Randomly scales all of the values in each channel
    """
    MULTIPLY_SCALES = {
      0: [1, 1],
      1: [0.9, 1.1],
      2: [0.85, 1.15],
      3: [0.8, 1.2],
      4: [0.75, 1.25],
      5: [0.7, 1.3],
      6: [0.65, 1.325],
      7: [0.6, 1.35],
      8: [0.55, 1.375],
      9: [0.50, 1.4],
      10: [0.48, 1.42],
      11: [0.46, 1.44],
      12: [0.44, 1.46],
      13: [0.42, 1.48],
      14: [0.4, 1.5],
      15: [0.35, 1.6],
    }
    channels = imgs.shape[-1]
    minscale, maxscale = MULTIPLY_SCALES[DIFFICULTY]
    scales = tf.random.uniform([channels], minval=minscale, maxval=maxscale)
    imgs = imgs * scales
    return imgs


  @staticmethod
  def cutout(imgs, DIFFICULTY, randomize):
    MASK_PERCENT = {
      0: 0.00,
      1: 0.075,
      2: 0.1,
      3: 0.125,
      4: 0.15,
      5: 0.175,
      6: 0.2,
      7: 0.225,
      8: 0.25,
      9: 0.275,
      10: 0.3,
      11: 0.325,
      12: 0.34,
      13: 0.36,
      14: 0.38,
      15: 0.4,
    }
    y_size = tf.random.uniform([], 0, MASK_PERCENT[DIFFICULTY])
    x_size = tf.random.uniform([], 0, MASK_PERCENT[DIFFICULTY])
    y_size = tf.cast(imgs.shape[1] * x_size / 2, tf.int32) * 2
    x_size = tf.cast(imgs.shape[2] * y_size / 2, tf.int32) * 2
    for _ in range(2):
      imgs = tfa.image.random_cutout(
        imgs,
        mask_size=[y_size, x_size]
      )
    return imgs

  @staticmethod
  def sharp_tanh(imgs, DIFFICULTY, randomize):
    TANH_AMT = {
      0: 1.0,
      1: 1.1,
      2: 1.2,
      3: 1.3,
      4: 1.4,
      5: 1.5,
      6: 1.6,
      7: 1.7,
      8: 1.8,
      9: 1.9,
      10: 2.0,
      11: 2.1,
      12: 2.2,
      13: 2.3,
      14: 2.4,
      15: 2.5,
    }
    tanh_amt = tf.random.uniform([], 0, TANH_AMT[DIFFICULTY])
    imgs = tf.nn.tanh(imgs * tanh_amt) * 1.31
    return imgs


def get_noisy_channel(func_names=[
    'static',
    'blur',
    'cutout', # tfa
    'random_scale',
    'translate', # tfa
    'resize',
    'fractional_rotate', # tfa
    'sharp_tanh'
  ]):
  """Return a function that adds noise to a batch of images, conditioned on a
  difficulty value.
  """
  @tf.function
  def noise_pipeline(images, funcs, DIFFICULTY):
    """Apply a series of functions to images, in order
    """
    if DIFFICULTY == 0:
      return images
    else:
      for func in funcs:
        images = func(images, DIFFICULTY, randomize=True)
      return images
  funcs = []
  for func_name in func_names:
    assert func_name in dir(DifferentiableAugment), f"Function '{func_name}' doesn't exist"
    funcs.append(getattr(DifferentiableAugment, func_name))
  return lambda images, DIFFICULTY: noise_pipeline(images, funcs, DIFFICULTY)


if __name__ == "__main__":
  import os
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=""  # specify which GPU(s) to be used

  imgs = tf.random.normal([4, 128, 128, 3])
  channel = get_noisy_channel()
  for diff in range(16):
    x = channel(imgs, diff)
    fig, axes = plt.subplots(2, 2)
    # scale x to be readable
    x = x - tf.math.reduce_min(x)
    x = x / tf.math.reduce_max(x)
    axes[0][0].imshow(x[0])
    axes[0][1].imshow(x[1])
    axes[1][0].imshow(x[2])
    axes[1][1].imshow(x[3])
    plt.savefig(f"noise_{diff}.png")
