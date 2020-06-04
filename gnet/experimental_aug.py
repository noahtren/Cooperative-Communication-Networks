"""Originally from https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96#Data-Augmentation,
modified to augment a batch of images rather than a single image. (For a tf.data
pipeline, you may want to use the original code at the link above.)
"""

import math
import code

import tensorflow as tf
import matplotlib.pyplot as plt

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
  """Return a 3x3 transformmatrix which transforms indicies of original images
  """

  # CONVERT DEGREES TO RADIANS
  rotation = math.pi * rotation / 180.
  shear = math.pi * shear / 180.

  # ROTATION MATRIX
  c1 = tf.math.cos(rotation)
  s1 = tf.math.sin(rotation)
  rotation_matrix = tf.reshape(tf.concat([c1,s1,[0], -s1,c1,[0], [0],[0],[1]],axis=0),[3,3])

  # SHEAR MATRIX
  c2 = tf.math.cos(shear)
  s2 = tf.math.sin(shear)
  shear_matrix = tf.reshape(tf.concat([[1],s2,[0], [0],c2,[0], [0],[0],[1]],axis=0),[3,3])    
  
  # ZOOM MATRIX
  zoom_matrix = tf.reshape( tf.concat([[1]/height_zoom,[0],[0], [0],[1]/width_zoom,[0], [0],[0],[1]],axis=0),[3,3])
  
  # SHIFT MATRIX
  shift_matrix = tf.reshape( tf.concat([[1],[0],height_shift, [0],[1],width_shift, [0],[0],[1]],axis=0),[3,3])

  return tf.matmul(tf.matmul(rotation_matrix, shear_matrix), tf.matmul(zoom_matrix, shift_matrix))


def transform_batch(images,
                    max_rot_deg,
                    max_shear_deg,
                    max_zoom_diff_pct,
                    max_shift_pct,
                    experimental_tpu_efficiency=True):
  """Transform a batch of square images with the same randomized affine
  transformation.
  """

  def clipped_random():
    rand = tf.random.normal([1], dtype=tf.float32)
    rand = tf.clip_by_value(rand, -2., 2.) / 2.
    return rand

  batch_size = images.shape[0]
  tf.debugging.assert_equal(
    images.shape[1],
    images.shape[2],
    "Images should be square")
  DIM = images.shape[1]
  XDIM = DIM % 2

  rot = max_rot_deg * clipped_random()
  shr = max_shear_deg * clipped_random() 
  h_zoom = 1.0 + clipped_random()*max_zoom_diff_pct
  w_zoom = 1.0 + clipped_random()*max_zoom_diff_pct
  h_shift = clipped_random()*(DIM*max_shift_pct)
  w_shift = clipped_random()*(DIM*max_shift_pct)

  # GET TRANSFORMATION MATRIX
  m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

  # LIST DESTINATION PIXEL INDICES
  x = tf.repeat(tf.range(DIM//2,-DIM//2,-1), DIM) # 10000,
  y = tf.tile(tf.range(-DIM//2,DIM//2),[DIM])
  z = tf.ones([DIM*DIM],tf.int32)
  idx = tf.stack( [x,y,z] ) # [3, 10000]

  # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
  idx2 = tf.matmul(m,tf.cast(idx,tf.float32))
  idx2 = tf.cast(idx2,tf.int32)
  idx2 = tf.clip_by_value(idx2,-DIM//2+XDIM+1,DIM//2)

  # FIND ORIGIN PIXEL VALUES           
  idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
  idx3 = tf.transpose(idx3)
  batched_idx3 = tf.tile(idx3[tf.newaxis], [batch_size, 1, 1])

  if experimental_tpu_efficiency:
    # This reduces excessive padding in the original tf.gather_nd op
    idx4 = idx3[:, 0] * DIM + idx3[:, 1]
    images = tf.reshape(images, [batch_size, DIM * DIM, 3])
    d = tf.gather(images, idx4, axis=1)
    return tf.reshape(d, [batch_size,DIM,DIM,3])
  else:
    d = tf.gather_nd(images, batched_idx3, batch_dims=1)
    return tf.reshape(d,[batch_size,DIM,DIM,3])


if __name__ == "__main__":
  import os
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=""
  x = tf.random.normal((4, 100, 100, 3))
  x = x - tf.math.reduce_min(x)
  x = x / tf.math.reduce_max(x)
  x_aug = transform(x)

  fig, axes = plt.subplots(4, 2)

  for b in range(4):
    img = x[b]
    img_aug = x_aug[b]
    axes[b][0].imshow(img)
    axes[b][1].imshow(img_aug)

  plt.show()