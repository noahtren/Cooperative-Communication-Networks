"""Useful for debugging the vision pipeline, this
trains the generator and discriminator on the toy
problem of reconstructing simple one-hot vectors.
"""

import code
import random

import tensorflow as tf
import matplotlib.pyplot as plt

from vision import CPPN, ImageDecoder
from cfg import CFG

NUM_SYMBOLS = 4


def make_data():
  x = tf.random.uniform((CFG['num_samples'],), 0, NUM_SYMBOLS, dtype=tf.int32)
  x = tf.one_hot(x, depth=NUM_SYMBOLS)
  samples = tf.range(NUM_SYMBOLS)
  samples = tf.one_hot(samples, depth=NUM_SYMBOLS)
  return x, samples


@tf.function
def train_step(models, symbols):
  with tf.GradientTape(persistent=True) as tape:
    imgs = models['generator'][0](symbols)
    predictions = models['discriminator'][0](imgs)
    batch_loss = tf.keras.losses.categorical_crossentropy(symbols, predictions)
    batch_loss = tf.math.reduce_mean(batch_loss)
  for module, optim in models.values():
    grads = tape.gradient(batch_loss, module.trainable_variables)
    optim.apply_gradients(zip(grads, module.trainable_variables))
  return batch_loss


def make_decoder_classifier(decoder):
  """Takes an image decoder and adds a final classification
  layer with as many output classes as the number of symbols
  in the toy problem.
  """
  inputs = decoder.inputs
  x = decoder.outputs[0]
  x = tf.keras.layers.Dense(NUM_SYMBOLS)(x)
  x = tf.keras.layers.Activation(tf.nn.softmax)(x)
  x = tf.keras.layers.Reshape((NUM_SYMBOLS,))(x)
  model = tf.keras.Model(inputs=inputs, outputs=[x])
  return model


if __name__ == "__main__":
  data, samples = make_data()
  generator = CPPN(**CFG)
  discriminator = ImageDecoder
  discriminator = make_decoder_classifier(discriminator)
  models = {
    'generator': [generator, tf.keras.optimizers.Adam()],
    'discriminator': [discriminator, tf.keras.optimizers.Adam()],
  }
  num_batches = CFG['num_samples'] // CFG['batch_size']
  for e_i in range(CFG['epochs']):
    epoch_loss = 0
    # TRAIN
    for b_i in range(num_batches):
      end_b = min([CFG['num_samples'], (b_i + 1) * CFG['batch_size']])
      start_b = end_b - CFG['batch_size']
      batch = data[start_b:end_b]
      batch_loss = train_step(models, batch)
      epoch_loss += batch_loss
      print(f"e [{e_i}/{CFG['epochs']}] b [{end_b}/{data.shape[0]}] loss {batch_loss}", end="\r")
    epoch_loss = epoch_loss / num_batches
    print(f"EPOCH {e_i} LOSS: {epoch_loss}")
    # VISUALIZE
    if e_i // 2 == 0:
      fig, axes = plt.subplots(2, 2)
      sample_imgs = generator(samples)
      # scale tanh to visual range
      sample_img = (sample_imgs + 1) / 2
      axes[0][0].imshow(sample_imgs[0])
      axes[0][1].imshow(sample_imgs[1])
      axes[1][0].imshow(sample_imgs[2])
      axes[1][1].imshow(sample_imgs[3])
      plt.show()
