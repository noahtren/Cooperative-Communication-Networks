"""Useful for debugging the vision pipeline, this
trains the generator and discriminator on the toy
problem of reconstructing simple one-hot vectors.
"""

import code
import random
import json

import tensorflow as tf
import matplotlib.pyplot as plt

from vision import Generator, ImageDecoder, modify_decoder, perceptual_loss
from cfg import CFG
from aug import get_noisy_channel
from main import save_ckpts, load_ckpts
from ml_utils import shuffle_together, update_data_dict, normalize_data_dict


NUM_SYMBOLS = 512


"""Hyperparameters

Image augmentation: the types of image augmentation to do, as well as the
curriculum to follow for changes to the image augmentation pipeline.

Image properties: size of images and number of channels.
Also of interest is circular images, which may have interesting properties.

Generative Models: generator as a CPPN, or some other generative model.
* the generative model should be scalable to large batch sizes while
still remaining stable. Stability seems like a major problem for these
kinds of things.

Discriminative Model: Using inception as the backbone, or some other thing?
Maybe:
  * ResNet
  * VGG

Optimizers: which optimizers to use and what learning rate to use

Number of symbols: in the case of the toy problem, the number of samples
can increase difficulty since it becomes harder to differentiate between
images.

TODO: I would like to learn a bit more about optimizer scheduling,
especially things like warmup periods, which seem important so that
exploding gradients don't nuke the results.

CONSIDER: starting with a pretrained generator.

In general, I need to explore the stricty visual space quite a bit more
before trying to adapt it to the graph problem. It's definitely the limiting
factor right now.

PROBLEM: compilation times take a long time, I think it is for compiling the
inception architecture.

NOTE: perceptual loss is quite nice and works pretty well, I can use the same
starting pretrained model for the discrminator and the perceptual loss model.
"""


def make_data():
  x = tf.random.uniform((CFG['num_samples'],), 0, NUM_SYMBOLS, dtype=tf.int32)
  x = tf.one_hot(x, depth=NUM_SYMBOLS)
  samples = tf.range(NUM_SYMBOLS)
  samples = tf.one_hot(samples, depth=NUM_SYMBOLS)
  return x, samples



def update_difficulty(difficulty, epoch_loss, epoch_acc):
  if epoch_acc > 0.8 and difficulty < 15:
    difficulty += 1
  if epoch_acc < 0.1 and difficulty > 0:
    difficulty -= 1
  return difficulty


@tf.function
def train_step(models, symbols, noisy_channel, difficulty, e_i):
  reg_loss = {}
  with tf.GradientTape(persistent=True) as tape:
    batch_loss = 0
    imgs = models['generator'][0](symbols)
    repel_loss = perceptual_loss(imgs)
    reg_loss['perceptual'] = repel_loss
    batch_loss += repel_loss
    imgs = noisy_channel(imgs, difficulty)
    predictions = models['discriminator'][0](imgs)
    batch_loss += tf.keras.losses.categorical_crossentropy(symbols, predictions, label_smoothing=CFG['label_smoothing'])
    for name, (model, _) in models.items():
      reg_loss[name] = tf.math.reduce_sum(model.losses)
      batch_loss += reg_loss[name]
    batch_loss = tf.math.reduce_mean(batch_loss)
    acc = tf.keras.metrics.categorical_accuracy(symbols, predictions)
    acc = tf.math.reduce_mean(acc)
  for module, optim in models.values():
    grads = tape.gradient(batch_loss, module.trainable_variables)
    optim.apply_gradients(zip(grads, module.trainable_variables))
  return batch_loss, acc, reg_loss


def dummy_input(models, data):
  symbols = data[:CFG['batch_size']]
  imgs = models['generator'][0](symbols)
  predictions = models['discriminator'][0](imgs)



def main():
  # ==================== DATA AND MODELS ====================
  data, samples = make_data()
  generator = Generator()
  discriminator = modify_decoder(ImageDecoder, just_GAP=False, NUM_SYMBOLS=NUM_SYMBOLS)
  models = {
    'generator': [generator, tf.keras.optimizers.Adam(lr=CFG['generator_lr'])],
    'discriminator': [discriminator, tf.keras.optimizers.Adam(lr=CFG['discriminator_lr'])],
  }
  dummy_input(models, data)
  if CFG['load_name'] is not None:
    load_ckpts(models, CFG['load_name'], 'latest')
  noisy_channel = get_noisy_channel()
  num_batches = CFG['num_samples'] // CFG['batch_size']
  difficulty = 0
  # ==================== TRAINING LOOP ====================
  for e_i in range(CFG['epochs']):
    epoch_loss = 0
    acc = 0
    reg_loss = {}
    data = shuffle_together(data)[0]
    # TRAIN ====================
    for b_i in range(num_batches):
      end_b = min([CFG['num_samples'], (b_i + 1) * CFG['batch_size']])
      start_b = end_b - CFG['batch_size']
      batch = data[start_b:end_b]
      batch_loss, batch_acc, batch_reg_loss = train_step(models, batch, noisy_channel, difficulty, e_i)
      epoch_loss += batch_loss
      acc += batch_acc
      update_data_dict(reg_loss, batch_reg_loss)
      print(f"e [{e_i}/{CFG['epochs']}] b [{end_b}/{data.shape[0]}] loss {batch_loss}", end="\r")
    epoch_loss = epoch_loss / num_batches
    acc = acc / num_batches
    reg_loss = normalize_data_dict(reg_loss, num_batches)
    print(f"EPOCH {e_i} LOSS: {epoch_loss} ACCURACY: {acc}")
    print(f"Regularizer losses: {json.dumps(reg_loss, indent=4)}")
    difficulty = update_difficulty(difficulty, epoch_loss, acc)
    print(f"DIFFICULTY FOR NEXT EPOCH: {difficulty}")
    # SAVE WEIGHTS ====================
    save_ckpts(models, f"logs/{CFG['run_name']}", 'latest')
    # VISUALIZE ====================
    if e_i % 2 == 0:
      fig, axes = plt.subplots(2, 2)
      sample_idxs = tf.random.uniform([CFG['batch_size']], 0, NUM_SYMBOLS, tf.int32)
      these_samples = tf.gather(samples, sample_idxs)
      sample_imgs = generator(these_samples)
      # scale tanh to visual range
      sample_imgs = (sample_imgs + 1) / 2
      axes[0][0].imshow(sample_imgs[0])
      axes[0][1].imshow(sample_imgs[1])
      axes[1][0].imshow(sample_imgs[2])
      axes[1][1].imshow(sample_imgs[3])
      plt.savefig(f"{e_i}.png")


if __name__ == "__main__":
  main()
