import code
import datetime
import json
import os
import io

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from graph_match import minimum_loss_permutation
from vision import Perceptor
from cfg import CFG
from graph_data import get_dataset
from ml_utils import dense_regularization, update_data_dict, normalize_data_dict
from upload import upload_data
from models import get_model, get_optim, run_dummy_batch, load_weights, save_weights


# Hyperparameters to explore:

# Regularization for each dense layer in the graph autoencoder

# Relative learning rates between different components. The full pipeline
# has 4 distinct components, each with its own potential learning rate!

# Parameter size difference between components

# Whether each module is pretrained on the smaller subtask, or starting
# from scratch.


def update_difficulty(difficulty, epoch_acc):
  # TODO: consider raising the accuracy requirement as
  # difficulty approaches maximum
  if epoch_acc['values'] >= 0.99 and difficulty < 15:
    difficulty += 1
  if epoch_acc['values'] < 0.1 and difficulty > 0:
    difficulty -= 1
  return difficulty


def predict(model, batch, difficulty, perceptor):
  if CFG['VISION']:
    adj_pred, nf_pred, imgs, aug_imgs = model(batch, difficulty)
    # TODO: implement perceptual loss here, with images returned from model
  else:
    adj_pred, nf_pred = model(batch, difficulty)
  batch_loss, acc = minimum_loss_permutation(
    batch['adj_labels'],
    batch['nf_labels'],
    adj_pred,
    nf_pred
  )
  reg_loss = {}
  for sub_model in model.layers:
    name = sub_model.name
    reg_loss[name] = tf.math.reduce_sum(sub_model.losses)
    batch_loss += reg_loss[name]
  return batch_loss, acc, reg_loss


@tf.function
def train_step(model, optim, batch, difficulty, perceptor=None):
  with tf.GradientTape(persistent=True) as tape:
    batch_loss, acc, reg_loss = predict(model, batch, difficulty, perceptor)
  grads = tape.gradient(batch_loss, model.trainable_variables)
  optim.apply_gradients(zip(grads, model.trainable_variables))
  return batch_loss, acc, reg_loss


@tf.function
def test_step(model, batch, difficulty, perceptor=None):
  batch_loss, acc, reg_loss = predict(model, batch, difficulty, perceptor)
  return batch_loss, acc, reg_loss


def get_batch(start_b, end_b, adj, node_features, adj_labels, nf_labels,
  num_nodes, **kwargs):
  batch = {
    'adj': adj[start_b:end_b],
    'node_features': {name: tensor[start_b:end_b] for
      name, tensor in node_features.items()},
    'adj_labels': adj_labels[start_b:end_b],
    'nf_labels': {name: tensor[start_b:end_b] for
      name, tensor in nf_labels.items()},
    'num_nodes': num_nodes[start_b:end_b],
  }
  return batch


if __name__ == "__main__":
  # ==================== DATA AND MODELS ====================
  train_ds = get_dataset(**CFG, test=False)
  test_ds = get_dataset(**CFG, test=True)

  model = get_model()
  run_dummy_batch(model)
  load_weights(model)
  optim = get_optim()
  # perceptor = Perceptor()
  perceptor = None

  # ==================== NOISY CHANNEL ====================
  difficulty = tf.convert_to_tensor(0)

  # ==================== LOGGING ====================
  log_dir = f"logs/{CFG['run_name']}"
  train_log_dir = f"{log_dir}/train"
  test_log_dir = f"{log_dir}/test"
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  test_summary_writer = tf.summary.create_file_writer(test_log_dir)
  current_time = str(datetime.datetime.now())
  CFG['current_time'] = current_time
  with open(os.path.join(log_dir, 'report.json'), 'w+') as f:
    f.write(json.dumps(CFG, indent=4))

  # ==================== TRAIN LOOP ====================
  tr_num_samples = train_ds['adj'].shape[0]
  test_num_samples = test_ds['adj'].shape[0]
  tr_num_batches = (tr_num_samples // CFG['batch_size']) + 1
  test_num_batches = (test_num_samples // CFG['batch_size']) + 1
  best_epoch_loss = tf.float32.max
  for e_i in range(CFG['epochs']):
    # RESET METRICS
    tr_epoch_loss = 0
    test_epoch_loss = 0
    tr_epoch_acc = {}
    test_epoch_acc = {}
    reg_loss = {}
    # TRAIN BATCHES
    for b_i in range(tr_num_batches):
      end_b = min([(b_i + 1) * CFG['batch_size'], tr_num_samples])
      start_b = end_b - CFG['batch_size']
      train_batch = get_batch(start_b, end_b, **train_ds)
      batch_loss, batch_acc, batch_reg_loss = train_step(model, optim, train_batch, difficulty, perceptor)
      tr_epoch_loss += batch_loss
      update_data_dict(tr_epoch_acc, batch_acc)
      update_data_dict(reg_loss, batch_reg_loss)
      print(f"(TRAIN) e [{e_i}/{CFG['epochs']}] b [{end_b}/{tr_num_samples}] loss {batch_loss}", end="\r")
    # TEST BATCHES
    for b_i in range(test_num_batches):
      end_b = min([(b_i + 1) * CFG['batch_size'], test_num_samples])
      start_b = end_b - CFG['batch_size']
      test_batch = get_batch(start_b, end_b, **test_ds)
      batch_loss, batch_acc, _ = test_step(model, test_batch, difficulty, perceptor)
      test_epoch_loss += batch_loss
      update_data_dict(test_epoch_acc, batch_acc)
      print(f"(TEST) e [{e_i}/{CFG['epochs']}] b [{end_b}/{test_num_samples}] loss {batch_loss}", end="\r")
    # END-OF-EPOCH METRICS
    tr_epoch_loss = tr_epoch_loss / tr_num_batches
    test_epoch_loss = test_epoch_loss / test_num_batches
    tr_epoch_acc = normalize_data_dict(tr_epoch_acc, tr_num_batches)
    test_epoch_acc = normalize_data_dict(test_epoch_acc, test_num_batches)
    reg_loss = normalize_data_dict(reg_loss, tr_num_batches)
    print(f"EPOCH {e_i} TRAIN LOSS: {tr_epoch_loss} TEST LOSS: {test_epoch_loss}")
    print(f"Train accuracies: {json.dumps(tr_epoch_acc, indent=4)}")
    print(f"Test accuracies: {json.dumps(test_epoch_acc, indent=4)}")
    print(f"Regularizer losses: {json.dumps(reg_loss, indent=4)}")
    difficulty = update_difficulty(difficulty, tr_epoch_acc)
    print(f"DIFFICULTY FOR NEXT EPOCH: {difficulty}")
    # write metrics to log
    if CFG['run_name'] != 'NOLOG':
      with train_summary_writer.as_default():
        tf.summary.scalar('loss', tr_epoch_loss, step=e_i)
        for name, metric in tr_epoch_acc.items():
          tf.summary.scalar(name, metric, step=e_i)
      with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_epoch_loss, step=e_i)
        for name, metric in test_epoch_acc.items():
          tf.summary.scalar(name, metric, step=e_i)
    # SAVE CHECKPOINTS
    if test_epoch_loss < best_epoch_loss:
      best_epoch_loss = test_epoch_loss
      save_weights(model)
    # GENERATE VISUAL SAMPLE
    if CFG['VISION'] and e_i % 2 == 0:
      fig, axes = plt.subplots(4, 2)
      sample_idxs = tf.random.uniform([CFG['batch_size']], 0, test_num_samples, tf.int32)
      vis_batch = {
        'adj': tf.gather(test_ds['adj'], sample_idxs),
        'node_features': {name: tf.gather(tensor, sample_idxs) for
          name, tensor in test_ds['node_features'].items()},
        'adj_labels': tf.gather(test_ds['adj_labels'], sample_idxs),
        'nf_labels': {name: tf.gather(tensor, sample_idxs) for
          name, tensor in test_ds['nf_labels'].items()},
        'num_nodes': tf.gather(test_ds['num_nodes'], sample_idxs),
      }
      _, _, sample_imgs, aug_imgs = model(vis_batch, difficulty)
      # scale tanh to visual range
      sample_imgs = (sample_imgs + 1) / 2
      aug_imgs = (aug_imgs + 1) / 2
      print(f"0: {tf.math.reduce_mean(sample_imgs[0])}")
      print(f"1: {tf.math.reduce_mean(sample_imgs[1])}")
      print(f"2: {tf.math.reduce_mean(sample_imgs[2])}")
      print(f"3: {tf.math.reduce_mean(sample_imgs[3])}")
      for img_i in range(4):
        axes[img_i][0].imshow(sample_imgs[img_i])
        axes[img_i][1].imshow(aug_imgs[img_i])
        
      img_data = io.BytesIO()
      img_name = f"gallery/{CFG['run_name']}/{e_i}.png"
      plt.savefig(img_data, format='png')
      img_data.seek(0)
      upload_data(img_name, img_data)
      plt.clf()
      plt.cla()
      plt.close()
