import code
import datetime
import json
import os

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from graph_data import TensorGraph
from encoder import Encoder 
from decoder import GraphDecoder
from vision import CPPN, ImageDecoder, modify_decoder
from graph_match import minimum_loss_permutation
from cfg import CFG
from ml_utils import dense_regularization


# TODO: a way to evaluate, at least statistically, how much the training and
# testing datasets overlap.

# NOTE: consider pre-training the graph autoencoder, and then squishing the
# vision modules into the latent space afterwards. may be easier to learn than
# doing everything end-to-end from scratch.


"""Documentation of hyperparameters

Regularization for each dense layer in the graph autoencoder
"""


def get_dataset(language_spec:str, min_num_values:int, max_num_values:int,
                max_nodes:int, num_samples:int, vis_first=False, test=False, **kwargs):
  instances = []
  print(f"Generating {'test' if test else 'training'} dataset")
  num_samples = num_samples // 5 if test else num_samples
  pbar = tqdm(total=num_samples)
  while len(instances) < num_samples:
    # make tree
    adj_list, values_list, value_tokens_list, order_list = \
      TensorGraph.random_tree(language_spec, min_num_values, max_num_values)
    # convert to instance
    instance = TensorGraph.tree_to_instance(adj_list, values_list, value_tokens_list,
      order_list, max_nodes, language_spec)
    if instance is not None:
      instances.append(instance)
      pbar.update(1)
      if vis_first and len(instances) == 1:
        instance.visualize()
  pbar.close()
  adj, node_features, node_feature_specs, num_nodes = TensorGraph.instances_to_tensors(instances)

  # postprocessing for data labels
  nf_labels = {}
  for name in node_features.keys():
    empty_rows = (tf.math.reduce_max(node_features[name], axis=-1) == 0)[:, :, tf.newaxis]
    empty_rows = tf.cast(empty_rows, tf.float32)
    nf_labels[name] = tf.concat([node_features[name], empty_rows], axis=-1)

  adj_labels = adj + tf.eye(adj.shape[1], adj.shape[1],
                            batch_shape=[adj.shape[0]], dtype=tf.int32)

  return adj, node_features, node_feature_specs, num_nodes, adj_labels, nf_labels


def predict(models, batch):
  x = models['encoder'][0](batch)
  if 'generator' in models:
    img = models['generator'][0](x)
    x = models['discriminator'][0](img)
  adj_pred, nf_pred = models['decoder'][0](x)
  batch_loss, acc = minimum_loss_permutation(
    batch['adj_labels'],
    batch['nf_labels'],
    adj_pred,
    nf_pred
  )
  return batch_loss, acc


@tf.function
def train_step(models, batch, test=False):
  if test:
    batch_loss, acc = predict(models, batch)
    return batch_loss, acc
  else:
    reg_loss = {}
    with tf.GradientTape(persistent=True) as tape:
      batch_loss, acc = predict(models, batch)
      for name, (model, _) in models.items():
        reg_loss[name] = tf.math.reduce_sum(model.losses)
        batch_loss += reg_loss[name]
    for model, optim in models.values():
      grads = tape.gradient(batch_loss, model.trainable_variables)
      optim.apply_gradients(zip(grads, model.trainable_variables))
  return batch_loss, acc, reg_loss


def update_data_dict(data_dict, batch_dict):
  for name in batch_dict.keys():
    if name not in data_dict:
      data_dict[name] = batch_dict[name]
    else:
      data_dict[name] += batch_dict[name]
  return data_dict


def normalize_data_dict(data_dict, num_batches):
  for name, value in data_dict.items():
    data_dict[name] = (value / num_batches).numpy().item()
  return data_dict


def save_ckpts(models, log_dir, ckpt_name:str):
  for model_name, model in models.items():
    model_path = os.path.join(log_dir, model_name, ckpt_name)
    os.makedirs(model_path, exist_ok=True)
    model[0].save_weights(model_path)


def dummy_batch(models,
                tr_adj, tr_node_features, tr_adj_labels, tr_nf_labels, tr_num_nodes):
  """Only for graph model. Used to populate weights before loading.
  """
  batch = {
    'adj': tr_adj[:CFG['batch_size']],
    'node_features': {name: tensor[:CFG['batch_size']] for
      name, tensor in tr_node_features.items()},
    'adj_labels': tr_adj_labels[:CFG['batch_size']],
    'nf_labels': {name: tensor[:CFG['batch_size']] for
      name, tensor in tr_nf_labels.items()},
    'num_nodes': tr_num_nodes[:CFG['batch_size']],
  }
  x = models['encoder'][0](batch)
  adj_pred, nf_pred = models['decoder'][0](x)


def load_ckpts(models, load_name, ckpt_name='best'):
  log_dir = f"logs/{load_name}"
  for model_name, (model, _) in models.items():
    model_path = os.path.join(log_dir, model_name, ckpt_name)
    if os.path.exists(model_path):
      model.load_weights(model_path)
      print(f"Loaded weights for {model_name}")
  if CFG['VISION']:
    models['decoder'][0].expand_w = \
      tf.keras.layers.Dense(CFG['max_nodes'] * CFG['hidden_size'], **dense_regularization)
    print("Overrode decoder input layer (for vision compatibility)")


if __name__ == "__main__":
  # ==================== DATA AND MODELS ====================
  tr_adj, tr_node_features, tr_node_feature_specs, tr_num_nodes, \
  tr_adj_labels, tr_nf_labels = get_dataset(**CFG, test=False)

  test_adj, test_node_features, test_node_feature_specs, test_num_nodes, \
  test_adj_labels, test_nf_labels = get_dataset(**CFG, test=True)

  CFG['node_feature_specs'] = tr_node_feature_specs
  encoder = Encoder(**CFG)
  decoder = GraphDecoder(**CFG)
  lr = CFG['initial_lr']
  if CFG['use_exponential_rate_scheduler']:
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
      CFG['initial_lr'], decay_steps=100_000, decay_rate=0.96, staircase=True)
  models = {
    'encoder': [encoder, tf.keras.optimizers.Adam(lr)],
    'decoder': [decoder, tf.keras.optimizers.Adam(lr)],
  }
  if CFG['VISION']:
    generator = CPPN(**CFG)
    discriminator = modify_decoder(ImageDecoder)
    gen_lr = tf.keras.optimizers.schedules.ExponentialDecay(
      CFG['generator_lr'], decay_steps=100_000, decay_rate=0.96, staircase=True)
    disc_lr = tf.keras.optimizers.schedules.ExponentialDecay(
      CFG['discriminator_lr'], decay_steps=100_000, decay_rate=0.96, staircase=True)
    models['generator'] = [generator, tf.keras.optimizers.Adam(gen_lr)]
    models['discriminator'] = [discriminator, tf.keras.optimizers.Adam(disc_lr)]
  if CFG['load_name'] is not None:
    dummy_batch(models, tr_adj, tr_node_features, tr_adj_labels, tr_nf_labels, tr_num_nodes)
    load_ckpts(models, CFG['load_name'])

  # ==================== LOGGING ====================
  log_dir = f"logs/{CFG['run_name']}"
  train_log_dir = f"{log_dir}/train"
  test_log_dir = f"{log_dir}/test"
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  test_summary_writer = tf.summary.create_file_writer(test_log_dir)
  # os.makedirs(train_log_dir)
  # os.makedirs(test_log_dir)
  current_time = str(datetime.datetime.now())
  CFG['current_time'] = current_time
  with open(os.path.join(log_dir, 'report.json'), 'w+') as f:
    f.write(json.dumps(CFG, indent=4))

  # ==================== TRAIN LOOP ====================
  tr_num_batches = (tr_adj.shape[0] // CFG['batch_size']) + 1
  test_num_batches = (test_adj.shape[0] // CFG['batch_size']) + 1
  best_epoch_loss = tf.float32.max
  for e_i in range(CFG['epochs']):
    # RESET METRICS
    tr_epoch_loss = 0
    test_epoch_loss = 0
    tr_epoch_acc = {}
    test_epoch_acc = {}
    reg_loss = {}
    # TRAIN BATCHES =============================
    for b_i in range(tr_num_batches):
      end_b = min([(b_i + 1) * CFG['batch_size'], tr_adj.shape[0]])
      start_b = end_b - CFG['batch_size']
      batch = {
        'adj': tr_adj[start_b:end_b],
        'node_features': {name: tensor[start_b:end_b] for
          name, tensor in tr_node_features.items()},
        'adj_labels': tr_adj_labels[start_b:end_b],
        'nf_labels': {name: tensor[start_b:end_b] for
          name, tensor in tr_nf_labels.items()},
        'num_nodes': tr_num_nodes[start_b:end_b],
      }
      batch_loss, batch_acc, batch_reg_loss = train_step(models, batch)
      tr_epoch_loss += batch_loss
      update_data_dict(tr_epoch_acc, batch_acc)
      update_data_dict(reg_loss, batch_reg_loss)
      print(f"(TRAIN) e [{e_i}/{CFG['epochs']}] b [{end_b}/{tr_adj.shape[0]}] loss {batch_loss}", end="\r")
    # TEST BATCHES =============================
    for b_i in range(test_num_batches):
      end_b = min([(b_i + 1) * CFG['batch_size'], test_adj.shape[0]])
      start_b = end_b - CFG['batch_size']
      batch = {
        'adj': test_adj[start_b:end_b],
        'node_features': {name: tensor[start_b:end_b] for
          name, tensor in test_node_features.items()},
        'adj_labels': test_adj_labels[start_b:end_b],
        'nf_labels': {name: tensor[start_b:end_b] for
          name, tensor in test_nf_labels.items()},
        'num_nodes': test_num_nodes[start_b:end_b],
      }
      batch_loss, batch_acc = train_step(models, batch, test=True)
      test_epoch_loss += batch_loss
      update_data_dict(test_epoch_acc, batch_acc)
      print(f"(TEST) e [{e_i}/{CFG['epochs']}] b [{end_b}/{test_adj.shape[0]}] loss {batch_loss}", end="\r")
    # END OF EPOCH METRICS
    tr_epoch_loss = tr_epoch_loss / tr_num_batches
    test_epoch_loss = test_epoch_loss / test_num_batches
    tr_epoch_acc = normalize_data_dict(tr_epoch_acc, tr_num_batches)
    test_epoch_acc = normalize_data_dict(test_epoch_acc, test_num_batches)
    reg_loss = normalize_data_dict(reg_loss, tr_num_batches)
    print(f"EPOCH {e_i} TRAIN LOSS: {tr_epoch_loss} TEST LOSS: {test_epoch_loss}")
    print(f"Train accuracies: {json.dumps(tr_epoch_acc, indent=4)}")
    print(f"Test accuracies: {json.dumps(test_epoch_acc, indent=4)}")
    print(f"Regularizer losses: {json.dumps(reg_loss, indent=4)}")
    if CFG['run_name'] != 'NOLOG':
      with train_summary_writer.as_default():
        tf.summary.scalar('loss', tr_epoch_loss, step=e_i)
        for name, metric in tr_epoch_acc.items():
          tf.summary.scalar(name, metric, step=e_i)
      with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_epoch_loss, step=e_i)
        for name, metric in test_epoch_acc.items():
          tf.summary.scalar(name, metric, step=e_i)
    # SAVING CHECKPOINTS
    # save_ckpts(models, log_dir, str(e_i))
    if test_epoch_loss < best_epoch_loss:
      best_epoch_loss = test_epoch_loss
      save_ckpts(models, log_dir, 'best')
    if CFG['VISION'] and e_i % 2 == 0:
      fig, axes = plt.subplots(2, 2)
      sample_idxs = tf.random.uniform([CFG['batch_size']], 0, CFG['num_samples'], tf.int32)
      vis_batch = {
        'adj': tf.gather(test_adj, sample_idxs),
        'node_features': {name: tf.gather(tensor, sample_idxs) for
          name, tensor in test_node_features.items()},
        'adj_labels': tf.gather(test_adj_labels, sample_idxs),
        'nf_labels': {name: tf.gather(tensor, sample_idxs) for
          name, tensor in test_nf_labels.items()},
        'num_nodes': tf.gather(test_num_nodes, sample_idxs),
      }
      Zs = models['encoder'][0](vis_batch)
      sample_imgs = models['generator'][0](Zs)
      # scale tanh to visual range
      sample_imgs = (sample_imgs + 1) / 2
      axes[0][0].imshow(sample_imgs[0])
      axes[0][1].imshow(sample_imgs[1])
      axes[1][0].imshow(sample_imgs[2])
      axes[1][1].imshow(sample_imgs[3])
      plt.savefig(f"{e_i}.png")
