import code
import datetime
import json
import os

import tensorflow as tf
from tqdm import tqdm

from graph_data import TensorGraph
from encoder import Encoder 
from decoder import GraphDecoder
from graph_match import minimum_loss_permutation
from cfg import CFG


# TODO: a way to evaluate, at least statistically, how much the training and
# testing datasets overlap.


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


@tf.function
def train_step(models, batch, test=False):
  if test:
    x = models['encoder'][0](batch)
    adj_pred, nf_pred = models['decoder'][0](x)
    batch_loss = minimum_loss_permutation(
      batch['adj_labels'],
      batch['nf_labels'],
      adj_pred,
      nf_pred
    )
    return batch_loss
  else:
    with tf.GradientTape(persistent=True) as tape:
      x = models['encoder'][0](batch)
      adj_pred, nf_pred = models['decoder'][0](x)
      batch_loss = minimum_loss_permutation(
        batch['adj_labels'],
        batch['nf_labels'],
        adj_pred,
        nf_pred
      )
    for module, optim in models.values():
      grads = tape.gradient(batch_loss, module.trainable_variables)
      optim.apply_gradients(zip(grads, module.trainable_variables))
  return batch_loss


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
      CFG['initial_lr'], decay_steps=100_000, decay_rate=0.96, staircase=True
    )
  models = {
    'encoder': [encoder, tf.keras.optimizers.Adam(lr)],
    'decoder': [decoder, tf.keras.optimizers.Adam(lr)],
  }

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
  for e_i in range(CFG['epochs']):
    tr_epoch_loss = 0
    test_epoch_loss = 0
    # TRAIN BATCHES
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
      batch_loss = train_step(models, batch)
      tr_epoch_loss += batch_loss
      print(f"(TRAIN) e [{e_i}/{CFG['epochs']}] b [{end_b}/{tr_adj.shape[0]}] loss {batch_loss}", end="\r")
    # TEST BATCHES
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
      batch_loss = train_step(models, batch, test=True)
      test_epoch_loss += batch_loss
      print(f"(TEST) e [{e_i}/{CFG['epochs']}] b [{end_b}/{test_adj.shape[0]}] loss {batch_loss}", end="\r")
    # EPOCH METRICS
    tr_epoch_loss = tr_epoch_loss / tr_num_batches
    test_epoch_loss = test_epoch_loss / test_num_batches
    print(f"EPOCH {e_i} TRAIN LOSS: {tr_epoch_loss} TEST LOSS: {test_epoch_loss}")
    with train_summary_writer.as_default():
      tf.summary.scalar('loss', tr_epoch_loss, step=e_i)
    with test_summary_writer.as_default():
      tf.summary.scalar('loss', test_epoch_loss, step=e_i)
