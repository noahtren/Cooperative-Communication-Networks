import code

import tensorflow as tf
from tqdm import tqdm

from graph_data import TensorGraph
from encoder import Encoder 
from decoder import GraphDecoder
from graph_match import minimum_loss_permutation
from cfg import CFG


# NOTE: encoder heads are not in parallel. there could probably be performance
# gains if I redid the multi-head attention layer to do multiple heads in parallel

# TODO: add a dimension to labels and predictions that tells if a particular
# value does not exist. this is not a problem for adjacency matrices but it is
# a problem for node features

def get_dataset(language_spec:str, min_num_values:int, max_num_values:int,
                max_nodes:int, num_samples:int, vis_first=False, **kwargs):
  instances = []
  print(f'Generating dataset')
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
def train_step(models, batch):
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
  adj, node_features, node_feature_specs, num_nodes, adj_labels, nf_labels = \
    get_dataset(**CFG)
  CFG['node_feature_specs'] = node_feature_specs
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

  # ==================== TRAIN LOOP ====================
  num_batches = (adj.shape[0] // CFG['batch_size']) + 1
  for e_i in range(CFG['epochs']):
    epoch_loss = 0
    for b_i in range(num_batches):
      end_b = min([(b_i + 1) * CFG['batch_size'], adj.shape[0]])
      start_b = end_b - CFG['batch_size']
      batch = {
        'adj': adj[start_b:end_b],
        'node_features': {name: tensor[start_b:end_b] for
          name, tensor in node_features.items()},
        'adj_labels': adj_labels[start_b:end_b],
        'nf_labels': {name: tensor[start_b:end_b] for
          name, tensor in nf_labels.items()},
        'num_nodes': num_nodes[start_b:end_b],
      }
      batch_loss = train_step(models, batch)
      epoch_loss += batch_loss
      print(f"e [{e_i}/{CFG['epochs']}] b [{end_b}/{adj.shape[0]}] loss {batch_loss}")
    epoch_loss = epoch_loss / num_batches
    print(f"EPOCH {e_i} LOSS {epoch_loss}")
