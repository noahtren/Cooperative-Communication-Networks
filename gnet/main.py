import code

import tensorflow as tf
from tqdm import tqdm

from graph_data import TensorGraph
from encoder import Encoder 
from decoder import GraphDecoder
from graph_match import minimum_loss_permutation
from cfg import CFG

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
  return adj, node_features, node_feature_specs, num_nodes


def train_step(models, batch):
  with tf.GradientTape(persistent=True) as tape:
    x = models['encoder'][0](batch)
    adj_pred, nf_pred = models['decoder'][0]({'x': x})
    loss = minimum_loss_permutation(
      batch['adj'],
      batch['node_features'],
      adj_pred,
      nf_pred
    )
  # TODO: optimizer steps using gradient tape

if __name__ == "__main__":
  # ==================== DATA AND MODELS ====================
  adj, node_features, node_feature_specs, num_nodes = get_dataset(**CFG)
  CFG['node_feature_specs'] = node_feature_specs
  encoder = Encoder(**CFG)
  decoder = GraphDecoder(**CFG)
  models = {
    'encoder': [encoder, tf.keras.optimizers.Adam()],
    'decoder': [decoder, tf.keras.optimizers.Adam()],
  }

  # ==================== TRAIN LOOP ====================
  num_batches = (adj.shape[0] // CFG['batch_size']) + 1
  for e_i in range(CFG['epochs']):
    for b_i in range(num_batches):
      start_b = b_i * CFG['batch_size']
      end_b = min([(b_i + 1) * CFG['batch_size'], adj.shape[0]])
      batch = {
        'adj': adj[start_b:end_b],
        'node_features': {name: tensor[start_b:end_b] for
          name, tensor in node_features.items()},
        'num_nodes': num_nodes[start_b:end_b]
      }
      train_step(models, batch)
      print(f"e [{e_i}/{CFG['epochs']}] b [{end_b}/{adj.shape[0]}] loss {b_loss}")
