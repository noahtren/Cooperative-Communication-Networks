import tensorflow as tf
from tqdm import tqdm

from graph_data import TensorGraph
from encoder import NodeFeatureEmbed, GlobalAttention
from decoder import GraphDecoder

def get_dataset(language_spec='arithmetic', min_num_values=3, max_num_values=3,
                max_nodes=5, num_samples=1_000, vis_first=False):
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
  adj, node_features, node_feature_specs = TensorGraph.instances_to_tensors(instances)
  return adj, node_features, node_feature_specs

if __name__ == "__main__":
  adj, node_features, node_feature_specs = get_dataset(vis_first=True)


