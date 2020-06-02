"""Representing trees as a subset of graph data structures, where all
information is provided as TensorFlow tensors during training time.
"""

import os
import random
import code
from typing import Dict

import tensorflow as tf
import numpy as np
from graphviz import Digraph
from tqdm import tqdm


class NodeType:
  def __init__(self, token:str=None, is_value:bool=None, ordered_children:bool=None, 
               min_children:int=None, max_children:int=None, parent_blacklist=[]):
    # input validation
    assert (token is None) ^ (is_value is None), 'node type must be value XOR token'
    if token:
      assert ordered_children is not None, 'order_children must be specified for token node type'
      assert max_children is not None, 'max_children must be specified for token node type'
      assert min_children is not None, 'min_children must be specified for token node type'
    else:
      assert ordered_children is None, 'order_children should not be specified for value node type'
      assert max_children is None, 'max_children should not be specified for token node type'
      assert min_children is None, 'min_children should not be specified for token node type'
    # assignment
    self.token = token
    self.is_value = is_value
    self.ordered_children = ordered_children
    self.min_children = min_children
    self.max_children = max_children
    self.parent_blacklist = parent_blacklist


LanguageSpecs = {
  'arithmetic': {
    'data_structure': 'tree',
    'node_types': [
      NodeType(is_value=True),
      NodeType(token='+', ordered_children=False, min_children=2, max_children=3),
      NodeType(token='-', ordered_children=True, min_children=2, max_children=2),
      NodeType(token='*', ordered_children=False, min_children=2, max_children=3),
      NodeType(token='/', ordered_children=True, min_children=2, max_children=2),
      NodeType(token='abs', ordered_children=False, min_children=1, max_children=1, parent_blacklist=['abs']),
    ],
    'value_tokens': ['a', 'b', 'c', 'd'],
    'max_children': 3
  }
}


class TensorGraph:
  """Representation of a graph data structure as TensorFlow tensors. All graphs
  are defined by an adjacency matrix (adj) and node-wise features (nodes and
  order.)

  This implementation requires that edges are directed but not weighted.
  
  `adj` is a matrix of shape [N, N] where the i, j index is 1 if i has a
  directed edge to j, otherwise it is 0.

  `node_features` is a dictionary of named node features, each of shape [n, ?]
  where n is the maximum number of nodes and ? is the encoded feature space of
  that feature.

  `num_nodes` is an integer that is less than or equal to N. It is the number of
  nodes that are actually in the graph.
  """
  def __init__(self, adj:tf.Tensor, node_features:Dict[str, tf.Tensor],
               num_nodes:int, attrs:dict, language_spec:str):
    N = adj.shape[0]
    assert adj.shape[0] == adj.shape[1]
    for nf_name, node_feature in node_features.items():
      assert node_feature.shape[0] == N, f'{nf_name} first dimension should match N ({N}), but was {node_feature.shape[0]}'
    self.adj = adj
    self.node_features = node_features
    self.num_nodes = num_nodes
    self.attrs = attrs
    self.language_spec = language_spec


  def visualize(self, name='unnamed', view=True):
    dot = Digraph(name=name)

    # nodes
    node_names = []
    for n_i in range(self.num_nodes):
      ord_ = np.argmax(self.node_features['order'][n_i]) - 1
      node_name = f"{self.attrs['value_tokens_list'][n_i]} ord={ord_}, i={n_i}"
      dot.node(node_name)
      node_names.append(node_name)

    # edges
    for n_i in range(self.num_nodes):
      src_name = node_names[n_i]
      for a_i, adj_bool in enumerate(self.adj[n_i]):
        if adj_bool:
          targ_name = node_names[a_i]
          dot.edge(src_name, targ_name)

    os.makedirs('gviz', exist_ok=True)
    dot.render(os.path.join('gviz', f'{name}'), format='pdf', view=view)


  @classmethod
  def tree_to_instance(cls, adj_list:list, values_list:list, value_tokens_list:list,
                       order_list:list, max_nodes:int, language_spec:str):
    """Convert Python data types to a TensorGraph instance with max_nodes, using
    padding where necessary. If the Python data has too many nodes (more than
    `max_nodes`), then this function returns none.
    """
    if len(adj_list) > max_nodes:
      return None

    pad_amt = max_nodes - len(adj_list)
    # values and order lists are already in correct numpy format. they just need
    # to be padded. padding only needs to occur along the first axis
    values = np.stack(values_list)
    order = np.stack(order_list)
    values = np.pad(values, [[0, pad_amt], [0, 0]])
    order = np.pad(order, [[0, pad_amt], [0, 0]])

    adj_arrs = []
    for adj_indices in adj_list:
      adj_arr = np.zeros((len(adj_list)), dtype=np.int32)
      for n_i in adj_indices:
        adj_arr[n_i] = 1
      adj_arrs.append(adj_arr)
    
    adj = np.stack(adj_arrs)
    adj = np.pad(adj, [[0, pad_amt], [0, pad_amt]])

    return TensorGraph(adj,
      node_features={
        'values': values,
        'order': order
      },
      num_nodes=len(adj_list),
      attrs={
        'value_tokens_list': value_tokens_list
      },
      language_spec=language_spec
    )


  @classmethod
  def random_tree(cls, language_spec:str, min_num_values:int,
                  max_num_values:int, **kwargs):
    """Generate a random, valid tree based on a language spec. Trees are
    generated as NumPy arrays and can be converted to TensorFlow tensors later.
    """
    SPEC = LanguageSpecs[language_spec]
    assert SPEC['data_structure'] == 'tree', \
      f'can not make tree batch with {language_spec} because it is not for trees.'

    adj_list = []
    values_list = []
    value_tokens_list = []
    order_list = []

    NUM_VALUE_TOKENS = len(SPEC['value_tokens'])
    NON_VALUE_NODE_TYPES = [nt for nt in SPEC['node_types'] if not nt.is_value]
    NUM_NODE_VALUES = len(NON_VALUE_NODE_TYPES) + NUM_VALUE_TOKENS
    NUM_ORDER_INDICES = 1 + SPEC['max_children']

    # make initial values
    num_values = random.randint(min_num_values, max_num_values)
    values = []
    for val_num in range(num_values):
      adj = []
      val = np.zeros((NUM_NODE_VALUES), dtype=np.int32)
      val_idx = random.randint(0, NUM_VALUE_TOKENS - 1)
      val[val_idx] = 1
      val_token = SPEC['value_tokens'][val_idx]
      order = np.zeros((NUM_ORDER_INDICES), dtype=np.int32)
      adj_list.append(adj)
      values_list.append(val)
      value_tokens_list.append(val_token)
      order_list.append(order)
    
    # build tree from node stack until it has a single value
    tree_stack = list(range(num_values))

    next_idx = len(tree_stack)
    while len(tree_stack) != 1:
      node_type_i = random.randint(0, len(NON_VALUE_NODE_TYPES) - 1)
      new_node_type = NON_VALUE_NODE_TYPES[node_type_i]
      child_select_num = min([
        len(tree_stack),
        random.randint(new_node_type.min_children, new_node_type.max_children)
      ])
      valid_children = []
      for child_i in tree_stack:
        child_token = value_tokens_list[child_i]
        if child_token not in new_node_type.parent_blacklist:
          valid_children.append(child_i)
        else:
          pass
          # print(f'IGNORE INDEX {child_i}')

      children = random.sample(tree_stack, k=child_select_num)
      
      # update child information
      for o_i, child_i in enumerate(children):
        if new_node_type.ordered_children:
          order_list[child_i][o_i + 1] = 1
        else:
          order_list[child_i][0] = 1

      # create parent information
      p_adj = children
      p_val = np.zeros((NUM_NODE_VALUES), dtype=np.int32)
      p_val[NUM_VALUE_TOKENS + node_type_i] = 1
      p_val_token = new_node_type.token
      p_order = np.zeros((NUM_ORDER_INDICES), dtype=np.int32)
      adj_list.append(p_adj)
      values_list.append(p_val)
      value_tokens_list.append(p_val_token)
      order_list.append(p_order)

      # update stack
      for child_i in children:
        tree_stack.remove(child_i)
      tree_stack.append(next_idx)
      next_idx += 1
    
    # set root node order to unordered
    order_list[-1][0] = 1
    return adj_list, values_list, value_tokens_list, order_list


  @classmethod
  def parse_tree(cls, language_spec, adj_list, value_tokens_list, order_list, max_nodes):
    """Given an adjacency matrix, token names, and a list of token orders,
    validate that it is a tree and then produce an instance of the tree as
    a TensorGraph.
    """
    SPEC = LanguageSpecs[language_spec]
    NON_VALUE_NODE_TYPES = [nt for nt in SPEC['node_types'] if not nt.is_value]
    NON_VALUE_NODE_TOKENS = [nt.token for nt in NON_VALUE_NODE_TYPES]

    values = []
    for token in value_tokens_list:
      if token in NON_VALUE_NODE_TOKENS:
        token_i = NON_VALUE_NODE_TOKENS.index(token) + len(NON_VALUE_NODE_TOKENS)
      elif token in SPEC['value_tokens']:
        token_i = SPEC['value_tokens'].index(token)
      else:
        raise RuntimeError(f"token {token} doesn't exist in spec '{language_spec}'")
      value = np.zeros((len(NON_VALUE_NODE_TOKENS) + len(SPEC['value_tokens'])),
        dtype=np.int32)
      value[token_i] = 1
      values.append(value)
    
    orders = []
    for order_idx in order_list:
      order = np.zeros((1 + SPEC['max_children']), dtype=np.int32)
      order[order_idx + 1] = 1
      orders.append(order)
    
    return adj_list, values, value_tokens_list, orders


  @classmethod
  def instances_to_tensors(cls, tensor_graphs:list):
    """Given a list of instances of this class, turn it into a training dataset
    stored in memory.

    Returns:
      batched adj tensor
      batched dictionary of node feature tensors
      node_feature_specs dictionary, mapping node feature names to dimensionality
    """
    adj = []
    node_features = {}
    node_feature_specs = {}
    num_nodes = []
    for tg in tensor_graphs:
      adj.append(tg.adj)
      num_nodes.append(tg.num_nodes)
      for name, tensor in tg.node_features.items():
        if name not in node_feature_specs:
          node_features[name] = [tensor]
          node_feature_specs[name] = tensor.shape[-1]
        else:
          node_features[name].append(tensor)
    # stack along batch axis
    adj = tf.stack(adj, axis=0)
    node_features = {name: tf.cast(tf.stack(nf_list, axis=0), tf.float32) for name, nf_list in
      node_features.items()}
    num_nodes = tf.stack(num_nodes, axis=0)
    return adj, node_features, node_feature_specs, num_nodes


def label_data(node_features, adj, num_nodes):
  # postprocessing for data labels
  nf_labels = {}
  for name in node_features.keys():
    empty_rows = (tf.math.reduce_max(node_features[name], axis=-1) == 0)[:, :, tf.newaxis]
    empty_rows = tf.cast(empty_rows, tf.float32)
    nf_labels[name] = tf.concat([node_features[name], empty_rows], axis=-1)

  identity = tf.eye(adj.shape[1], adj.shape[1],
    batch_shape=[adj.shape[0]], dtype=tf.int32)
  # only apply identity matrix to nodes that exist
  grid_max_idxs = tf.tile(tf.expand_dims(tf.math.reduce_max(
      tf.cast(tf.reshape(tf.where(
        tf.ones((adj.shape[1], adj.shape[1]))
      ), (adj.shape[1], adj.shape[1], 2)), tf.int32), axis=-1), 0),
      [num_nodes.shape[0], 1, 1])
  identity = tf.where(grid_max_idxs < num_nodes[:, tf.newaxis, tf.newaxis],
    identity, tf.zeros_like(identity))
  # NOTE: I ran 2 experiments to prove that this provides a significant
  # performance boost
  adj_labels = adj + identity
  return nf_labels, adj_labels


def get_dataset(language_spec:str, min_num_values:int, max_num_values:int,
                max_nodes:int, num_samples:int, vis_first=False, test=False,
                dummy=False, **kwargs):
  instances = []
  gen_name = 'train'
  if test:
    gen_name = 'test'
  if dummy:
    gen_name = 'dummy'
  print(f"Generating {gen_name} dataset")
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
  nf_labels, adj_labels = label_data(node_features, adj, num_nodes)
  ds = {
    'adj': adj,
    'node_features': node_features,
    'node_feature_specs': node_feature_specs,
    'num_nodes': num_nodes,
    'adj_labels': adj_labels,
    'nf_labels': nf_labels
  }
  return ds


if __name__ == "__main__":
  for _ in range(10000):
    adj_list, values_list, value_tokens_list, order_list = TensorGraph.random_tree('arithmetic', 3, 5)
    tg = TensorGraph.tree_to_instance(adj_list, values_list, value_tokens_list, order_list, 6, 'arithmetic')
    if tg is not None:
      tg.visualize()
      code.interact(local={**locals(), **globals()})
