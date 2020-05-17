"""Representing trees as a subset of graph data structures, where all
information is provided as TensorFlow tensors during training time.
"""

import os
import random
from typing import Dict

import tensorflow as tf
import numpy as np
from graphviz import Digraph


class NodeType:
  def __init__(self, token:str=None, is_value:bool, ordered_children:bool=None, 
               min_children:int=None, max_children:int=None, parent_blacklist=[]):
    # input validation
    assert token ^ is_value, 'node type must be value XOR token'
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
    self.num_children = num_children
    self.parent_blacklist = parent_blacklist


LanguageSpecs = {
  'arithmetic': {
    'data_structure': 'tree'
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

  `nodes` and `order` are both matrices, `nodes` with shape [N, nf] and `order`
  with shape [N, of], where nf is the number of label node features and of is
  the number of possible child indices that a node could have.

  `num` is an integer that is less than or equal to N. It is the number of
  nodes that are actually in the graph.
  """
  def __init__(self, adj:tf.Tensor, node_features:Dict[str, tf.Tensor],
               num_nodes:int, language_spec:str):
    N = adj.shape[0]
    assert adj.shape[0] == adj.shape[1]
    for nf_name, node_feature in node_features.items():
      assert node_feature.shape[0] == N, f'{nf_name} first dimension should match N ({N}), but was {node_feature.shape[0]}'
    self.adj = adj
    self.node_features = node_features
    self.num_nodes = num_nodes
    self.language_spec = language_spec


  def visualize(self, name='unnamed', view=True):
    def node_feature_to_symbol(node_feature)
    dot = Digraph(name=name)

    # nodes
    for n_i in self.num:
      dot.node(str(n_i))

    # edges
    for n_i in self.num:
      dot.edge()

    os.makedirs('gviz', exist_ok=True)
    dot.render(os.path.join('gviz', f'{name}.png'), view=view)


  @classmethod
  def random_tree(self, language_spec:str, min_num_values:int,
                  max_num_values:int):
    """Generate a random, valid tree based on a language spec. Trees are
    generated as NumPy arrays and can be converted to TensorFlow tensors later.
    """
    SPEC = LanguageSpecs[language_spec]
    assert SPEC['data_structure'] == 'tree', \
      f'can not make tree batch with {language_spec} because it is not for trees.'

    adj_list = []
    values_list = []
    order_list = []

    NUM_VALUE_TOKENS = len(SPEC['value_tokens'])
    NUM_NODE_TYPES = len(SPEC['node_types']) + NUM_VALUE_TOKENS
    NUM_ORDER_INDICES = 1 + SPEC['max_children']
    NON_VALUE_NODE_TYPES = [nt for nt in SPEC['node_types'] if not nt.is_value]

    # make initial values
    num_values = random.randint(min_num_values, max_num_values)
    values = []
    for val_num in range(num_values):
      adj = []
      val = np.zeros((NUM_VALUE_TOKENS), dtype=np.int32)
      val[random.randint(NUM_VALUE_TOKENS)] = 1
      order = np.zeros((NUM_ORDER_INDICES), dtype=np.int32)
      adj_list.append(adj)
      values_list.append(val)
      order_list.append(order)
    
    # build tree from node stack until it has a single value
    tree_stack = list(range(num_values))

    while len(tree_stack) != 1:
      new_node_type = random.choice(NON_VALUE_NODE_TYPES)
      child_select_num = min([
        len(tree_stack),
        random.randint(new_node_type.min_children, new_node_type.max_children)
      ])
      children = random.choices(tree_stack, k=child_select_num)
      # update child information

      # create parent information

      # update stack

    
    tree_stack = values