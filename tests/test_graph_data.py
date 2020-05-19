from gnet.graph_data import TensorGraph


def test_arithmetic():
  adj_list, values_list, value_tokens_list, order_list = \
    TensorGraph.random_tree('arithmetic', 3, 3)
  tg = TensorGraph.tree_to_instance(adj_list, values_list, value_tokens_list,
    order_list, 10, 'arithmetic')
