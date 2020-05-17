from gnet.graph_data import TensorGraph

def test_arithmetic():
  for _ in range(100):
    TensorGraph.random_tree('arithmetic', 3, 5)
