
CFG = {
  # dataset
  'max_nodes': 4,
  'language_spec': 'arithmetic',
  'min_num_values': 3,
  'max_num_values': 3,
  'num_samples': 1_000,
  # 6! = 720

  # models
  'hidden_size': 512,
  'attention_layers': 3,
  'num_heads': 4,

  # training
  'batch_size': 16,
  'epochs': 10
}
RESULTS = {}
