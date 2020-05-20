
CFG = {
  # dataset
  'max_nodes': 5,
  'language_spec': 'arithmetic',
  'min_num_values': 2,
  'max_num_values': 3,
  'num_samples': 10_000,
  # 6! = 720

  # models
  'hidden_size': 784,
  'attention_layers': 3,
  'num_heads': 6,

  # training
  'batch_size': 128,
  'epochs': 300,
  'mse_loss_only': True,
  'lr': 0.0005
}
RESULTS = {}
