
CFG = {
  # dataset
  'max_nodes': 5,
  'language_spec': 'arithmetic',
  'min_num_values': 3,
  'max_num_values': 3,
  'num_samples': 10_000,
  # 6! = 720

  # models
  'hidden_size': 784,
  'attention_layers': 4,
  'num_heads': 6,

  # training
  'batch_size': 128,
  'epochs': 300,
  'mse_loss_only': True,
  'initial_lr': 0.001,
  'use_exponential_rate_scheduler': True
}
RESULTS = {}
