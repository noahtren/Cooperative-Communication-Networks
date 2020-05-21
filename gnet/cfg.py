
CFG = {
  # dataset
  'max_nodes': 6,
  # 5! = 120
  # 6! = 720
  'language_spec': 'arithmetic',
  'min_num_values': 3,
  'max_num_values': 4,
  'num_samples': 100_000,

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
