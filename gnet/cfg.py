
CFG = {
  # dataset
  'max_nodes': 3,
  'language_spec': 'arithmetic',
  'min_num_values': 2,
  'max_num_values': 2,
  'num_samples': 1_000,
  # 6! = 720

  # models
  'hidden_size': 512,
  'attention_layers': 2,
  'num_heads': 4,

  # training
  'batch_size': 32,
  'epochs': 50,
  'mse_loss_only': True
}
RESULTS = {}
