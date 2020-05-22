
CFG = {
  # dataset
  'max_nodes': 6,
  # 5! = 120
  # 6! = 720
  'language_spec': 'arithmetic',
  'min_num_values': 3,
  'max_num_values': 4,
  'num_samples': 1000,

  # models
  'hidden_size': 784,
  'encoder_attention_layers': 6,
  'decoder_attention_layers': 2,
  'num_heads': 6,

  # training
  'batch_size': 128,
  'epochs': 3_000,
  'mse_loss_only': False,
  'initial_lr': 0.0002,
  'use_exponential_rate_scheduler': True,

  # checkpointing and saving
  # set to NOLOG to prevent logging
  'run_name': 'NOLOG'
}
RESULTS = {}
