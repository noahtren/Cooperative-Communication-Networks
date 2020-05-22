
CFG = {
  # dataset
  'max_nodes': 6,
  # 5! = 120
  # 6! = 720
  'language_spec': 'arithmetic',
  'min_num_values': 3,
  'max_num_values': 4,
  'num_samples': 2_000,

  # models
  'VISION': True,
  'hidden_size': 512,
  'encoder_attention_layers': 4,
  'decoder_attention_layers': 4,
  'num_heads': 6,
  'y_dim': 96,
  'x_dim': 96,
  'G_hidden_size': 256,
  'G_num_layers': 3,
  'cppn_loc_embed_dim': 128,
  'c_out': 3,

  # training
  'batch_size': 16,
  'epochs': 3_000,
  'mse_loss_only': False,
  'initial_lr': 0.0005,
  'use_exponential_rate_scheduler': True,

  # checkpointing and saving
  # set to NOLOG to prevent logging
  'run_name': 'NOLOG'
}
RESULTS = {}
