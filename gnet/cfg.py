
CFG = {
  # dataset
  'max_nodes': 5,
  # 5! = 120
  # 6! = 720
  'language_spec': 'arithmetic',
  'min_num_values': 2,
  'max_num_values': 3,
  'num_samples': 15_000,

  # graph models
  'hidden_size': 512,
  'encoder_attention_layers': 4,
  'decoder_attention_layers': 4,
  'num_heads': 6,
  'initial_lr': 0.0005,
  'use_exponential_rate_scheduler': True,

  # vision models
  'VISION': False,
  'y_dim': 128,
  'x_dim': 128,
  'G_hidden_size': 128,
  'G_num_layers': 6,
  'cppn_loc_embed_dim': 128,
  'cppn_Z_embed_dim': 128,
  'c_out': 3,
  'generator_lr': 0.0001,
  'discriminator_lr': 0.0001,

  # training
  'batch_size': 64,
  'epochs': 3_000,
  'mse_loss_only': False,
  'label_smoothing': 0.00,

  # checkpointing and saving
  # set to NOLOG to prevent logging
  'run_name': 'graph_pretrain',
  # 'run_name': 'full_1',
  'load_name': None,
  # 'load_name': 'graph_pretrain',
}
RESULTS = {}
