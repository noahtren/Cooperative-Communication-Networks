VISION = True
JUST_VISION = True
CFG = {
  # dataset
  'max_nodes': 5,
  # 5! = 120
  # 6! = 720
  'language_spec': 'arithmetic',
  'min_num_values': 2,
  'max_num_values': 3,
  'num_samples': 1_000 if JUST_VISION else 15_000,

  # graph models
  'hidden_size': 512,
  'encoder_attention_layers': 4,
  'decoder_attention_layers': 4,
  'num_heads': 6,
  # the inline if statement is a helpful pattern
  'initial_lr': 0.00005 if VISION else 0.0005,
  'use_exponential_rate_scheduler': True,

  # vision models
  'VISION': VISION,
  'y_dim': 128,
  'x_dim': 128,
  'G_hidden_size': 128,
  'G_num_layers': 6,
  'cppn_loc_embed_dim': 128,
  'cppn_Z_embed_dim': 128,
  'c_out': 3,
  'generator_lr': 0.0002,
  'discriminator_lr': 0.0002,

  # training
  'batch_size': 24,
  'epochs': 3_000,
  'mse_loss_only': False,
  'label_smoothing': 0.001 if VISION else 0.0,

  # checkpointing and saving (set to NOLOG to prevent logging)
  'run_name': 'vision_only_5',
  # 'run_name': 'graph_pretrain',
  'load_name': 'vision_only_4',
  # 'load_name': 'graph_pretrain',
  # 'load_name': None,
}
RESULTS = {}
