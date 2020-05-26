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
  'vision_model': 'conv',
  'DISC_MODEL' : 'ResNet50V2',
  'y_dim': 128,
  'x_dim': 128,
  'R': 7,
  'vision_hidden_size': 512,
  'cppn_loc_embed_dim': 32,
  'gen_Z_embed_dim': 512,
  'c_out': 3,
  'generator_lr': 0.0001,
  'discriminator_lr': 0.0001,
  'NUM_SYMBOLS': 256, # for toy problem
  'use_perceptual_loss': False,
  'use_distance_loss': False,
  'pretrained_disc': 'ResNet50V2',
  'use_custom_disc': True,

  # training
  'batch_size': 32,
  'epochs': 3_000,
  'mse_loss_only': False,
  'label_smoothing': 0.0 if VISION else 0.0,

  # checkpointing and saving (set to NOLOG to prevent logging)
  'run_name': 'slow_normal_experiment_256symbols_contd2',
  # 'run_name': 'graph_pretrain',
  # 'load_name': 'graph_pretrain',
  'load_name': 'slow_normal_experiment_256symbols_contd',
}
RESULTS = {}

if CFG['vision_model']:
  assert CFG['x_dim'] == CFG['y_dim']
  