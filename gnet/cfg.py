# optionally read config from disk, maybe to restore a particular checkpoint

VISION = False
JUST_VISION = False
CFG = {
  # dataset
  'max_nodes': 5,
  # 5! = 120
  # 6! = 720
  'language_spec': 'arithmetic',
  'min_num_values': 2,
  'max_num_values': 3,
  'num_samples': 2_500 if JUST_VISION else 15_000,

  # graph models
  'graph_hidden_size': 512, # also used in discriminator module
  'encoder_attention_layers': 4,
  'decoder_attention_layers': 4,
  'num_heads': 6,
  # the inline if statement is a helpful pattern
  'initial_lr': 0.00001 if VISION else 0.0001,
  'use_exponential_rate_scheduler': True,

  # vision models
  'JUST_VISION': JUST_VISION,
  'VISION': VISION,
  'vision_model': 'conv', # could also be CPPN
  'DISC_MODEL' : 'ResNet50V2',
  'y_dim': 128,
  'x_dim': 128,
  'R': 7,
  'vision_hidden_size': 512,
  'minimum_filters': 0, # set to 0 for no minimum
  'Z_embed_num': 5,
  'cppn_loc_embed_dim': 32,
  'gen_Z_embed_dim': 512,
  'c_out': 1,
  'generator_lr': 0.0001,
  'discriminator_lr': 0.0001,
  'NUM_SYMBOLS': 256, # for toy problem
  'use_perceptual_loss': False,
  'use_distance_loss': False,
  'pretrained_disc': 'ResNet50V2',
  'use_custom_disc': True,

  # training
  'batch_size': 24,
  'epochs': 3_000,
  'mse_loss_only': False,
  'label_smoothing': 0.0,

  # checkpointing and saving (set to NOLOG to prevent logging)
  'run_name': 'new_run_graph',
  # 'load_graph_name': None,
  'load_graph_name': None,
  # 'load_vision_name': 'full_may27_4',
  'load_vision_name': None,
  'aws_filepath': '/gnet',
  'aws_bucket': 'gestalt-graph',
}
RESULTS = {}

if CFG['vision_model'] == 'conv':
  assert CFG['x_dim'] == CFG['y_dim']
