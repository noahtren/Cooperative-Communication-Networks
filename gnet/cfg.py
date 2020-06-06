"""Read config from file when running session. Config manages all
hyperparameters of each training session.
"""

import code
import os
import json

# add Google Cloud credentials to environment variables
code_path = os.path.dirname(os.path.abspath(__file__))
access_path = os.path.join(code_path, "gestalt-graph-59b01bb414f3.json")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = access_path


def validate_cfg(CFG):
  if CFG['vision_model'] == 'conv':
    assert CFG['x_dim'] == CFG['y_dim']
  if 'composite_colors' in CFG:
    assert CFG['c_out'] == len(CFG['composite_colors'])


def populate_cfg(CFG):
  CFG['TPU'] = False
  if 'IS_TPU' in os.environ:
    if os.environ['IS_TPU'] == 'y':
      CFG['TPU'] = True
  if 'use_spy' not in CFG:
    CFG['use_spy'] = False
  return CFG


def parse_cfg(CFG):
  CFG['save_checkpoint_every'] = CFG['cloud_save_checkpoint_every']
  if 'cloud_image_every' in CFG:
    CFG['image_every'] = CFG['cloud_image_every']
  CFG['root_filepath'] = CFG['gs_root_filepath']
  CFG['VISION'] = CFG['JUST_VISION'] or CFG['FULL']
  # if CFG['USE_S3']:
  #   CFG['save_checkpoint_every'] = CFG['cloud_save_checkpoint_every']
  #   CFG['root_filepath'] = CFG['s3_root_filepath']
  # else:
  #   CFG['save_checkpoint_every'] = CFG['default_save_checkpoint_every']
  #   CFG['root_filepath'] = CFG['default_root_filepath']
  return CFG


def refine_cfg(CFG):
  """Parsing and populating all values of config stored in JSON
  """
  CFG = populate_cfg(CFG)
  CFG = parse_cfg(CFG)
  validate_cfg(CFG)
  return CFG


def read_config():
  CFG = json.load(open(os.path.join(code_path, 'config.json'), 'r'))
  CFG = refine_cfg(CFG)
  print(f"Config: {json.dumps(CFG, indent=4)}")
  return CFG


def read_config_from_string(cfg_str):
  CFG = json.loads(cfg_str)
  CFG = refine_cfg(CFG)
  return CFG


def set_config(cfg):
  global CFG
  CFG = cfg


def get_config():
  global CFG
  try:
    CFG
    return CFG
  except NameError:
    cfg = read_config()
    set_config(cfg)
    return cfg
