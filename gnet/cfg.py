"""Read config from file when running session. This can be set up for training
multiple different models with different hyperparameter settings.
"""

import os
import json

code_path = os.path.dirname(os.path.abspath(__file__))

access_path = os.path.join(code_path, "gestalt-graph-59b01bb414f3.json")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = access_path


def validate_cfg(CFG):
  if CFG['vision_model'] == 'conv':
    assert CFG['x_dim'] == CFG['y_dim']


def populate_cfg(CFG):
  CFG['TPU'] = False
  if 'IS_TPU' in os.environ:
    if os.environ['IS_TPU'] == 'y':
      CFG['TPU'] = True
  return CFG


def parse_cfg(CFG):
  CFG['save_checkpoint_every'] = CFG['cloud_save_checkpoint_every']
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


CFG = json.load(open(os.path.join(code_path, 'config.json'), 'r'))
CFG = populate_cfg(CFG)
CFG = parse_cfg(CFG)
validate_cfg(CFG)
print(f"Config: {json.dumps(CFG, indent=4)}")
