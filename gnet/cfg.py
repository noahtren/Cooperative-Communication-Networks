"""Read config from file when running session. This can be set up for training
multiple different models with different hyperparameter settings.
"""

import os
import json

code_path = os.path.dirname(os.path.abspath(__file__))


def validate_cfg(CFG):
  if CFG['vision_model'] == 'conv':
    assert CFG['x_dim'] == CFG['y_dim']


def parse_cfg(CFG):
  if CFG['USE_S3']:
    CFG['save_checkpoint_every'] = CFG['s3_save_checkpoint_every']
    CFG['root_filepath'] = CFG['s3_root_filepath']
  else:
    CFG['save_checkpoint_every'] = CFG['default_save_checkpoint_every']
    CFG['root_filepath'] = CFG['default_root_filepath']
  return CFG


CFG = json.load(open(os.path.join(code_path, 'config.json'), 'r'))
CFG = parse_cfg(CFG)
print(f"Config: {json.dumps(CFG, indent=4)}")
