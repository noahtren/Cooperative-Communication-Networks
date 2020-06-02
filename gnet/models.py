import os

import tensorflow as tf

from cfg import CFG
from vision import Generator, Decoder
from graph_models import GraphEncoder, GraphDecoder
from aug import get_noisy_channel
from graph_data import get_dataset
from vision import make_symbol_data
from adamlrm import AdamLRM


def print_model_prefixes(model):
  """Find the model prefix for each model
  """
  for sub_model in model.layers:
    first_var = sub_model.variables[0]
    print(f"{sub_model}: {first_var.name}")


# ==================== GRAPH-ONLY MODEL ====================
class GraphModel(tf.keras.Model):
  def __init__(self, g_encoder, g_decoder):
    super(GraphModel, self).__init__()
    self._name = 'root_model'
    self.g_encoder = g_encoder
    self.g_decoder = g_decoder


  def call(self, batch, debug=False):
    Z = self.g_encoder(batch, debug)
    adj_out, nf_out = self.g_decoder(Z, debug)
    return adj_out, nf_out


# ==================== VISION-ONLY MODEL ====================
class VisionModel(tf.keras.Model):
  def __init__(self, generator, decoder, noisy_channel):
    super(VisionModel, self).__init__()
    self._name = 'root_model'
    self.generator = generator
    self.decoder = decoder
    self.noisy_channel = noisy_channel


  def call(self, Z, difficulty, debug=False):
    imgs = self.generator(Z, debug)
    aug_imgs = noisy_channel(imgs, difficulty)
    Z_pred = self.decoder(aug_imgs, debug)
    return Z_pred


# ==================== FULL GRAPH-GESTALT MODEL ====================
class FullModel(tf.keras.Model):
  def __init__(self, g_encoder, g_decoder, generator, decoder, noisy_channel):
    super(FullModel, self).__init__()
    self._name = 'root_model'
    self.g_encoder = g_encoder
    self.g_decoder = g_decoder
    self.generator = generator
    self.decoder = decoder
    self.noisy_channel = noisy_channel


  def call(self, batch, difficulty, debug=False):
    Z = self.g_encoder(batch, debug)
    imgs = self.generator(Z, debug)
    aug_imgs = self.noisy_channel(imgs, difficulty)
    Z_pred = self.decoder(imgs, debug)
    adj_out, nf_out = self.g_decoder(Z_pred, debug)
    return adj_out, nf_out, imgs, aug_imgs


def run_dummy_batch(model):
  """Used to populate weights before loading.
  """
  # TODO: test if this is necessary
  print(f"RUNNING DUMMY BATCH FOR MODEL: {model}")
  difficulty = tf.convert_to_tensor(0)
  if CFG['JUST_VISION']:
    Z, _ = make_symbol_data(CFG['batch_size'])
    Z_pred = model(Z, difficulty, debug=True)
  else:
    batch = get_dataset(**{**CFG, 'num_samples': CFG['batch_size']}, test=False)
    if CFG['VISION']:
      adj_pred, nf_pred, _, _ = model(batch, difficulty, debug=True)
    else:
      adj_pred, nf_pred = model(batch, debug=True)
    print("Input shapes:")
    print(f"adj_labels: {batch['adj_labels'].shape}")
    for key, tensor in batch['nf_labels'].items():
      print(f"nf_labels [{key}]: {tensor.shape}")
    print(f"num_nodes: {batch['num_nodes'].shape}")
    print("Output shapes:")
    print(f"adj_pred: {adj_pred.shape}")
    for key, tensor in nf_pred.items():
      print(f"nf_pred [{key}]: {tensor.shape}")


def get_model():
  # graph modules
  if not CFG['JUST_VISION']:
    ds = get_dataset(**{**CFG, 'num_samples': 1}, test=False)
    CFG['node_feature_specs'] = ds['node_feature_specs']
    g_encoder = GraphEncoder(**CFG)
    g_decoder = GraphDecoder(**CFG)
  # vision modules
  if CFG['VISION']:
    generator = Generator()
    decoder = Decoder()
    noisy_channel = get_noisy_channel()
  # create models
  if CFG['VISION']:
    if CFG['JUST_VISION']:
      return VisionModel(generator, decoder, noisy_channel)     
    else:
      return FullModel(g_encoder, g_decoder, generator, decoder, noisy_channel)
  else:
    return GraphModel(g_encoder, g_decoder)    


def load_weights(model):
  model_path = f"checkpoints/{CFG['run_name']}"
  if os.path.exists(model_path):
    print(f"Checkpoint exists at {model_path}, loaded these weights!")
    model.load_weights(model_path)
  else:
    print(f"No checkpoint found. Weights are initialized randomly.")


def save_weights(model):
  model_path = f"checkpoints/{CFG['run_name']}"
  model.save_weights(model_path)


def get_optim():
  # TODO: consider exponential decay code here
  # if CFG['use_exponential_rate_scheduler']:
  #   lr = tf.keras.optimizers.schedules.ExponentialDecay(
  #     CFG['initial_lr'], decay_steps=100_000, decay_rate=0.96, staircase=True)
  lr_multiplier = {
    'root_model/g_encoder': 0.0001,
    'root_model/g_decoder': 0.0001,
    'root_model/generator': 0.0001,
    'root_model/decoder':   0.0001,
  }
  optim = AdamLRM(lr=1., lr_multiplier=lr_multiplier)
  return optim


if __name__ == "__main__":
  model = get_model()
  optim = get_optim()
  run_dummy_batch(model)
  print_model_prefixes(model)
