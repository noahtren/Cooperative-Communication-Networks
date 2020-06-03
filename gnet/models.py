import os
import code

import tensorflow as tf

from cfg import CFG
from vision import Generator, Decoder
from graph_models import GraphEncoder, GraphDecoder
from aug import get_noisy_channel
from graph_data import get_dataset
from vision import make_symbol_data
from adamlrm import AdamLRM
from upload import gs_folder_exists


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


  def call(self, symbols, difficulty, debug=False):
    imgs = self.generator(symbols, debug)
    aug_imgs = self.noisy_channel(imgs, difficulty)
    predictions = self.decoder(aug_imgs, debug)
    return predictions, imgs, aug_imgs


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
    symbol_batch, _ = make_symbol_data(**{**CFG, 'num_samples': CFG['batch_size']})
    symbol_pred = model(symbol_batch, difficulty, debug=True)
  else:
    batch, _ = get_dataset(**{**CFG, 'num_samples': CFG['batch_size']}, test=False)
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
    ds, node_feature_specs = get_dataset(**{**CFG, 'num_samples': 1}, test=False)
    CFG['node_feature_specs'] = node_feature_specs
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
      print("Using vision model")
      return VisionModel(generator, decoder, noisy_channel)     
    else:
      print("Using full model")
      return FullModel(g_encoder, g_decoder, generator, decoder, noisy_channel)
  else:
    print("Using graph model")
    return GraphModel(g_encoder, g_decoder)    


def load_weights(model, path_prefix):
  model_path = f"{path_prefix}checkpoints/{CFG['load_name']}"
  path_exists = False
  print(f"Attempting to load weights from {model_path}...")
  try:
    model.load_weights(os.path.join(model_path, 'best'))
  except Exception as e:
    print(e)
    print("No weights found")


def save_weights(model, path_prefix):
  model_path = f"{path_prefix}checkpoints/{CFG['run_name']}/best"
  model.save_weights(model_path)


def get_optim():
  lr_multiplier = {
    'root_model/g_encoder': CFG['g_encoder_lr'],
    'root_model/g_decoder': CFG['g_decoder_lr'],
    'root_model/generator': CFG['generator_lr'],
    'root_model/decoder':   CFG['decoder_lr'],
  }
  optim = AdamLRM(lr=1., lr_multiplier=lr_multiplier)
  return optim


if __name__ == "__main__":
  model = get_model()
  optim = get_optim()
  run_dummy_batch(model)
  print_model_prefixes(model)
