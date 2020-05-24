"""Optimizer utilities. Found on Phil Culliton's Kaggle page.
https://www.kaggle.com/philculliton/bert-optimization
"""

import code

import tensorflow as tf


def shuffle_together(*tensors):
  idxs = tf.range(tensors[0].shape[0])
  out = []
  for tensor in tensors:
    out.append(tf.gather(tensor, idxs))
  return tuple(out)


def update_data_dict(data_dict, batch_dict):
  for name in batch_dict.keys():
    if name not in data_dict:
      data_dict[name] = batch_dict[name]
    else:
      data_dict[name] += batch_dict[name]
  return data_dict


def normalize_data_dict(data_dict, num_batches):
  for name, value in data_dict.items():
    data_dict[name] = (value / num_batches).numpy().item()
  return data_dict


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applys a warmup schedule on a given learning rate decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_schedule_fn,
      warmup_steps,
      power=1.0,
      name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(global_step_float < warmup_steps_float,
                     lambda: warmup_learning_rate,
                     lambda: self.decay_schedule_fn(step),
                     name=name)

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'power': self.power,
        'name': self.name
    }


def create_optimizer(init_lr, num_train_steps, num_warmup_steps):
  """Creates an optimizer with learning rate schedule."""
  # Implements linear decay of the learning rate.
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=init_lr,
      decay_steps=num_train_steps,
      end_learning_rate=0.0)
  if num_warmup_steps:
    learning_rate_fn = WarmUp(initial_learning_rate=init_lr,
                              decay_schedule_fn=learning_rate_fn,
                              warmup_steps=num_warmup_steps)
  optimizer = AdamWeightDecay(
      learning_rate=learning_rate_fn,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=['layer_norm', 'bias'])
  return optimizer


dense_regularization = {
  'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  'bias_regularizer': tf.keras.regularizers.l2(1e-4),
  'activity_regularizer': tf.keras.regularizers.l2(1e-6)
}