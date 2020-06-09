import code
import datetime
import json
import os
import io
import contextlib

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from ccn.cfg import get_config; CFG = get_config()
from ccn.graph_match import minimum_loss_permutation
from ccn.vision import Perceptor, perceptual_loss, make_symbol_data, color_composite
from ccn.graph_data import get_dataset
from ccn.ml_utils import dense_regularization, update_data_dict, normalize_data_dict
from ccn.models import get_model, get_optim, get_spy_optim, run_dummy_batch, load_weights, save_weights
from ccn.upload import gs_upload_blob_from_memory, gs_upload_blob_from_string

strategy = None

if CFG['TPU']:
  TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)
else:
  strategy = tf.distribute.get_strategy()

num_replicas = strategy.num_replicas_in_sync
print("NUMBER OF REPLICAS: ", num_replicas)


def update_difficulty(difficulty, epoch_acc):
  target_metric = 'symbols' if CFG['JUST_VISION'] else 'values'
  if epoch_acc[target_metric] >= 0.995 and difficulty < 15:
    difficulty += 1
  if epoch_acc[target_metric] < 0.1 and difficulty > 0:
    difficulty -= 1
  return difficulty


def get_batch(start_b, end_b, adj, node_features, adj_labels, nf_labels,
  num_nodes, **kwargs):
  batch = {
    'adj': adj[start_b:end_b],
    'node_features': {name: tensor[start_b:end_b] for
      name, tensor in node_features.items()},
    'adj_labels': adj_labels[start_b:end_b],
    'nf_labels': {name: tensor[start_b:end_b] for
      name, tensor in nf_labels.items()},
    'num_nodes': num_nodes[start_b:end_b],
  }
  return batch


def get_visual_samples(test_ds, model, test_num_samples, difficulty):
  sample_idxs = tf.random.uniform([CFG['batch_size']], 0, test_num_samples, tf.int32)
  spy_imgs = None
  if CFG['JUST_VISION']:
    vis_batch = tf.gather(test_ds, sample_idxs)
    _, sample_imgs, aug_imgs = model(vis_batch, difficulty)
    if CFG['use_spy']:
      _, spy_imgs, _ = model(vis_batch, difficulty, spy_turn=True)
  else:
    vis_batch = {
      'adj': tf.gather(test_ds['adj'], sample_idxs),
      'node_features': {name: tf.gather(tensor, sample_idxs) for
        name, tensor in test_ds['node_features'].items()},
      'adj_labels': tf.gather(test_ds['adj_labels'], sample_idxs),
      'nf_labels': {name: tf.gather(tensor, sample_idxs) for
        name, tensor in test_ds['nf_labels'].items()},
      'num_nodes': tf.gather(test_ds['num_nodes'], sample_idxs),
    }
    _, _, sample_imgs, aug_imgs, _, _ = model(vis_batch, difficulty)
    if CFG['use_spy']:
      _, _, spy_imgs, _, _, _ = model(vis_batch, difficulty, spy_turn=True)
  # scale tanh to visual range
  sample_imgs = (sample_imgs + 1) / 2
  aug_imgs = (aug_imgs + 1) / 2
  if CFG['use_spy']:
    spy_imgs = (spy_imgs + 1) / 2
  return sample_imgs, aug_imgs, spy_imgs


def main():
  print(f"Using strategy: {strategy}")
  with strategy.scope():
    def vision_only_predict(model, batch, difficulty, perceptor):
      all_losses = {}
      symbols = batch
      predictions, imgs, aug_imgs = model(symbols, difficulty)
      acc = tf.keras.metrics.categorical_accuracy(symbols, predictions)
      acc = tf.math.reduce_mean(acc)
      acc = {'symbols': acc}
      lfn = tf.keras.losses.mean_squared_error if CFG['use_mse_loss'] else \
        lambda true, pred: tf.keras.losses.categorical_crossentropy(true, pred, label_smoothing=CFG['label_smoothing'])
      recon_loss = tf.keras.losses.categorical_crossentropy(
        symbols,
        predictions,
      )
      recon_loss = tf.math.reduce_sum(recon_loss)
      all_losses['reconstruction'] = recon_loss
      if CFG['use_perceptual_loss']:
        features = perceptor(imgs)
        percept_loss = perceptual_loss(features)
        all_losses['perceptual'] = percept_loss
      if CFG['use_spy']:
        # Wassertein-ish loss function
        predictions_s, imgs_s, aug_imgs_s = model(symbols, difficulty, spy_turn=True)
        acc_s = tf.keras.metrics.categorical_accuracy(symbols, predictions_s)
        acc_s = tf.math.reduce_mean(acc_s)
        acc['spy_symbols'] = acc_s
        spy_loss = lfn(
          symbols,
          predictions_s,
        )
        spy_loss = tf.math.reduce_sum(spy_loss)

      # get reg loss
      for sub_model in model.layers:
        name = sub_model.name
        all_losses[name] = tf.math.reduce_sum(sub_model.losses)

      # get loss sum
      loss_sum = 0.
      for loss_name in all_losses.keys():
        loss_sum += all_losses[loss_name]
      if CFG['use_spy']:
        all_losses['spy_reconstruction'] = spy_loss
        # make logistic loss linear
        all_losses['spy_scaled'] = tf.math.exp(spy_loss * -1)
        loss_sum += all_losses['spy_scaled']
      return loss_sum, acc, all_losses

    

    def graph_or_full_predict(model, batch, difficulty, perceptor):
      all_losses = {}
      if CFG['VISION']:
        adj_pred, nf_pred, imgs, aug_imgs, _, _ = model(batch, difficulty)
        spy_adj_pred, spy_nf_pred, _, _, _, _ = model(batch, difficulty, spy_turn=True)
        if CFG['use_perceptual_loss']:
          features = perceptor(imgs)
          percept_loss = perceptual_loss(features)
          all_losses['perceptual'] = percept_loss
      else:
        adj_pred, nf_pred = model(batch, difficulty)
        spy_adj_pred, spy_nf_pred = model(batch, difficulty, spy_turn=True)
      recon_loss, acc = minimum_loss_permutation(
        batch['adj_labels'],
        batch['nf_labels'],
        adj_pred,
        nf_pred
      )
      all_losses['reconstruction'] = recon_loss

      spy_loss, spy_acc = minimum_loss_permutation(
        batch['adj_labels'],
        batch['nf_labels'],
        spy_adj_pred,
        spy_nf_pred
      )
      for key in spy_acc.keys():
        acc[f"spy_{key}"] = spy_acc[key]

      # get reg loss
      for sub_model in model.layers:
        name = sub_model.name
        all_losses[name] = tf.math.reduce_sum(sub_model.losses)

      # get loss sum
      loss_sum = 0.
      for loss_name in all_losses.keys():
        loss_sum += all_losses[loss_name]

      if CFG['use_spy']:
        all_losses['spy_reconstruction'] = spy_loss
        # make logistic loss linear
        all_losses['spy_scaled'] = tf.math.exp(spy_loss * -1)
        loss_sum += all_losses['spy_scaled']
      return loss_sum, acc, all_losses


    def predict(model, batch, difficulty, perceptor):
      if CFG['JUST_VISION']:
        loss_sum, acc, all_losses = vision_only_predict(model, batch, difficulty, perceptor)
      else:
        loss_sum, acc, all_losses = graph_or_full_predict(model, batch, difficulty, perceptor)
      return loss_sum, acc, all_losses


    @tf.function
    def train_step(batch, difficulty):
      with tf.GradientTape(persistent=True) as tape:
        loss_sum, acc, all_losses = predict(model, batch, difficulty, perceptor)
      # reconstruction loss
      grads = tape.gradient(loss_sum, model.trainable_variables)
      if CFG['TPU']:
        replica_ctx = tf.distribute.get_replica_context()
        grads = replica_ctx.all_reduce("mean", grads)
      optim.apply_gradients(zip(grads, model.trainable_variables))
      if CFG['use_spy']:
        # spy loss
        spy_grads = tape.gradient(all_losses['spy_reconstruction'], model.spy.trainable_variables)
        if CFG['TPU']:
          replica_ctx = tf.distribute.get_replica_context()
          spy_grads = replica_ctx.all_reduce("mean", spy_grads)
        spy_optim.apply_gradients(zip(spy_grads, model.spy.trainable_variables))
      return loss_sum, acc, all_losses


    @tf.function
    def test_step(batch, difficulty):
      batch_loss, acc, all_losses = predict(model, batch, difficulty, perceptor)
      return batch_loss, acc, all_losses


    def aggregate_results(batch_loss, acc, all_losses):
      batch_loss = strategy.reduce("mean", batch_loss, axis=None)
      out_acc = {}
      out_all_losses = {}
      for key in acc.keys():
        out_acc[key] = strategy.reduce("mean", acc[key], axis=None)
      for key in all_losses.keys():
        out_all_losses[key] = strategy.reduce("mean", all_losses[key], axis=None)
      return batch_loss, out_acc, out_all_losses


    def step_fn(batch, difficulty, test=False):
      if test:
        if CFG['TPU']:
          results = strategy.run(test_step, args=(batch, difficulty))
          return aggregate_results(*results)
        else:
          return test_step(batch, difficulty)
      else:
        if CFG['TPU']:
          results = strategy.run(train_step, args=(batch, difficulty))
          return aggregate_results(*results)
        else:
          return train_step(batch, difficulty)


    # ==================== DATA AND MODELS ====================
    if CFG['JUST_VISION']:
      train_ds, _ = make_symbol_data(**CFG, test=False)
      test_ds, _ = make_symbol_data(**CFG, test=True)
    else:
      train_ds, _ = get_dataset(**CFG, test=False)
      test_ds, _ = get_dataset(**CFG, test=True)
    replica_batch_size = CFG['batch_size']
    global_batch_size = strategy.num_replicas_in_sync * replica_batch_size
    tf_train_ds = tf.data.Dataset.from_tensor_slices(train_ds).batch(global_batch_size, drop_remainder=True)
    tf_test_ds = tf.data.Dataset.from_tensor_slices(test_ds).batch(global_batch_size, drop_remainder=True)

    path_prefix = CFG['root_filepath']
    model = get_model()
    run_dummy_batch(model)
    load_weights(model, path_prefix)
    optim = get_optim()
    spy_optim = get_spy_optim()
    perceptor = None
    # perceptor = Perceptor()

  # ==================== NOISY CHANNEL ====================
  difficulty = tf.convert_to_tensor(0)

  # ==================== LOGGING ====================
  log_dir = f"logs/{CFG['run_name']}"
  os.makedirs(log_dir, exist_ok=True)
  train_log_dir = f"{path_prefix}{log_dir}/train"
  test_log_dir = f"{path_prefix}{log_dir}/test"
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  test_summary_writer = tf.summary.create_file_writer(test_log_dir)
  current_time = str(datetime.datetime.now())
  CFG['current_time'] = current_time
  cfg_dir = os.path.join(log_dir, 'config.json')
  if CFG['USE_GS']:
    gs_upload_blob_from_string(json.dumps(CFG, indent=4), cfg_dir, print_str=True)
  else:
    with open(cfg_dir, 'w+') as f:
      f.write(json.dumps(CFG, indent=4))

  # ==================== TRAIN LOOP ====================
  tr_num_samples = train_ds.shape[0] if CFG['JUST_VISION'] else train_ds['adj'].shape[0]
  test_num_samples = test_ds.shape[0] if CFG['JUST_VISION'] else test_ds['adj'].shape[0]
  tr_num_batches = (tr_num_samples // global_batch_size)
  test_num_batches = (test_num_samples // global_batch_size)
  best_epoch_loss = tf.float32.max
  for e_i in range(CFG['epochs']):
    # RESET METRICS
    tr_epoch_loss = 0
    test_epoch_loss = 0
    tr_epoch_acc = {}
    test_epoch_acc = {}
    all_losses = {}
    # TRAIN BATCHES
    b_i = 0
    for train_batch in tf_train_ds:
      b_i += global_batch_size
      batch_loss, batch_acc, batch_all_losses = step_fn(train_batch, difficulty)
      tr_epoch_loss += batch_loss
      update_data_dict(tr_epoch_acc, batch_acc)
      update_data_dict(all_losses, batch_all_losses)
      print(f"(TRAIN) e [{e_i}/{CFG['epochs']}] b [{b_i}/{tr_num_samples}] loss {batch_loss}", end="\r")
    # TEST BATCHES
    b_i = 0
    for test_batch in tf_test_ds:
      b_i += global_batch_size
      batch_loss, batch_acc, _ = step_fn(test_batch, difficulty, test=True)
      test_epoch_loss += batch_loss
      update_data_dict(test_epoch_acc, batch_acc)
      print(f"(TEST) e [{e_i}/{CFG['epochs']}] b [{b_i}/{test_num_samples}] loss {batch_loss}", end="\r")
    # END-OF-EPOCH METRICS
    tr_epoch_loss = tr_epoch_loss / tr_num_batches
    test_epoch_loss = test_epoch_loss / test_num_batches
    tr_epoch_acc = normalize_data_dict(tr_epoch_acc, tr_num_batches)
    test_epoch_acc = normalize_data_dict(test_epoch_acc, test_num_batches)
    all_losses = normalize_data_dict(all_losses, tr_num_batches)
    print(f"EPOCH {e_i} TRAIN LOSS: {tr_epoch_loss} TEST LOSS: {test_epoch_loss}")
    print(f"Train accuracies: {json.dumps(tr_epoch_acc, indent=4)}")
    print(f"Test accuracies: {json.dumps(test_epoch_acc, indent=4)}")
    print(f"All losses: {json.dumps(all_losses, indent=4)}")
    difficulty = update_difficulty(difficulty, tr_epoch_acc)
    print(f"DIFFICULTY FOR NEXT EPOCH: {difficulty}")
    # write metrics to log
    if not CFG['NOLOG']:
      with train_summary_writer.as_default():
        tf.summary.scalar('loss', tr_epoch_loss, step=e_i)
        for name, metric in tr_epoch_acc.items():
          tf.summary.scalar(name, metric, step=e_i)
        for name, specific_loss in all_losses.items():
          tf.summary.scalar(name, specific_loss, step=e_i)
      with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_epoch_loss, step=e_i)
        for name, metric in test_epoch_acc.items():
          tf.summary.scalar(name, metric, step=e_i)
    # SAVE CHECKPOINTS
    if e_i % CFG['save_checkpoint_every'] == 0 and e_i != 0:
      if test_epoch_loss < best_epoch_loss:
        print(f"Saving checkpoint...")
        best_epoch_loss = test_epoch_loss
        save_weights(model, path_prefix)
    # GENERATE VISUAL SAMPLE
    if CFG['VISION'] and e_i % CFG['image_every'] == 0 and e_i != 0:
      sample_imgs, aug_imgs, spy_imgs = get_visual_samples(test_ds, model, test_num_samples, difficulty)
      sample_imgs = color_composite(sample_imgs)
      aug_imgs = color_composite(aug_imgs)
      if CFG['use_spy']: spy_imgs = color_composite(spy_imgs)
      fig, axes = plt.subplots(3, 4) if CFG['use_spy'] else plt.subplots(2, 4)
      for img_i in range(4):
        axes[0][img_i].imshow(sample_imgs[img_i])
        axes[1][img_i].imshow(aug_imgs[img_i])
        if CFG['use_spy']: axes[2][img_i].imshow(spy_imgs[img_i])
      # upload snapshot (or save locally)
      if CFG['USE_GS']:
        gallery_dir = f"gallery/{CFG['run_name']}"
        img_name = os.path.join(gallery_dir, f"{e_i}.png")
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        gs_upload_blob_from_memory(img_data, img_name)
      else:
        gallery_dir = f"{path_prefix}gallery/{CFG['run_name']}"
        os.makedirs(gallery_dir, exist_ok=True)
        img_name = os.path.join(gallery_dir, f"{e_i}.png")
        plt.savefig(img_name, format='png')
      plt.clf()
      plt.cla()
      plt.close()


if __name__ == "__main__":
  main()
