# coding=utf-8
# ==============================================================================
# Modifications made by Jordan Bourque for master thesis research.
# University of New Brunswick, 2026.
# Purpose: Guidance and Scheduling ARDM experimentation and evaluation.
# ==============================================================================

"""Backfill FID / IS / SSIM for saved UGARDM checkpoints.

This does not train. It loads selected ckpt_X files, samples images, saves them
to generated_eval/epoch_X, computes metrics, and writes TensorBoard scalars.
"""

import os

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from flax import serialization
import jax
import numpy as np
import optax
import tensorflow as tf

from ml_collections.config_flags import config_flags

from ardm_basicdisc_middle_end.experiments.images import checkpoint
from ardm_basicdisc_middle_end.experiments.images import custom_train_state
from ardm_basicdisc_middle_end.experiments.images import datasets
from ardm_basicdisc_middle_end.experiments.images import metric_logging
from ardm_basicdisc_middle_end.experiments.images import train


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "work_unit_dir",
    None,
    "Run directory containing ckpt_X files.")
flags.DEFINE_list(
    "epochs",
    None,
    "Comma-separated checkpoint numbers to evaluate, e.g. 50,100,150,200.")
flags.DEFINE_integer(
    "num_samples_override",
    None,
    "Override config.num_samples for evaluation generation.")
flags.DEFINE_bool(
    "overwrite_existing",
    False,
    "If false, skips epochs that already have a metrics JSON.")

config_flags.DEFINE_config_file("config", lock_config=False)


def _restore_exact_checkpoint(work_dir, state, ckpt_num):
  """Restore exactly ckpt_<ckpt_num>, not the latest checkpoint."""
  ckpt_path = os.path.join(work_dir, f"ckpt_{ckpt_num}")
  if not tf.io.gfile.exists(ckpt_path):
    raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

  logging.info("Restoring exact checkpoint: %s", ckpt_path)
  save_state = checkpoint.SaveState(state, ckpt_num)
  with tf.io.gfile.GFile(ckpt_path, "rb") as fp:
    save_state = serialization.from_bytes(save_state, fp.read())
  return save_state.train_state


def _make_initial_state(config):
  """Initialize model and empty train state with the right shape."""
  rng = jax.random.PRNGKey(config.seed)
  data_rng, rng = jax.random.split(rng)

  _, _, shape, num_classes = datasets.get_dataset(config, data_rng)
  config.data_shape = shape
  config.num_classes = num_classes

  rng, init_rng = jax.random.split(rng)
  model, variables = train.model_setup(init_rng, config)

  tx = optax.adam(
      config.learning_rate,
      b1=0.9,
      b2=config.beta2,
      eps=1e-08,
      eps_root=0.0)

  state = custom_train_state.TrainState.create(
      params=variables["params"],
      tx=tx)

  return model, state, rng


def main(argv):
  del argv

  tf.config.experimental.set_visible_devices([], "GPU")

  if FLAGS.work_unit_dir is None:
    raise ValueError("--work_unit_dir is required.")
  if FLAGS.epochs is None:
    raise ValueError("--epochs is required, e.g. --epochs=50,100,150,200")

  work_dir = FLAGS.work_unit_dir
  config = FLAGS.config

  if FLAGS.num_samples_override is not None:
    config.num_samples = FLAGS.num_samples_override

  # Local generated_eval/metrics under this run, shared real_cifar10 folder.
  train._prepare_run_output_dirs(config, work_dir)

  writer = metric_writers.create_default_writer(work_dir)
  model, init_state, rng = _make_initial_state(config)

  ckpt_nums = [int(x) for x in FLAGS.epochs]

  for ckpt_num in ckpt_nums:
    json_path = os.path.join(config.metrics_dir, f"epoch_{ckpt_num}.json")
    if tf.io.gfile.exists(json_path) and not FLAGS.overwrite_existing:
      logging.info("Skipping ckpt_%d because metrics already exist: %s",
                   ckpt_num, json_path)
      continue

    state = _restore_exact_checkpoint(work_dir, init_state, ckpt_num)
    sample_rng = jax.random.fold_in(rng, ckpt_num)

    logging.info("Sampling %d images from ckpt_%d", config.num_samples, ckpt_num)
    chain = model.sample(sample_rng, state.ema_params, config.num_samples)
    final_images = np.asarray(chain[-1])

    metric_dict = metric_logging.run_generation_metrics(
        final_images=final_images,
        epoch=ckpt_num,
        real_image_dir=config.real_image_dir,
        generated_eval_dir=config.generated_eval_dir,
        metrics_dir=config.metrics_dir,
        compute_fid=config.compute_fid,
        compute_is=config.compute_is,
        compute_clip=config.compute_clip,
        compute_ssim=config.compute_ssim,
        clip_prompt=config.clip_prompt,
        num_clip_images=config.num_clip_images,
        num_ssim_images=config.num_ssim_images,
    )

    if metric_dict:
      writer.write_scalars(ckpt_num, metric_dict)
      writer.flush()

    logging.info("Finished ckpt_%d metrics: %s", ckpt_num, metric_dict)

  writer.close()


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
