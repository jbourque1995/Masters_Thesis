# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# Modifications made by Jordan Bourque for master thesis research.
# University of New Brunswick, 2026.
# Purpose: Guidance and Scheduling ARDM experimentation and evaluation.
# ==============================================================================

"""Contains the methods to get different datasets.

This patched version avoids TFDS for CIFAR-10 and loads directly from the
local torchvision CIFAR-10 files already present on disk.
"""

import os

import jax
import ml_collections
import numpy as np
import tensorflow as tf
from torchvision.datasets import CIFAR10


def _get_data_root(config):
  """Returns the local CIFAR-10 root directory."""
  if hasattr(config, 'data_dir') and config.data_dir:
    return config.data_dir
  return '/home/jordan/datasets/cifar10'


def _load_cifar10_local(config):
  """Load CIFAR-10 directly from local extracted files via torchvision."""
  data_root = _get_data_root(config)

  train_set = CIFAR10(root=data_root, train=True, download=False)
  test_set = CIFAR10(root=data_root, train=False, download=False)

  train_images = np.asarray(train_set.data, dtype=np.int32)
  test_images = np.asarray(test_set.data, dtype=np.int32)

  return train_images, test_images


def create_datasets(config, data_rng):
  """Create datasets for training and evaluation."""
  del data_rng  # Not needed in this local CIFAR loader.

  if config.dataset != 'cifar10':
    raise ValueError(
        'This patched datasets.py currently supports only config.dataset="cifar10".'
    )

  if config.batch_size % jax.device_count() != 0:
    raise ValueError(
        f'Batch size ({config.batch_size}) must be divisible by '
        f'the number of devices ({jax.device_count()}).'
    )

  if config.test_batch_size % jax.local_device_count() != 0:
    raise ValueError(
        f'Test batch size ({config.test_batch_size}) must be divisible by '
        f'the number of local devices ({jax.local_device_count()}).'
    )

  per_device_batch_size = config.batch_size // jax.device_count()
  test_device_batch_size = config.test_batch_size // jax.local_device_count()

  train_images, test_images = _load_cifar10_local(config)

  print('train', train_images.shape[0])
  print('test', test_images.shape[0])

  def drop_info(batch):
    """Removes unwanted keys from batch."""
    if 'id' in batch:
      batch.pop('id')
    if 'rng' in batch:
      batch.pop('rng')
    return batch

  if config.data_augmentation:
    should_augment = True
    should_randflip = True
    should_rotate = True
  else:
    should_augment = False
    should_randflip = False
    should_rotate = False

  def augment(batch):
    img = tf.cast(batch['image'], tf.float32)
    aug = None
    if should_augment:
      if should_randflip:
        img_flipped = tf.image.flip_left_right(img)
        aug = tf.random.uniform(shape=[]) > 0.5
        img = tf.where(aug, img_flipped, img)
      if should_rotate:
        u = tf.random.uniform(shape=[])
        k = tf.cast(tf.floor(4. * u), tf.int32)
        img = tf.image.rot90(img, k=k)
        aug = aug | (k > 0)
    if aug is None:
      aug = tf.convert_to_tensor(False, dtype=tf.bool)

    out = batch.copy()
    out['image'] = tf.cast(img, tf.int32)
    return out

  def preprocess_train(batch):
    return augment(drop_info(batch))

  def preprocess_eval(batch):
    batch = drop_info(batch)
    batch['image'] = tf.cast(batch['image'], tf.int32)
    return batch

  train_ds = tf.data.Dataset.from_tensor_slices({
      'image': train_images,
  })
  train_ds = train_ds.shuffle(
      buffer_size=train_images.shape[0],
      reshuffle_each_iteration=True)
  train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = train_ds.batch(
      jax.local_device_count() * per_device_batch_size,
      drop_remainder=True)
  train_ds = train_ds.map(
      lambda batch: {
          'image': tf.reshape(
              batch['image'],
              [jax.local_device_count(), per_device_batch_size, 32, 32, 3])
      },
      num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

  eval_ds = tf.data.Dataset.from_tensor_slices({
      'image': test_images,
  })
  eval_ds = eval_ds.repeat(config.num_eval_passes)
  eval_ds = eval_ds.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)
  eval_ds = eval_ds.batch(
      jax.local_device_count() * test_device_batch_size,
      drop_remainder=False)
  eval_ds = eval_ds.map(
      lambda batch: {
          'image': tf.reshape(
              batch['image'],
              [jax.local_device_count(), -1, 32, 32, 3])
      },
      num_parallel_calls=tf.data.AUTOTUNE)
  eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

  info = {
      'train_examples': int(train_images.shape[0]),
      'test_examples': int(test_images.shape[0]),
  }

  return info, train_ds, eval_ds


def get_dataset(config, data_rng):
  """Function that combines data loading for different datasets."""
  _, train_ds, test_ds = create_datasets(config, data_rng)

  if config.dataset == 'mnist':
    shape = (28, 28, 1)
    n_classes = 256
  elif config.dataset == 'binarized_mnist':
    shape = (28, 28, 1)
    n_classes = 2
  elif config.dataset == 'cifar10':
    shape = (32, 32, 3)
    n_classes = 256
  else:
    raise ValueError

  return train_ds, test_ds, shape, n_classes