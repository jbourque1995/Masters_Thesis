# coding=utf-8
# ==============================================================================
# Modifications made by Jordan Bourque for master thesis research.
# University of New Brunswick, 2026.
# Purpose: Guidance and Scheduling ARDM experimentation and evaluation.
# ==============================================================================

"""Export real CIFAR-10 images to PNG files for evaluation metrics."""

import os

from absl import app
from absl import flags
from absl import logging
from PIL import Image
from torchvision.datasets import CIFAR10
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_root',
    '/home/jordan/datasets',
    'Root directory where torchvision stores/downloads CIFAR-10.')
flags.DEFINE_string(
    'output_dir',
    '/home/jordan/runs/ardm/real_cifar10',
    'Directory where PNG images will be exported.')
flags.DEFINE_bool(
    'train_split',
    True,
    'Whether to export the training split. If False, exports the test split.')
flags.DEFINE_bool(
    'download',
    True,
    'Whether to download CIFAR-10 if it is missing.')


def main(argv):
  del argv

  tf.io.gfile.makedirs(FLAGS.output_dir)

  dataset = CIFAR10(
      root=FLAGS.data_root,
      train=FLAGS.train_split,
      download=FLAGS.download)

  split_name = 'train' if FLAGS.train_split else 'test'
  logging.info('Exporting CIFAR-10 %s split with %d images to %s',
               split_name, len(dataset), FLAGS.output_dir)

  for idx, (img, label) in enumerate(dataset):
    # img is already a PIL Image from torchvision.
    filename = f'{split_name}_{idx:05d}_label{label}.png'
    out_path = os.path.join(FLAGS.output_dir, filename)
    with tf.io.gfile.GFile(out_path, 'wb') as fp:
      img.save(fp, format='PNG')

    if (idx + 1) % 5000 == 0:
      logging.info('Exported %d / %d images', idx + 1, len(dataset))

  logging.info('Done. Exported %d images to %s', len(dataset), FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)