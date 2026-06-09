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

"""Configuration file for autoregressive diffusion on images."""

# pylint: disable=invalid-name

import ml_collections


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.save_every = 5 #was 2
  config.model = 'bit_ao'  # ao_arm, bit_ao
  # upscale_mode choices: zero_least_significant, augment_least_significant
  config.upscale_mode = 'zero_least_significant'
  config.upscale_direct_parametrization = False
  config.upscale_branch_factor = 4
  config.elbo_mode = 'curriculum_t'  # choices: uniform, antithetic, stratified_t, adaptive_t, curriculum_t
  config.ce_term = 0.001
  config.learning_rate = 0.0001
  config.beta2 = 0.999  # Turns out to be more stable.
  config.momentum = 0.9
  config.clip_grad = 100.
  config.batch_size = 16
  config.test_batch_size = 16  # Divisible by 8 (TPU pods), divides test 10000
  config.num_epochs = 200 #was 6000
  config.dataset = 'cifar10'
  config.data_augmentation = False
  config.detailed_eval_every = 50 #was 50
  config.num_eval_passes = 4
  config.seed = 0
  config.num_samples = 64
  # MODIFICATION BY JORDAN BOURQUE
  config.eval_gen_every = 50
  config.num_eval_gen_samples = 64
  config.real_image_dir = '/home/jordan/runs/ardm/real_cifar10'
  config.generated_eval_dir = 'generated_eval'
  config.metrics_dir = 'metrics'
  config.data_dir = '/home/jordan/datasets/cifar10'
  config.compute_fid = True
  config.compute_is = True
  config.compute_clip = False
  config.compute_ssim = True
  config.clip_prompt = 'a CIFAR-10 image'
  config.num_clip_images = 100
  config.num_ssim_images = 100

  # Only for ao_arm and upscale ardm, not possible for more fancy
  # destruction processes:
  config.output_distribution = 'softmax'
  config.num_mixtures = 30



  # Adaptive ARDM subproblem scheduling experiments.
  # These options do not increase model size, training data, or architecture shape.
  # stratified_t: unbiased stratified timestep sampling within each batch.
  # adaptive_t: loss-aware timestep bucket sampling using KL/NELBO history.
  # curriculum_t: starts from easier high-observed-mask states and anneals to uniform.
  config.adaptive_start_epoch = 10
  config.adaptive_num_buckets = 128
  config.adaptive_alpha = 0.5
  config.adaptive_uniform_mix = 0.25
  config.adaptive_eps = 1e-6
  config.adaptive_max_prob_ratio = 8.0

  config.curriculum_start_observed_fraction = 0.75
  config.curriculum_end_observed_fraction = 0.0
  config.curriculum_anneal_epochs = 100

  config.architecture = D(
      n_channels=64,
      num_res_blocks=2,
      num_heads=1,
      ch_mult=[1, 2, 2, 2],
      attn_resolutions=[32, 16, 8, 4],
      dropout=0.0,
  )
  return config