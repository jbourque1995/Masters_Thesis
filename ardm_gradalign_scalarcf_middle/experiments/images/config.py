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
  config.save_every = 5
  config.model = 'bit_ao'  # ao_arm, bit_ao
  # upscale_mode choices: zero_least_significant, augment_least_significant
  config.upscale_mode = 'zero_least_significant'
  config.upscale_direct_parametrization = False
  config.upscale_branch_factor = 4
  config.elbo_mode = 'uniform'  # choices: uniform, antithetic
  config.ce_term = 0.001
  config.learning_rate = 0.0001
  config.beta2 = 0.999  # Turns out to be more stable.
  config.momentum = 0.9
  config.clip_grad = 100.
  config.batch_size = 16
  config.test_batch_size = 16  # Divisible by local device count.
  config.num_epochs = 200
  config.dataset = 'cifar10'
  config.data_augmentation = False
  config.detailed_eval_every = 50 #was 50
  config.num_eval_passes = 4
  config.seed = 0
  config.num_samples = 64

  # UGARDM discriminator-feature experiment controls.
  # none | unet_end | unet_middle | unet_mid_end
  config.guidance_mode = 'unet_middle'
  config.disc_weight_mid = 0.0
  config.disc_weight_end = 0.0
  config.reason_weight_mid = 0.0
  config.reason_weight_end = 0.0
  # No discriminator-only pretraining: critic updates and generator guidance
  # begin together, then guidance ramps in smoothly.
  config.disc_warmup_epochs = 0  # compatibility alias for older code paths
  config.disc_start_epoch = 0
  config.guidance_ramp_epochs = 0
  config.reason_reference_mode = 'mask_matched_ema'
  config.disc_logit_penalty = 0.001
  config.disc_normalize_input = True
  config.disc_hidden_dim = 128
  config.disc_num_layers = 3
  config.disc_dropout = 0.05
  config.disc_learning_rate = 0.00001
  config.disc_beta1 = 0.5
  config.disc_beta2 = 0.999

  # No-extra-capacity credit-assignment experiments. These do not increase
  # training data, generator parameter count, width, depth, or tensor shape.
  config.align_aux_guidance = False
  config.align_aux_mode = 'off'  # off | hard | soft
  config.align_aux_min_cosine = 0.0
  config.align_aux_eps = 1e-8
  config.disc_generator_train = False
  config.cf_mid_weight = 0.0001
  config.cf_mid_drop_rate = 0.25
  config.cf_mid_delta_clip = 0.05
  config.cf_mid_inverted_dropout = False
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

  config.architecture = D(
      n_channels=64,
      num_res_blocks=2,
      num_heads=1,
      ch_mult=[1, 2, 2, 2],
      attn_resolutions=[32, 16, 8, 4],
      dropout=0.0,
  )
  return config