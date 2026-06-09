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

"""Train file.

Library file which executes the training and evaluation loop.
"""

# pytype: disable=wrong-keyword-args

import functools
import os
import pickle
import time


from absl import logging
from clu import metric_writers
from ardm_gradalign_scalarcf_middle.experiments.images import metric_logging
from ardm_gradalign_scalarcf_middle.experiments.images.architectures import discriminator
from flax.training import train_state
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

from ardm_gradalign_scalarcf_middle.experiments.images import checkpoint
from ardm_gradalign_scalarcf_middle.experiments.images import custom_train_state
from ardm_gradalign_scalarcf_middle.experiments.images import datasets
from ardm_gradalign_scalarcf_middle.experiments.images.architectures import unet
from ardm_gradalign_scalarcf_middle.model.autoregressive_diffusion import ao_arm
from ardm_gradalign_scalarcf_middle.model.autoregressive_diffusion import bit_ao
from ardm_gradalign_scalarcf_middle.utils import util_fns


def discriminator_hinge_loss(real_logits, fake_logits, logit_penalty=0.0):
  """Patch-hinge discriminator loss with optional logit-magnitude penalty."""
  real_loss = jnp.mean(jax.nn.relu(1.0 - real_logits))
  fake_loss = jnp.mean(jax.nn.relu(1.0 + fake_logits))
  loss = real_loss + fake_loss
  if logit_penalty > 0.0:
    # Prevents the critic from driving logits to very large magnitudes after it
    # has already separated real/fake internal states.
    loss = loss + logit_penalty * 0.5 * (
        jnp.mean(jnp.square(real_logits)) + jnp.mean(jnp.square(fake_logits)))
  return loss


def generator_hinge_loss(fake_logits):
  return -jnp.mean(fake_logits)


def spatial_feature_alignment_loss(fake_features, real_features, eps=1e-6):
  """Cosine-distance feature matching against a frozen/EMA reference map.

  This is the explicit reason-guidance term. It keeps spatial layout intact and
  asks each current internal feature vector to point toward the corresponding
  teacher/reference feature vector.
  """
  real_features = jax.lax.stop_gradient(real_features)
  fake = fake_features.astype(jnp.float32)
  real = real_features.astype(jnp.float32)
  fake = fake / (jnp.linalg.norm(fake, axis=-1, keepdims=True) + eps)
  real = real / (jnp.linalg.norm(real, axis=-1, keepdims=True) + eps)
  cosine = jnp.sum(fake * real, axis=-1)
  return jnp.mean(1.0 - cosine)


def mean_feature_cosine(fake_features, real_features, eps=1e-6):
  real_features = jax.lax.stop_gradient(real_features)
  fake = fake_features.astype(jnp.float32)
  real = real_features.astype(jnp.float32)
  fake = fake / (jnp.linalg.norm(fake, axis=-1, keepdims=True) + eps)
  real = real / (jnp.linalg.norm(real, axis=-1, keepdims=True) + eps)
  return jnp.mean(jnp.sum(fake * real, axis=-1))


def mean_feature_norm(features, eps=1e-12):
  features = features.astype(jnp.float32)
  return jnp.mean(jnp.sqrt(jnp.sum(jnp.square(features), axis=-1) + eps))


def _guidance_progress(epoch, config):
  """Linear ramp for generator-side discriminator/reason guidance."""
  start = getattr(config, 'disc_start_epoch', config.disc_warmup_epochs)
  ramp_epochs = max(1, getattr(config, 'guidance_ramp_epochs', 1))
  epoch_f = jnp.asarray(epoch, dtype=jnp.float32)
  start_f = jnp.asarray(start, dtype=jnp.float32)
  progress = (epoch_f - start_f + 1.0) / float(ramp_epochs)
  return jnp.clip(jnp.where(epoch_f < start_f, 0.0, progress), 0.0, 1.0)


def _disc_update_factor(epoch, config):
  """0 before disc_start_epoch, then 1. Used to avoid critic pretraining alone."""
  start = getattr(config, 'disc_start_epoch', config.disc_warmup_epochs)
  return jnp.where(jnp.asarray(epoch) < start, 0.0, 1.0)


def _scale_tree(tree, scale):
  return jax.tree_util.tree_map(lambda x: x * scale.astype(x.dtype), tree)


def _tree_dot(tree_a, tree_b):
  leaves_a = jax.tree_util.tree_leaves(tree_a)
  leaves_b = jax.tree_util.tree_leaves(tree_b)
  dots = [jnp.sum(a.astype(jnp.float32) * b.astype(jnp.float32))
          for a, b in zip(leaves_a, leaves_b)]
  return functools.reduce(lambda x, y: x + y, dots, jnp.array(0.0, jnp.float32))


def _tree_norm(tree, eps=1e-12):
  return jnp.sqrt(_tree_dot(tree, tree) + eps)


def _combine_gradient_aligned(ardm_grads, aux_grads, config):
  """Combine ARDM and auxiliary gradients using gradient agreement.

  The auxiliary gradient is allowed to update the generator only when it agrees
  with the ARDM gradient. This changes the learning rule without adding model
  parameters, data, width, depth, or tensor-shape changes.
  """
  ardm_norm = _tree_norm(ardm_grads)
  aux_norm = _tree_norm(aux_grads)
  eps = getattr(config, 'align_aux_eps', 1e-8)
  cosine = _tree_dot(ardm_grads, aux_grads) / (ardm_norm * aux_norm + eps)
  min_cosine = getattr(config, 'align_aux_min_cosine', 0.0)
  mode = getattr(config, 'align_aux_mode', 'soft')

  if mode == 'off':
    factor = jnp.array(1.0, dtype=jnp.float32)
  elif mode == 'hard':
    factor = jnp.where(cosine > min_cosine, 1.0, 0.0).astype(jnp.float32)
  elif mode == 'soft':
    # Smoothly trust the auxiliary gradient in proportion to its agreement.
    # cosine <= min_cosine gives zero; cosine near 1 gives near full trust.
    denom = jnp.maximum(1.0 - min_cosine, eps)
    factor = jnp.clip((cosine - min_cosine) / denom, 0.0, 1.0)
  else:
    raise ValueError(
        f'Unknown align_aux_mode={mode!r}; expected off, hard, or soft.')

  grads = jax.tree_util.tree_map(lambda ga, gx: ga + factor.astype(gx.dtype) * gx,
                                 ardm_grads, aux_grads)
  return grads, cosine, factor, ardm_norm, aux_norm


def _make_channel_dropout_mask(rng, num_channels, drop_rate, dtype,
                               inverted_dropout=False):
  """Create a per-channel counterfactual mask with no trainable params."""
  drop_rate = jnp.asarray(drop_rate, dtype=jnp.float32)
  keep_prob = jnp.clip(1.0 - drop_rate, 1e-6, 1.0)
  keep = jax.random.bernoulli(
      rng, keep_prob, shape=(1, 1, 1, int(num_channels))).astype(dtype)
  if inverted_dropout:
    keep = keep / keep_prob.astype(dtype)
  return keep


def _normalized_channel_energy(features, eps=1e-6):
  """Per-channel energy after per-location RMS normalization.

  The normalization makes the counterfactual reward redistribute feature energy
  across existing channels instead of simply increasing the whole feature norm.
  """
  h = features.astype(jnp.float32)
  h = h / jnp.sqrt(jnp.mean(jnp.square(h), axis=-1, keepdims=True) + eps)
  return jnp.mean(jnp.square(h), axis=(0, 1, 2))


def counterfactual_channel_credit_loss(fake_features, channel_mask,
                                       full_loss, masked_loss, config):
  """Credit assignment from counterfactual middle-channel dropout.

  If masking dropped channels makes ARDM worse, those dropped channels receive
  positive credit. If masking makes ARDM better, those dropped channels receive
  negative credit. The loss uses normalized channel energy so it can suppress or
  redistribute existing features without adding parameters or changing shape.
  """
  delta = jax.lax.stop_gradient(masked_loss - full_loss)
  delta_clip = getattr(config, 'cf_mid_delta_clip', 0.05)
  delta = jnp.clip(delta, -delta_clip, delta_clip)

  dropped = (channel_mask <= 1e-6).astype(jnp.float32).reshape((-1,))
  drop_count = jnp.maximum(jnp.sum(dropped), 1.0)
  channel_energy = _normalized_channel_energy(fake_features)
  dropped_energy = jnp.sum(channel_energy * dropped) / drop_count

  # Minimization behavior:
  #   delta > 0: dropout hurt, so increase energy in dropped/helpful channels.
  #   delta < 0: dropout helped, so decrease energy in dropped/harmful channels.
  return -delta * dropped_energy, delta, dropped_energy


def _uses_mid(config):
  return config.guidance_mode in ('unet_middle', 'unet_mid_end')


def _uses_end(config):
  return config.guidance_mode in ('unet_end', 'unet_mid_end')


def _apply_discriminator(model, params, features, train, rng=None):
  """Apply discriminator safely with optional dropout RNG.

  Flax dropout requires a `dropout` RNG whenever a Dropout module is executed
  in training mode. Passing the RNG unconditionally for training calls makes the
  discriminator safe for both `disc_dropout = 0.0` and `disc_dropout > 0.0`.
  """
  if train:
    if rng is None:
      raise ValueError('Training discriminator apply requires a dropout RNG.')
    return model.apply(
        {'params': params}, features, train=True, rngs={'dropout': rng})
  return model.apply({'params': params}, features, train=False)


def _sanitize_guidance_config(config):
  """Normalize UGARDM config values to avoid silent mismatches."""
  valid_modes = ('none', 'unet_end', 'unet_middle', 'unet_mid_end')
  if config.guidance_mode not in valid_modes:
    raise ValueError(
        f'Unknown guidance_mode={config.guidance_mode!r}; expected {valid_modes}.')

  # Disabled branches should have exactly zero weight. This avoids logs that
  # look like a branch was active when it was not.
  if config.guidance_mode == 'none':
    config.disc_weight_mid = 0.0
    config.disc_weight_end = 0.0
    config.reason_weight_mid = 0.0
    config.reason_weight_end = 0.0
  elif config.guidance_mode == 'unet_end':
    config.disc_weight_mid = 0.0
    config.reason_weight_mid = 0.0
  elif config.guidance_mode == 'unet_middle':
    config.disc_weight_end = 0.0
    config.reason_weight_end = 0.0

  if not hasattr(config, 'disc_start_epoch'):
    config.disc_start_epoch = config.disc_warmup_epochs
  if not hasattr(config, 'guidance_ramp_epochs'):
    config.guidance_ramp_epochs = 1
  if not hasattr(config, 'disc_logit_penalty'):
    config.disc_logit_penalty = 0.0
  if not hasattr(config, 'reason_reference_mode'):
    config.reason_reference_mode = 'mask_matched_ema'

  # No-extra-capacity credit-assignment controls. These alter gradient routing
  # and feature credit only; they do not change data, generator parameter count,
  # tensor shapes, width, or depth.
  if not hasattr(config, 'align_aux_guidance'):
    config.align_aux_guidance = False
  if not hasattr(config, 'align_aux_mode'):
    config.align_aux_mode = 'soft'
  if not hasattr(config, 'align_aux_min_cosine'):
    config.align_aux_min_cosine = 0.0
  if not hasattr(config, 'align_aux_eps'):
    config.align_aux_eps = 1e-8
  if config.align_aux_mode not in ('off', 'hard', 'soft'):
    raise ValueError(
        f'Unknown align_aux_mode={config.align_aux_mode!r}; expected off, hard, or soft.')

  if not hasattr(config, 'disc_generator_train'):
    # The generator should see a stable critic. The discriminator itself still
    # trains with dropout in its own update.
    config.disc_generator_train = False

  if not hasattr(config, 'cf_mid_weight'):
    config.cf_mid_weight = 0.0
  if not hasattr(config, 'cf_mid_drop_rate'):
    config.cf_mid_drop_rate = 0.25
  if not hasattr(config, 'cf_mid_delta_clip'):
    config.cf_mid_delta_clip = 0.05
  if not hasattr(config, 'cf_mid_inverted_dropout'):
    config.cf_mid_inverted_dropout = False
  if config.cf_mid_drop_rate < 0.0 or config.cf_mid_drop_rate >= 1.0:
    raise ValueError('cf_mid_drop_rate must be in [0.0, 1.0).')
  if config.cf_mid_weight > 0.0 and not _uses_mid(config):
    logging.warning(
        'cf_mid_weight is nonzero but guidance_mode does not use the middle branch. '
        'Counterfactual middle credit will be inactive.')

  active_weight = (config.disc_weight_mid + config.disc_weight_end +
                   config.reason_weight_mid + config.reason_weight_end +
                   config.cf_mid_weight)
  if active_weight > 0 and config.num_epochs < config.disc_start_epoch:
    logging.warning(
        'Guidance weights are nonzero, but num_epochs=%s is < '
        'disc_start_epoch=%s. Guidance will not affect the generator.',
        config.num_epochs, config.disc_start_epoch)

  if config.batch_size % jax.local_device_count() != 0:
    raise ValueError(
        f'batch_size={config.batch_size} must be divisible by '
        f'jax.local_device_count()={jax.local_device_count()}.')


def train_step(rng, batch, state, disc_mid_state, disc_end_state, epoch, model,
               disc_mid_model, disc_end_model, config):
  """Train for a single step with optional UGARDM feature discriminators."""
  logging.info('Training step...')
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  (model_rng, gen_mid_dropout_rng, gen_end_dropout_rng,
   disc_mid_real_dropout_rng, disc_mid_fake_dropout_rng,
   disc_end_real_dropout_rng, disc_end_fake_dropout_rng,
   cf_mid_mask_rng) = jax.random.split(rng, 8)

  use_mid = _uses_mid(config)
  use_end = _uses_end(config)
  use_cf_mid = bool(use_mid and getattr(config, 'cf_mid_weight', 0.0) > 0.0)

  mid_channels = config.architecture.n_channels * config.architecture.ch_mult[-1]
  cf_mid_mask = None
  if use_cf_mid:
    cf_mid_mask = _make_channel_dropout_mask(
        cf_mid_mask_rng, mid_channels, config.cf_mid_drop_rate, jnp.float32,
        inverted_dropout=getattr(config, 'cf_mid_inverted_dropout', False))

  def loss_components_fn(params):
    (elbo_value, elbo_per_t, ce_value, t, fake_features, real_features,
     feature_info) = model.elbo_with_features(
         model_rng, params, batch['image'], train=True,
         reference_params=state.ema_params, reference_train=False,
         middle_feature_mask=cf_mid_mask)
    ardm_loss = -elbo_value.mean(0) - config.ce_term * ce_value.mean(0)

    gen_mid_loss = jnp.array(0.0, dtype=ardm_loss.dtype)
    gen_end_loss = jnp.array(0.0, dtype=ardm_loss.dtype)
    reason_mid_loss = jnp.array(0.0, dtype=ardm_loss.dtype)
    reason_end_loss = jnp.array(0.0, dtype=ardm_loss.dtype)
    cf_mid_loss = jnp.array(0.0, dtype=ardm_loss.dtype)
    cf_mid_delta = jnp.array(0.0, dtype=ardm_loss.dtype)
    cf_mid_dropped_energy = jnp.array(0.0, dtype=ardm_loss.dtype)
    cf_mid_masked_ardm_loss = ardm_loss
    cf_mid_drop_fraction = jnp.array(0.0, dtype=ardm_loss.dtype)

    if use_mid:
      fake_mid_logits = _apply_discriminator(
          disc_mid_model, disc_mid_state.params, fake_features['middle'],
          train=getattr(config, 'disc_generator_train', False),
          rng=gen_mid_dropout_rng if getattr(config, 'disc_generator_train', False) else None)
      gen_mid_loss = generator_hinge_loss(fake_mid_logits)
      reason_mid_loss = spatial_feature_alignment_loss(
          fake_features['middle'], real_features['middle'])

      if use_cf_mid:
        cf_mid_elbo = feature_info['cf_mid_elbo']
        cf_mid_ce = feature_info['cf_mid_ce']
        cf_mid_masked_ardm_loss = (
            -cf_mid_elbo.mean(0) - config.ce_term * cf_mid_ce.mean(0))
        cf_mid_loss, cf_mid_delta, cf_mid_dropped_energy = (
            counterfactual_channel_credit_loss(
                fake_features['middle'], cf_mid_mask, ardm_loss,
                cf_mid_masked_ardm_loss, config))
        cf_mid_drop_fraction = jnp.mean((cf_mid_mask <= 1e-6).astype(jnp.float32))

    if use_end:
      fake_end_logits = _apply_discriminator(
          disc_end_model, disc_end_state.params, fake_features['end'],
          train=getattr(config, 'disc_generator_train', False),
          rng=gen_end_dropout_rng if getattr(config, 'disc_generator_train', False) else None)
      gen_end_loss = generator_hinge_loss(fake_end_logits)
      reason_end_loss = spatial_feature_alignment_loss(
          fake_features['end'], real_features['end'])

    guidance_progress = _guidance_progress(epoch, config)
    disc_weight_mid = config.disc_weight_mid * guidance_progress
    disc_weight_end = config.disc_weight_end * guidance_progress
    reason_weight_mid = config.reason_weight_mid * guidance_progress
    reason_weight_end = config.reason_weight_end * guidance_progress
    cf_mid_weight = config.cf_mid_weight * guidance_progress

    aux_loss = (disc_weight_mid * gen_mid_loss
                + disc_weight_end * gen_end_loss
                + reason_weight_mid * reason_mid_loss
                + reason_weight_end * reason_end_loss
                + cf_mid_weight * cf_mid_loss)
    loss = ardm_loss + aux_loss
    aux = (ardm_loss, aux_loss, gen_mid_loss, gen_end_loss, reason_mid_loss,
           reason_end_loss, cf_mid_loss, cf_mid_delta, cf_mid_dropped_energy,
           cf_mid_masked_ardm_loss, cf_mid_drop_fraction, disc_weight_mid,
           disc_weight_end, reason_weight_mid, reason_weight_end, cf_mid_weight,
           guidance_progress, elbo_value, elbo_per_t, ce_value, t,
           fake_features, real_features, feature_info)
    return loss, aux

  def ardm_loss_fn(params):
    _, aux = loss_components_fn(params)
    return aux[0], aux

  def aux_loss_fn(params):
    _, aux = loss_components_fn(params)
    return aux[1], aux

  align_aux_guidance = bool(getattr(config, 'align_aux_guidance', False))
  if align_aux_guidance:
    ardm_grad_fn = jax.value_and_grad(ardm_loss_fn, has_aux=True)
    (ardm_loss, aux), ardm_grads = ardm_grad_fn(state.params)
    aux_grad_fn = jax.value_and_grad(aux_loss_fn, has_aux=True)
    (aux_loss, _), aux_grads = aux_grad_fn(state.params)
    ardm_grads = jax.lax.pmean(ardm_grads, axis_name='batch')
    aux_grads = jax.lax.pmean(aux_grads, axis_name='batch')
    grads, aux_grad_cosine, aux_alignment_factor, ardm_grad_norm_raw, aux_grad_norm_raw = (
        _combine_gradient_aligned(ardm_grads, aux_grads, config))
    loss = ardm_loss + jax.lax.stop_gradient(aux_alignment_factor) * aux_loss
    (ardm_loss, aux_loss, gen_mid_loss, gen_end_loss, reason_mid_loss,
     reason_end_loss, cf_mid_loss, cf_mid_delta, cf_mid_dropped_energy,
     cf_mid_masked_ardm_loss, cf_mid_drop_fraction, disc_weight_mid,
     disc_weight_end, reason_weight_mid, reason_weight_end, cf_mid_weight,
     guidance_progress, elbo_value, elbo_per_t, ce_value, t, fake_features,
     real_features, feature_info) = aux
  else:
    grad_fn = jax.value_and_grad(loss_components_fn, has_aux=True)
    (loss, (ardm_loss, aux_loss, gen_mid_loss, gen_end_loss, reason_mid_loss,
            reason_end_loss, cf_mid_loss, cf_mid_delta, cf_mid_dropped_energy,
            cf_mid_masked_ardm_loss, cf_mid_drop_fraction, disc_weight_mid,
            disc_weight_end, reason_weight_mid, reason_weight_end, cf_mid_weight,
            guidance_progress, elbo_value, elbo_per_t, ce_value, t,
            fake_features, real_features, feature_info)), grads = (
                grad_fn(state.params))
    grads = jax.lax.pmean(grads, axis_name='batch')
    aux_grad_cosine = jnp.array(0.0, dtype=loss.dtype)
    aux_alignment_factor = jnp.array(1.0, dtype=loss.dtype)
    ardm_grad_norm_raw = jnp.array(0.0, dtype=loss.dtype)
    aux_grad_norm_raw = jnp.array(0.0, dtype=loss.dtype)
  if config.clip_grad > 0:
    grads, grad_norm = util_fns.clip_by_global_norm(
        grads, clip_norm=config.clip_grad)
  else:
    grad_norm = util_fns.global_norm(grads)
  state = state.apply_gradients(grads=grads)

  disc_mid_loss = jnp.array(0.0, dtype=loss.dtype)
  disc_end_loss = jnp.array(0.0, dtype=loss.dtype)
  disc_mid_real_logits = jnp.array(0.0, dtype=loss.dtype)
  disc_mid_fake_logits = jnp.array(0.0, dtype=loss.dtype)
  disc_end_real_logits = jnp.array(0.0, dtype=loss.dtype)
  disc_end_fake_logits = jnp.array(0.0, dtype=loss.dtype)
  disc_mid_grad_norm = jnp.array(0.0, dtype=loss.dtype)
  disc_end_grad_norm = jnp.array(0.0, dtype=loss.dtype)
  disc_mid_logit_abs_mean = jnp.array(0.0, dtype=loss.dtype)
  disc_end_logit_abs_mean = jnp.array(0.0, dtype=loss.dtype)
  disc_train_factor = _disc_update_factor(epoch, config)

  if use_mid:
    def disc_mid_loss_fn(disc_params):
      real_logits = _apply_discriminator(
          disc_mid_model, disc_params,
          jax.lax.stop_gradient(real_features['middle']),
          train=True, rng=disc_mid_real_dropout_rng)
      fake_logits = _apply_discriminator(
          disc_mid_model, disc_params,
          jax.lax.stop_gradient(fake_features['middle']),
          train=True, rng=disc_mid_fake_dropout_rng)
      return discriminator_hinge_loss(
          real_logits, fake_logits, config.disc_logit_penalty), (real_logits, fake_logits)

    disc_mid_grad_fn = jax.value_and_grad(disc_mid_loss_fn, has_aux=True)
    (disc_mid_loss, (mid_real_logits, mid_fake_logits)), disc_mid_grads = (
        disc_mid_grad_fn(disc_mid_state.params))
    disc_mid_grads = jax.lax.pmean(disc_mid_grads, axis_name='batch')
    disc_mid_grads = _scale_tree(disc_mid_grads, disc_train_factor)
    disc_mid_grad_norm = util_fns.global_norm(disc_mid_grads)
    disc_mid_state = disc_mid_state.apply_gradients(grads=disc_mid_grads)
    disc_mid_real_logits = jnp.mean(mid_real_logits)
    disc_mid_fake_logits = jnp.mean(mid_fake_logits)
    disc_mid_logit_abs_mean = 0.5 * (
        jnp.mean(jnp.abs(mid_real_logits)) + jnp.mean(jnp.abs(mid_fake_logits)))

  if use_end:
    def disc_end_loss_fn(disc_params):
      real_logits = _apply_discriminator(
          disc_end_model, disc_params,
          jax.lax.stop_gradient(real_features['end']),
          train=True, rng=disc_end_real_dropout_rng)
      fake_logits = _apply_discriminator(
          disc_end_model, disc_params,
          jax.lax.stop_gradient(fake_features['end']),
          train=True, rng=disc_end_fake_dropout_rng)
      return discriminator_hinge_loss(
          real_logits, fake_logits, config.disc_logit_penalty), (real_logits, fake_logits)

    disc_end_grad_fn = jax.value_and_grad(disc_end_loss_fn, has_aux=True)
    (disc_end_loss, (end_real_logits, end_fake_logits)), disc_end_grads = (
        disc_end_grad_fn(disc_end_state.params))
    disc_end_grads = jax.lax.pmean(disc_end_grads, axis_name='batch')
    disc_end_grads = _scale_tree(disc_end_grads, disc_train_factor)
    disc_end_grad_norm = util_fns.global_norm(disc_end_grads)
    disc_end_state = disc_end_state.apply_gradients(grads=disc_end_grads)
    disc_end_real_logits = jnp.mean(end_real_logits)
    disc_end_fake_logits = jnp.mean(end_fake_logits)
    disc_end_logit_abs_mean = 0.5 * (
        jnp.mean(jnp.abs(end_real_logits)) + jnp.mean(jnp.abs(end_fake_logits)))

  metrics = {
      'loss': jax.lax.pmean(loss, axis_name='batch'),
      'ardm_loss': jax.lax.pmean(ardm_loss, axis_name='batch'),
      'nelbo': jax.lax.pmean(-elbo_value, axis_name='batch'),
      'ce': jax.lax.pmean(-ce_value, axis_name='batch'),
      'nelbo_per_t_batch': jax.lax.all_gather(-elbo_per_t, axis_name='batch'),
      't_batch': jax.lax.all_gather(t, axis_name='batch'),
      'grad_norm': grad_norm,
      'guidance_progress': jax.lax.pmean(guidance_progress, axis_name='batch'),
      'disc_train_factor': jax.lax.pmean(disc_train_factor, axis_name='batch'),
      'aux_loss': jax.lax.pmean(aux_loss, axis_name='batch'),
      'aux_grad_cosine': jax.lax.pmean(aux_grad_cosine, axis_name='batch'),
      'aux_alignment_factor': jax.lax.pmean(
          aux_alignment_factor, axis_name='batch'),
      'ardm_grad_norm_raw': jax.lax.pmean(
          ardm_grad_norm_raw, axis_name='batch'),
      'aux_grad_norm_raw': jax.lax.pmean(
          aux_grad_norm_raw, axis_name='batch'),
      'mask_fraction_observed': jax.lax.pmean(
          feature_info['mask_fraction_observed'], axis_name='batch'),
  }
  if use_mid:
    metrics.update({
        'gen_mid_loss': jax.lax.pmean(gen_mid_loss, axis_name='batch'),
        'disc_mid_loss': jax.lax.pmean(disc_mid_loss, axis_name='batch'),
        'disc_mid_real_logits': jax.lax.pmean(disc_mid_real_logits, axis_name='batch'),
        'disc_mid_fake_logits': jax.lax.pmean(disc_mid_fake_logits, axis_name='batch'),
        'disc_weight_mid': jax.lax.pmean(disc_weight_mid, axis_name='batch'),
        'reason_mid_loss': jax.lax.pmean(reason_mid_loss, axis_name='batch'),
        'reason_weight_mid': jax.lax.pmean(reason_weight_mid, axis_name='batch'),
        'cf_mid_loss': jax.lax.pmean(cf_mid_loss, axis_name='batch'),
        'cf_mid_weight': jax.lax.pmean(cf_mid_weight, axis_name='batch'),
        'cf_mid_delta': jax.lax.pmean(cf_mid_delta, axis_name='batch'),
        'cf_mid_dropped_energy': jax.lax.pmean(
            cf_mid_dropped_energy, axis_name='batch'),
        'cf_mid_masked_ardm_loss': jax.lax.pmean(
            cf_mid_masked_ardm_loss, axis_name='batch'),
        'cf_mid_drop_fraction': jax.lax.pmean(
            cf_mid_drop_fraction, axis_name='batch'),
        'disc_mid_real_logits_std': jax.lax.pmean(jnp.std(mid_real_logits), axis_name='batch'),
        'disc_mid_fake_logits_std': jax.lax.pmean(jnp.std(mid_fake_logits), axis_name='batch'),
        'disc_mid_grad_norm': disc_mid_grad_norm,
        'disc_mid_logit_abs_mean': jax.lax.pmean(
            disc_mid_logit_abs_mean, axis_name='batch'),
        'fake_mid_feature_norm': jax.lax.pmean(
            mean_feature_norm(fake_features['middle']), axis_name='batch'),
        'real_mid_feature_norm': jax.lax.pmean(
            mean_feature_norm(real_features['middle']), axis_name='batch'),
        'mid_feature_cosine': jax.lax.pmean(
            mean_feature_cosine(fake_features['middle'], real_features['middle']),
            axis_name='batch'),
        'fake_mid_feature_mean': jax.lax.pmean(
            jnp.mean(fake_features['middle']), axis_name='batch'),
        'real_mid_feature_mean': jax.lax.pmean(
            jnp.mean(real_features['middle']), axis_name='batch'),
    })
  if use_end:
    metrics.update({
        'gen_end_loss': jax.lax.pmean(gen_end_loss, axis_name='batch'),
        'disc_end_loss': jax.lax.pmean(disc_end_loss, axis_name='batch'),
        'disc_end_real_logits': jax.lax.pmean(disc_end_real_logits, axis_name='batch'),
        'disc_end_fake_logits': jax.lax.pmean(disc_end_fake_logits, axis_name='batch'),
        'disc_weight_end': jax.lax.pmean(disc_weight_end, axis_name='batch'),
        'reason_end_loss': jax.lax.pmean(reason_end_loss, axis_name='batch'),
        'reason_weight_end': jax.lax.pmean(reason_weight_end, axis_name='batch'),
        'disc_end_real_logits_std': jax.lax.pmean(jnp.std(end_real_logits), axis_name='batch'),
        'disc_end_fake_logits_std': jax.lax.pmean(jnp.std(end_fake_logits), axis_name='batch'),
        'disc_end_grad_norm': disc_end_grad_norm,
        'disc_end_logit_abs_mean': jax.lax.pmean(
            disc_end_logit_abs_mean, axis_name='batch'),
        'fake_end_feature_norm': jax.lax.pmean(
            mean_feature_norm(fake_features['end']), axis_name='batch'),
        'real_end_feature_norm': jax.lax.pmean(
            mean_feature_norm(real_features['end']), axis_name='batch'),
        'end_feature_cosine': jax.lax.pmean(
            mean_feature_cosine(fake_features['end'], real_features['end']),
            axis_name='batch'),
        'fake_end_feature_mean': jax.lax.pmean(
            jnp.mean(fake_features['end']), axis_name='batch'),
        'real_end_feature_mean': jax.lax.pmean(
            jnp.mean(real_features['end']), axis_name='batch'),
    })
  return state, disc_mid_state, disc_end_state, metrics, rng_return


def eval_step(rng, batch, state, model):
  """Eval a single step."""
  logging.info('Eval step...')
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  elbo_value, _, ce_value, _ = model.elbo(
      rng, state.ema_params, batch['image'], train=False)
  metrics = {
      'nelbo': jax.lax.pmean(-elbo_value, axis_name='batch'),
      'ce': jax.lax.pmean(-ce_value, axis_name='batch')
  }
  return metrics, rng_return


def train_epoch(p_train_step, state, disc_mid_state, disc_end_state, train_ds,
                batch_size, epoch, rng, kl_tracker):
  """Train for a single epoch."""
  start_time = time.time()

  batch_metrics = []

  train_ds = util_fns.get_iterator(train_ds)
  with jax.profiler.StepTraceAnnotation('train', step_num=state.step):
    for batch in train_ds:
      state, disc_mid_state, disc_end_state, metrics, rng = p_train_step(
          rng, batch, state, disc_mid_state, disc_end_state, epoch)

      # UGARDM returns extra discriminator scalars. Copy each batch's metrics to
      # host immediately to avoid accumulating a large nested JAX object for a
      # single epoch-level device_get. This affects logging only, not gradients.
      metrics = jax.device_get(flax.jax_utils.unreplicate(metrics))
      batch_metrics.append(metrics)

  # This processes the loss per t, although two nested for-loops (counting the
  # one inside kl_tracker), it actually does not hurt timing performance
  # meaningfully.
  t_batches = [
      metrics['t_batch'].reshape(batch_size) for metrics in batch_metrics]
  nelbo_per_t_batches = [
      metrics['nelbo_per_t_batch'].reshape(batch_size)
      for metrics in batch_metrics]
  for t_batch, nelbo_per_t_batch in zip(t_batches, nelbo_per_t_batches):
    kl_tracker.update(t_batch, nelbo_per_t_batch)

  # Compute mean of metrics across each batch in epoch.
  epoch_metrics = {
      key: np.mean([metrics[key] for metrics in batch_metrics])
      for key in batch_metrics[0] if 'batch' not in key}

  message = f'Epoch took {time.time() - start_time:.1f} seconds.'
  logging.info(message)
  info_string = (
      f'train epoch: {epoch}, loss: {epoch_metrics["loss"]:.4f} '
      f'nelbo: {epoch_metrics["nelbo"]:.4f} ce: {epoch_metrics["ce"]:.4f}'
      )
  logging.info(info_string)

  return state, disc_mid_state, disc_end_state, epoch_metrics, rng


def eval_model(p_eval_step, rng, state, test_ds, epoch):
  """Eval for a single epoch."""
  start_time = time.time()
  batch_metrics = []

  test_ds = util_fns.get_iterator(test_ds)

  for batch in test_ds:
    metrics, rng = p_eval_step(rng, batch, state)

    # Better to leave metrics on device, and off-load after finishing epoch.
    batch_metrics.append(metrics)

  # Load to CPU.
  batch_metrics = jax.device_get(flax.jax_utils.unreplicate(batch_metrics))

  # Compute mean of metrics across each batch in epoch.
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics])
      for k in batch_metrics[0] if 'batch' not in k}

  nelbo = epoch_metrics_np['nelbo']
  message = f'Eval epoch took {time.time() - start_time:.1f} seconds.'
  logging.info(message)
  info_string = f'eval epoch: {epoch}, nelbo: {nelbo:.4f}'
  logging.info(info_string)

  return epoch_metrics_np, rng


# The axes that are broadcasted are the in- and output rng key ones, and the
# model, and the policy. The rng is the first arg, and the last return value.
@functools.partial(
    jax.pmap,
    static_broadcasted_argnums=(3,),
    in_axes=(None, 0, 0, None, None),
    out_axes=(0, None),
    axis_name='batch')
def eval_step_policy(rng, batch, state, model, policy):
  """Eval a single step."""
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  elbo_value, _, ce_value, _ = model.elbo_with_policy(
      rng, state.ema_params, batch['image'], policy=policy, train=False)
  metrics = {
      'nelbo': jax.lax.pmean(-elbo_value, axis_name='batch'),
      'ce': jax.lax.pmean(-ce_value, axis_name='batch')
  }
  return metrics, rng_return


def eval_policy(policy, rng, state, model, test_ds, epoch):
  """Eval for a single epoch."""
  batch_metrics = []

  policy = flax.jax_utils.unreplicate(flax.jax_utils.replicate(policy))

  # Function is recompiled for this specific policy.
  test_ds = util_fns.get_iterator(test_ds)
  for batch in test_ds:
    metrics, rng = eval_step_policy(rng, batch, state, model, policy)

    # Better to leave metrics on device, and off-load after finishing epoch.
    batch_metrics.append(metrics)

  # Load to CPU.
  batch_metrics = jax.device_get(flax.jax_utils.unreplicate(batch_metrics))
  # Compute mean of metrics across each batch in epoch.
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics])
      for k in batch_metrics[0] if 'batch' not in k}

  nelbo = epoch_metrics_np['nelbo']
  info_string = f'eval policy epoch: {epoch}, nelbo: {nelbo:.4f}'
  logging.info(info_string)

  return epoch_metrics_np


def log_standard_metrics(writer, train_metrics, eval_metrics, epoch):
  metric_dict = {
      'train_loss': train_metrics['loss'],
      'train_nelbo': train_metrics['nelbo'],
      'train_ce': train_metrics['ce'],
      'grad_norm': train_metrics['grad_norm'],
      'eval_nelbo': eval_metrics['nelbo'],
  }
  for key in ('ardm_loss', 'gen_mid_loss', 'gen_end_loss',
              'disc_mid_loss', 'disc_end_loss',
              'disc_mid_real_logits', 'disc_mid_fake_logits',
              'disc_end_real_logits', 'disc_end_fake_logits',
              'disc_weight_mid', 'disc_weight_end',
              'reason_mid_loss', 'reason_end_loss',
              'reason_weight_mid', 'reason_weight_end',
              'disc_mid_real_logits_std', 'disc_mid_fake_logits_std',
              'disc_end_real_logits_std', 'disc_end_fake_logits_std',
              'disc_mid_grad_norm', 'disc_end_grad_norm',
              'guidance_progress', 'disc_train_factor',
              'aux_loss', 'aux_grad_cosine', 'aux_alignment_factor',
              'ardm_grad_norm_raw', 'aux_grad_norm_raw',
              'mask_fraction_observed',
              'cf_mid_loss', 'cf_mid_weight', 'cf_mid_delta',
              'cf_mid_dropped_energy', 'cf_mid_masked_ardm_loss',
              'cf_mid_drop_fraction',
              'disc_mid_logit_abs_mean', 'disc_end_logit_abs_mean',
              'fake_mid_feature_norm', 'real_mid_feature_norm',
              'fake_end_feature_norm', 'real_end_feature_norm',
              'mid_feature_cosine', 'end_feature_cosine',
              'fake_mid_feature_mean', 'real_mid_feature_mean',
              'fake_end_feature_mean', 'real_end_feature_mean'):
    if key in train_metrics:
      metric_dict[key] = train_metrics[key]
  writer.write_scalars(epoch, metric_dict)


def extensive_eval(config, test_rng, writer,
                   output_path, model, state, kl_history, test_ds, epoch):
  """This function combines all extra eval benchmarks we want to run."""
  # Eval settings.
  is_first_host = jax.process_index() == 0
  max_num_steps = 25000
  max_num_steps_for_policy = 25000
  num_samples = config.num_samples
  n_rows = int(np.sqrt(num_samples))

  return_rng, rng1, rng2, rng3, rng4, rng5 = jax.random.split(test_rng, 6)

  # Plot loss components over time.
  if jax.process_index() == 0:
    fname = f'loss_t_{epoch}.png'
    filename = os.path.join(output_path, 'loss_plots', fname)
    util_fns.plot_loss_components(kl_history, filename, model.num_stages)

  # Sample from the model.
  if model.num_steps < max_num_steps:
    start = time.time()
    chain = model.sample(rng1, state.ema_params, num_samples)
    msg = f'Sampling took {time.time() - start:.2f} seconds'
    logging.info(msg)

    if is_first_host:
      filename = os.path.join(output_path, 'samples', f'chain_epoch{epoch}.gif')
      util_fns.save_chain_to_gif(chain, filename, n_rows)
      util_fns.plot_batch_images(chain[-1], n_rows, config.num_classes)
      writer.write_images(epoch, {'samples': chain[-1]})

      if epoch % config.eval_gen_every == 0:
        metric_dict = metric_logging.run_generation_metrics(
          final_images=chain[-1],
          epoch=epoch,
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
          writer.write_scalars(epoch, metric_dict)

    del chain

  # Validate and sample using naive policy.
  if model.policy_support:
    nelbo_policy_naive = eval_policy(model.get_naive_policy(), rng2,
                                     state, model, test_ds,
                                     epoch)['nelbo']
    naive_dict = {'eval_nelbo_policy_naive': nelbo_policy_naive}
    chain_naive = model.sample_with_naive_policy(
        rng3, state.ema_params, num_samples)
    if is_first_host:
      writer.write_scalars(epoch, naive_dict)
      filename = os.path.join(output_path, 'samples_naive',
                              f'chain_epoch_naive_{epoch}.gif')
      util_fns.save_chain_to_gif(
          chain_naive, filename, n_rows)
      util_fns.plot_batch_images(chain_naive[-1], n_rows, config.num_classes)

    del chain_naive

  # Val optimal policies.
  if model.policy_support and model.num_steps < max_num_steps_for_policy:
    # Check 25, 50 & 100 steps, just because they are interesting to see.
    budgets = [50, 100]

    # Compute policies and costs.
    start = time.time()
    policies, costs = model.compute_policies_and_costs(kl_history[-1], budgets)
    msg = f'Computing policy mats took {time.time() - start:.2f} secs'
    logging.info(msg)

    # Evaluate policy for budget 50.
    nelbo_policy_50 = eval_policy(policies[0], rng4, state, model,
                                  test_ds, epoch)['nelbo']
    metric_dict = {'eval_nelbo_policy_50': nelbo_policy_50}
    budget_results_train = {
        f'train_nelbo_steps_{b}': c
        for b, c in zip(budgets, costs)
    }
    metric_dict.update(budget_results_train)
    if jax.process_index() == 0:
      writer.write_scalars(epoch, metric_dict)

    # Sample with lowest policy.
    chain_policy = model.sample_with_policy(
        rng5, state.ema_params, num_samples, policies[0])

    if jax.process_index() == 0:
      filename = os.path.join(output_path, 'samples_policy',
                              f'chain_epoch_policy_{epoch}.gif')
      util_fns.save_chain_to_gif(
          chain_policy, filename, n_rows)
      util_fns.plot_batch_images(chain_policy[-1], n_rows, config.num_classes)

    del chain_policy, policies, costs
  return return_rng


def model_setup(init_rng, config):
  """Sets up the model and initializes params."""
  def get_architecture(num_input_classes, n_output_channels, num_steps):
    net = unet.UNet(
        num_classes=num_input_classes,
        ch=config.architecture.n_channels,
        out_ch=n_output_channels,
        ch_mult=config.architecture.ch_mult,
        num_res_blocks=config.architecture.num_res_blocks,
        full_attn_resolutions=config.architecture.attn_resolutions,
        num_heads=config.architecture.num_heads,
        dropout=config.architecture.dropout,
        max_time=float(num_steps))
    return net

  if config.model == 'ao_arm':
    model = ao_arm.ArbitraryOrderARM.create(
        config, get_architecture, absorbing_state=config.num_classes // 2)
  elif config.model == 'bit_ao':
    model = bit_ao.BitUpscaleAutoregressiveDiffusion.create(
        config, get_architecture)
  else:
    raise ValueError

  tmp_x, tmp_t = (jnp.ones([1, *config.data_shape], dtype=jnp.int32),
                  jnp.ones([1]))

  @functools.partial(jax.jit, backend='cpu')
  def init():
    return model.init_architecture(init_rng, tmp_x, tmp_t)

  logging.info('Initializing neural network')
  variables = init()
  return model, variables


def _prepare_run_output_dirs(config, work_dir):
  """Resolve per-run evaluation output dirs while sharing the real dataset.

  real_image_dir intentionally stays shared, e.g.:
    /home/jordan/runs/ardm/real_cifar10

  generated_eval_dir and metrics_dir are local to the active --work_unit_dir:
    <work_dir>/generated_eval
    <work_dir>/metrics
  """
  config.run_name = os.path.basename(os.path.normpath(work_dir))

  generated_eval_dir = getattr(config, 'generated_eval_dir', 'generated_eval')
  metrics_dir = getattr(config, 'metrics_dir', 'metrics')

  if not os.path.isabs(generated_eval_dir):
    generated_eval_dir = os.path.join(work_dir, generated_eval_dir)
  if not os.path.isabs(metrics_dir):
    metrics_dir = os.path.join(work_dir, metrics_dir)

  config.generated_eval_dir = generated_eval_dir
  config.metrics_dir = metrics_dir

  tf.io.gfile.makedirs(config.generated_eval_dir)
  tf.io.gfile.makedirs(config.metrics_dir)

  if not tf.io.gfile.exists(config.real_image_dir):
    logging.warning(
        'Shared real_image_dir does not exist yet: %s. Export real CIFAR-10 '
        'once to this path before computing FID/SSIM.', config.real_image_dir)

  logging.info('Run name: %s', config.run_name)
  logging.info('Shared real image dir: %s', config.real_image_dir)
  logging.info('Generated eval dir: %s', config.generated_eval_dir)
  logging.info('Metrics dir: %s', config.metrics_dir)


def train_and_evaluate(config,
                       work_dir, try_checkpoint=True):
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    work_dir: Directory where the tensorboard summaries are written to.
    try_checkpoint: Should try to load checkpoint (usually enabled, practical
        for debugging purposes to disable).

  Returns:
    The train state (which includes the `.params`).
  """
  # Init rng key.
  msg = f'Running with seed {config.seed}.'
  logging.info(msg)
  _sanitize_guidance_config(config)
  rng = jax.random.PRNGKey(config.seed)
  data_rng, rng = jax.random.split(rng)
  is_first_host = jax.process_index() == 0

  train_ds, test_ds, shape, num_classes = datasets.get_dataset(config, data_rng)

  _prepare_run_output_dirs(config, work_dir)

  # config.mask_shape = mask_shape
  config.data_shape = shape
  config.num_classes = num_classes

  writer = metric_writers.create_default_writer(
      work_dir, just_logging=jax.process_index() > 0)
  rng, init_rng = jax.random.split(rng)

  # Create output directory for saving samples.
  output_path = work_dir
  tf.io.gfile.makedirs(output_path)

  model, variables = model_setup(init_rng, config)

  rng, disc_mid_init_rng, disc_end_init_rng = jax.random.split(rng, 3)
  disc_mid_model = discriminator.FeatureDiscriminator(
      hidden_dim=config.disc_hidden_dim, dropout=config.disc_dropout,
      num_layers=config.disc_num_layers,
      normalize_input=getattr(config, 'disc_normalize_input', True))
  disc_end_model = discriminator.FeatureDiscriminator(
      hidden_dim=config.disc_hidden_dim, dropout=config.disc_dropout,
      num_layers=config.disc_num_layers,
      normalize_input=getattr(config, 'disc_normalize_input', True))

  image_resolution = config.data_shape[0]
  mid_resolution = image_resolution // (2 ** (len(config.architecture.ch_mult) - 1))
  mid_channels = config.architecture.n_channels * config.architecture.ch_mult[-1]
  end_channels = config.architecture.n_channels * config.architecture.ch_mult[0]
  dummy_mid_features = jnp.ones(
      (1, mid_resolution, mid_resolution, mid_channels), dtype=jnp.float32)
  dummy_end_features = jnp.ones(
      (1, image_resolution, image_resolution, end_channels), dtype=jnp.float32)
  disc_mid_variables = disc_mid_model.init(
      disc_mid_init_rng, dummy_mid_features, train=False)
  disc_end_variables = disc_end_model.init(
      disc_end_init_rng, dummy_end_features, train=False)
  disc_tx = optax.adam(
      learning_rate=config.disc_learning_rate,
      b1=config.disc_beta1,
      b2=config.disc_beta2)
  disc_mid_state = train_state.TrainState.create(
      apply_fn=disc_mid_model.apply,
      params=disc_mid_variables['params'],
      tx=disc_tx)
  disc_end_state = train_state.TrainState.create(
      apply_fn=disc_end_model.apply,
      params=disc_end_variables['params'],
      tx=disc_tx)

  # From now on we want different rng across hosts:
  rng = jax.random.fold_in(rng, jax.process_index())

  tx = optax.adam(
      config.learning_rate, b1=0.9, b2=config.beta2, eps=1e-08, eps_root=0.0)
  state = custom_train_state.TrainState.create(
      params=variables['params'], tx=tx)

  if try_checkpoint:
    state, start_epoch = checkpoint.restore_from_path(work_dir, state)
    if _uses_mid(config):
      disc_mid_state, _ = checkpoint.restore_from_path(
          work_dir, disc_mid_state, prefix='disc_mid_ckpt_')
    if _uses_end(config):
      disc_end_state, _ = checkpoint.restore_from_path(
          work_dir, disc_end_state, prefix='disc_end_ckpt_')
    if start_epoch is None:
      start_epoch = 1
  else:
    # For debugging we start at zero, so we immediately do detailed eval.
    start_epoch = 0

  if is_first_host and start_epoch == 1:
    config_dict = dict(config)
    writer.write_hparams(config_dict)

  if is_first_host and start_epoch in (0, 1):
    # Dump config file to work dir for easy model loading.
    config_path = os.path.join(work_dir, 'config')
    with tf.io.gfile.GFile(config_path, 'wb') as fp:
      pickle.dump(config, fp)

  test_rng, train_rng = jax.random.split(rng)

  kl_tracker_train = util_fns.KLTracker(num_steps=model.num_steps)
  kl_history = []

  p_train_step = jax.pmap(
      functools.partial(train_step, model=model, disc_mid_model=disc_mid_model,
                        disc_end_model=disc_end_model, config=config),
      axis_name='batch',
      in_axes=(None, 0, 0, 0, 0, None),
      out_axes=(0, 0, 0, 0, None),
      donate_argnums=(2, 3, 4))

  # The only axes that are broadcasted are the in- and output rng key ones. The
  # rng is the first arg, and the last return value.
  p_eval_step = jax.pmap(
      functools.partial(eval_step, model=model),
      axis_name='batch',
      in_axes=(None, 0, 0),
      out_axes=(0, None))

  # Replicate state.
  state = flax.jax_utils.replicate(state)
  disc_mid_state = flax.jax_utils.replicate(disc_mid_state)
  disc_end_state = flax.jax_utils.replicate(disc_end_state)

  with metric_writers.ensure_flushes(writer):
    for epoch in range(start_epoch, config.num_epochs + 1):
      # Train part.
      state, disc_mid_state, disc_end_state, train_metrics, train_rng = train_epoch(
          p_train_step, state, disc_mid_state, disc_end_state, train_ds,
          config.batch_size, epoch, train_rng, kl_tracker_train)

      # Val part.
      eval_metrics, test_rng = eval_model(p_eval_step, test_rng, state,
                                          test_ds, epoch)

      # Metric logging.
      if is_first_host:
        log_standard_metrics(writer, train_metrics, eval_metrics, epoch)

      kl_values = kl_tracker_train.get_kl_per_t()
      kl_history.append(np.array(kl_values))

      # Prune to avoid too much memory consumption.
      kl_history = kl_history[-50:]

      if epoch % config.detailed_eval_every == 0:
        if is_first_host:
          loss_components_path = os.path.join(work_dir, 'loss_components')
          with tf.io.gfile.GFile(loss_components_path, 'wb') as fp:
            pickle.dump(kl_history[-1], fp)

        test_rng = extensive_eval(config, test_rng, writer, output_path, model,
                                  state, kl_history, test_ds, epoch)

      # Save to checkpoint.
      if is_first_host and epoch % config.save_every == 0:
        # Save to epoch + 1 since current epoch has just been completed.
        logging.info('saving checkpoint')
        checkpoint.save_checkpoint(
            work_dir, state=flax.jax_utils.unreplicate(state), step=epoch + 1,
            keep=10)
        if _uses_mid(config):
          checkpoint.save_checkpoint(
              work_dir, state=flax.jax_utils.unreplicate(disc_mid_state),
              step=epoch + 1, keep=10, prefix='disc_mid_ckpt_')
        if _uses_end(config):
          checkpoint.save_checkpoint(
              work_dir, state=flax.jax_utils.unreplicate(disc_end_state),
              step=epoch + 1, keep=10, prefix='disc_end_ckpt_')
        logging.info('finished saving checkpoint')

    return state
