# coding=utf-8
# ==============================================================================
# Modifications made by Jordan Bourque for master thesis research.
# University of New Brunswick, 2026.
# Purpose: Guidance and Scheduling ARDM experimentation and evaluation.
# ==============================================================================

"""Spatial feature-level discriminator heads for UGARDM experiments.

These heads score UNet feature maps without global-average pooling.  The
critic receives spatial feature maps and returns a patch/logit map `(B, H, W)`.

The input is normalized before discrimination so the critic is pushed toward
structural/spatial differences rather than trivial feature-magnitude shortcuts.
"""

from flax import linen as nn
import jax.numpy as jnp


class FeatureDiscriminator(nn.Module):
  """Small spatial critic for internal UNet representations."""

  hidden_dim: int = 128
  dropout: float = 0.0
  num_layers: int = 3
  normalize_input: bool = True
  eps: float = 1e-6

  @nn.compact
  def __call__(self, features, train=True):
    x = features.astype(jnp.float32)
    if x.ndim == 2:
      x = x[:, None, None, :]
    elif x.ndim != 4:
      raise ValueError(
          f'FeatureDiscriminator expected 2D or 4D input, got {x.shape}')

    if self.normalize_input:
      # LayerNorm removes simple mean/scale differences across channels.
      x = nn.LayerNorm(name='input_layer_norm')(x)
      # Channel-wise L2 normalization further discourages magnitude shortcuts.
      x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + self.eps)

    for i in range(self.num_layers):
      x = nn.Conv(
          features=self.hidden_dim,
          kernel_size=(3, 3),
          padding='SAME',
          name=f'spatial_conv_{i}')(x)
      x = nn.LayerNorm(name=f'spatial_norm_{i}')(x)
      x = nn.leaky_relu(x, negative_slope=0.2)
      if self.dropout > 0.0:
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

    logits = nn.Conv(
        features=1, kernel_size=(1, 1), padding='SAME', name='patch_logits')(x)
    return jnp.squeeze(logits, axis=-1)
