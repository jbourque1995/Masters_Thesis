# coding=utf-8
# ==============================================================================
# Modifications made by Jordan Bourque for master thesis research.
# University of New Brunswick, 2026.
# Purpose: Guidance and Scheduling ARDM experimentation and evaluation.
# ==============================================================================

"""Feature-level discriminator heads for UGARDM experiments.

These heads are intentionally small. They score pooled UNet feature maps rather
than final generated images, so they can be attached at the middle bottleneck,
at the final pre-output representation, or at both points.
"""

from flax import linen as nn
import jax.numpy as jnp


class FeatureDiscriminator(nn.Module):
  """MLP discriminator for UNet feature tensors.

  Input may be a 4-D feature map `(B, H, W, C)` or a 2-D feature tensor
  `(B, C)`. Spatial maps are global-average pooled before classification.
  """

  hidden_dim: int = 128
  dropout: float = 0.0

  @nn.compact
  def __call__(self, features, train=True):
    x = features.astype(jnp.float32)
    if x.ndim == 4:
      x = jnp.mean(x, axis=(1, 2))
    elif x.ndim != 2:
      raise ValueError(f'FeatureDiscriminator expected 2D or 4D input, got {x.shape}')

    x = nn.Dense(self.hidden_dim, name='dense_1')(x)
    x = nn.leaky_relu(x, negative_slope=0.2)
    if self.dropout > 0.0:
      x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
    x = nn.Dense(self.hidden_dim, name='dense_2')(x)
    x = nn.leaky_relu(x, negative_slope=0.2)
    x = nn.Dense(1, name='logits')(x)
    return jnp.squeeze(x, axis=-1)
