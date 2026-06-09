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

"""Small discriminator architecture for guided ARDM image training."""

from flax import linen as nn


class ConvDiscriminator(nn.Module):
  """Small convolutional discriminator for 32x32 image inputs."""
  base_channels: int = 64

  @nn.compact
  def __call__(self, x, train=True):
    del train

    h = nn.Conv(
        features=self.base_channels,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='SAME',
        name='conv_0')(x)
    h = nn.leaky_relu(h, negative_slope=0.2)

    h = nn.Conv(
        features=self.base_channels * 2,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='SAME',
        name='conv_1')(h)
    h = nn.leaky_relu(h, negative_slope=0.2)

    h = nn.Conv(
        features=self.base_channels * 4,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='SAME',
        name='conv_2')(h)
    h = nn.leaky_relu(h, negative_slope=0.2)

    h = h.reshape((h.shape[0], -1))
    h = nn.Dense(features=1, name='logits')(h)
    return h.squeeze(-1)
