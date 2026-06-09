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

"""Strict generation metric helpers for image experiments.

Metrics are computed only from generated images saved by the current evaluation
call under `<generated_eval_dir>/epoch_<epoch>/`. The helpers do not delete
existing generated images. If image files already exist in the epoch directory
that are not part of the current generated set, metric computation raises an
error instead of silently mixing old samples into FID/IS/CLIP/SSIM.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf

try:
  from torch_fidelity import calculate_metrics
  _HAS_TORCH_FIDELITY = True
except ImportError:
  calculate_metrics = None
  _HAS_TORCH_FIDELITY = False

try:
  import torch
  import open_clip
  _HAS_CLIP = True
except ImportError:
  torch = None
  open_clip = None
  _HAS_CLIP = False

_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def _ensure_dir(path):
  tf.io.gfile.makedirs(path)


def _is_image_path(path):
  return str(path).lower().endswith(_IMAGE_EXTENSIONS)


def _immediate_image_paths(directory):
  """Return image files directly inside directory; no recursion."""
  base = Path(directory)
  if not base.exists():
    return []
  return sorted([p for p in base.iterdir() if p.is_file() and _is_image_path(p)])


def _to_numpy(images):
  if hasattr(images, '__array__'):
    return np.asarray(images)
  return np.array(images)


def _normalize_to_uint8(images):
  """Convert image tensor to uint8 RGB images."""
  images = _to_numpy(images)

  if images.ndim != 4:
    raise ValueError(f'Expected [N,H,W,C] images, got shape {images.shape}')

  if images.shape[-1] == 1:
    images = np.repeat(images, 3, axis=-1)

  if images.shape[-1] != 3:
    raise ValueError(f'Expected 1 or 3 channels, got shape {images.shape}')

  images = images.astype(np.float32)
  min_val = images.min()
  max_val = images.max()

  if min_val >= 0.0 and max_val <= 1.0:
    images = images * 255.0
  elif min_val >= -1.0 and max_val <= 1.0:
    images = (images + 1.0) * 127.5

  return np.clip(images, 0, 255).astype(np.uint8)


def save_generated_images(final_images, out_dir, prefix='gen'):
  """Save generated images to PNG files without deleting the directory."""
  _ensure_dir(out_dir)
  images = _normalize_to_uint8(final_images)

  saved_paths = []
  for i, img in enumerate(images):
    out_path = os.path.join(out_dir, f'{prefix}_{i:05d}.png')
    with tf.io.gfile.GFile(out_path, 'wb') as fp:
      Image.fromarray(img).save(fp, format='PNG')
    saved_paths.append(out_path)
  return saved_paths


def _validate_generated_metric_dir(epoch_gen_dir, saved_paths):
  """Ensure generated metrics use only this evaluation's saved images.

  This does not delete anything. It fails fast if old image files would be mixed
  into metrics. Non-image files such as Windows Zone.Identifier artifacts are
  ignored by this validation, but the metric directory itself should normally be
  produced by this script and contain only PNGs.
  """
  saved = {str(Path(p).resolve()) for p in saved_paths}
  present = {str(p.resolve()) for p in _immediate_image_paths(epoch_gen_dir)}
  extra = sorted(present - saved)
  missing = sorted(saved - present)
  if extra or missing:
    raise ValueError(
        'Generated metric directory is not a clean match for this evaluation. '
        f'epoch_gen_dir={epoch_gen_dir}, extra_image_files={extra[:10]}, '
        f'missing_image_files={missing[:10]}. This protects FID/IS/CLIP/SSIM '
        'from silently mixing old samples. Move old images elsewhere or rerun '
        'with the same sample count to overwrite them.')


def _validate_real_image_dir(real_dir):
  real_paths = _immediate_image_paths(real_dir)
  if not real_paths:
    raise ValueError(
        f'No real reference images found in real_image_dir={real_dir}. Export '
        'CIFAR-10 real images before computing FID/SSIM.')
  return real_paths


def compute_fid_and_is(generated_dir, real_dir, cuda=True):
  """Compute FID and Inception Score using torch-fidelity."""
  _validate_real_image_dir(real_dir)
  if not _immediate_image_paths(generated_dir):
    raise ValueError(f'No generated images found in generated_dir={generated_dir}')

  if not _HAS_TORCH_FIDELITY:
    return {
        'eval_fid': np.nan,
        'eval_is_mean': np.nan,
        'eval_is_std': np.nan,
    }

  metrics = calculate_metrics(
      input1=str(generated_dir),
      input2=str(real_dir),
      cuda=cuda,
      fid=True,
      isc=True,
      verbose=False,
  )

  return {
      'eval_fid': float(metrics['frechet_inception_distance']),
      'eval_is_mean': float(metrics['inception_score_mean']),
      'eval_is_std': float(metrics['inception_score_std']),
  }


def compute_clip_score(image_paths, prompt):
  """Compute average CLIP score for explicit generated image paths."""
  image_paths = [p for p in image_paths if _is_image_path(p)]
  if not _HAS_CLIP or not image_paths:
    return np.nan

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model, _, preprocess = open_clip.create_model_and_transforms(
      'ViT-B-32', pretrained='openai')
  tokenizer = open_clip.get_tokenizer('ViT-B-32')
  model = model.to(device)
  model.eval()

  text = tokenizer([prompt]).to(device)
  scores = []

  with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for path in image_paths:
      image = preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
      image_features = model.encode_image(image)
      image_features = image_features / image_features.norm(dim=-1, keepdim=True)
      scores.append((image_features @ text_features.T).item())

  if not scores:
    return np.nan
  return float(np.mean(scores))


def compute_ssim_against_real(generated_paths, real_dir, max_images=100):
  """Compute mean SSIM by pairing current generated paths with real images."""
  gen_paths = [Path(p) for p in generated_paths if _is_image_path(p)][:max_images]
  real_paths = _validate_real_image_dir(real_dir)[:max_images]

  if not gen_paths or not real_paths:
    return np.nan

  scores = []
  for gp, rp in zip(gen_paths, real_paths):
    gen_img = np.array(Image.open(gp).convert('RGB'))
    real_img = np.array(Image.open(rp).convert('RGB'))
    scores.append(ssim(gen_img, real_img, channel_axis=2, data_range=255))

  if not scores:
    return np.nan
  return float(np.mean(scores))


def _write_json(path, data):
  with tf.io.gfile.GFile(path, 'w') as fp:
    fp.write(json.dumps(data, indent=2, sort_keys=True))


def run_generation_metrics(
    final_images,
    epoch,
    real_image_dir,
    generated_eval_dir,
    metrics_dir,
    compute_fid=True,
    compute_is=True,
    compute_clip=False,
    compute_ssim=False,
    clip_prompt='a CIFAR-10 image',
    num_clip_images=100,
    num_ssim_images=100,
):
  """Save current generated images and compute FID/IS/CLIP/SSIM.

  Generated images are saved to `<generated_eval_dir>/epoch_<epoch>/` and all
  generated-side metrics are restricted to that exact epoch directory. Existing
  images are overwritten by filename, but no files are deleted.
  """
  epoch_gen_dir = os.path.join(generated_eval_dir, f'epoch_{epoch}')
  _ensure_dir(epoch_gen_dir)
  _ensure_dir(metrics_dir)

  saved_paths = save_generated_images(final_images, epoch_gen_dir, prefix='gen')
  _validate_generated_metric_dir(epoch_gen_dir, saved_paths)

  results = {}

  if compute_fid or compute_is:
    fid_is = compute_fid_and_is(
        generated_dir=epoch_gen_dir,
        real_dir=real_image_dir,
        cuda=True)
    if compute_fid:
      results['eval_fid'] = fid_is['eval_fid']
    if compute_is:
      results['eval_is_mean'] = fid_is['eval_is_mean']
      results['eval_is_std'] = fid_is['eval_is_std']

  if compute_clip:
    clip_paths = saved_paths[:num_clip_images]
    results['eval_clip'] = compute_clip_score(clip_paths, clip_prompt)

  if compute_ssim:
    results['eval_ssim'] = compute_ssim_against_real(
        generated_paths=saved_paths,
        real_dir=real_image_dir,
        max_images=num_ssim_images)

  json_path = os.path.join(metrics_dir, f'epoch_{epoch}.json')
  payload = {
      'epoch': int(epoch),
      'generated_dir': epoch_gen_dir,
      'real_image_dir': real_image_dir,
      'num_generated_images': len(saved_paths),
      'metrics': results,
  }
  _write_json(json_path, payload)

  return results
