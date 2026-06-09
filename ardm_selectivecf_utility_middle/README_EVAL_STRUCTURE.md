# ==============================================================================
# Modifications made by Jordan Bourque for master thesis research.
# University of New Brunswick, 2026.
# Purpose: Guidance and Scheduling ARDM experimentation and evaluation.
# ==============================================================================

# UGARDM evaluation-folder structure update

This experiment uses `config.guidance_mode = 'unet_middle'`.

The training loop remains on the normal detailed-eval trigger:

```python
epoch == 15 or epoch % config.detailed_eval_every == 0
```

The shared real CIFAR-10 folder is:

```text
/home/jordan/runs/ardm/real_cifar10
```

Each run writes its own generated evaluation images and metric JSON files inside its active `--work_unit_dir`:

```text
<RUN_DIR>/generated_eval/
<RUN_DIR>/metrics/
```

A backfill script is included at:

```text
experiments/images/backfill_generation_metrics.py
```

Example use after installing/copying this experiment as `autoregressive_diffusion`:

```bash
python -m ardm_selectivecf_utility_middle.experiments.images.backfill_generation_metrics \
  --config=ardm_selectivecf_utility_middle/experiments/images/config.py \
  --work_unit_dir="$RUN_DIR" \
  --epochs=50,100,150,200,250,300,350,400,450 \
  --num_samples_override=64
```
