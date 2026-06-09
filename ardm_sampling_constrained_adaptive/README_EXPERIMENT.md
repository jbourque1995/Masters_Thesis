# ==============================================================================
# Modifications made by Jordan Bourque for master thesis research.
# University of New Brunswick, 2026.
# Purpose: Guidance and Scheduling ARDM experimentation and evaluation.
# ==============================================================================

# ARDM Constrained Adaptive Timestep Sampling

This package implements a fixed-architecture ARDM training-sampler experiment.
It does not increase data, epochs, UNet width/depth, parameter count, or tensor
shape.

## Main idea

Earlier adaptive runs improved the timestep policy mechanically, but stronger
adaptation drifted toward lower-observed/harder mask states. This package adds
`constrained_adaptive_t`, which preserves the original uniform observed-fraction
/ mask-difficulty marginal while adapting inside comparable observed-fraction
buckets using ARDM's own KL/NELBO history.

In plain terms: the sampler is allowed to ask, "within this mask difficulty
level, which timesteps/stages need more training?" It is not allowed to improve
its score by simply training on harder masks more often.

## Main run

```bash
cd ~/projects
source ~/venvs/ardm/bin/activate

RUN_NAME="run_ardm_constrained_adaptive_t_200"
RUN_DIR="/home/jordan/runs/ardm/ardm_constrained_adaptive/${RUN_NAME}"

mkdir -p "$RUN_DIR"

python -m ardm_sampling_constrained_adaptive.experiments.images.main \
  --config=ardm_sampling_constrained_adaptive/experiments/images/config.py \
  --work_unit_dir="$RUN_DIR" \
  --config.num_epochs=200 \
  --config.elbo_mode=constrained_adaptive_t \
  --config.adaptive_start_epoch=10 \
  --config.adaptive_alpha=0.75 \
  --config.adaptive_uniform_mix=0.15 \
  --config.adaptive_max_prob_ratio=12.0 \
  --config.constrained_num_mask_buckets=64
```

## Watch first

- `sampled_observed_fraction` should remain near the baseline/uniform value, roughly 0.5.
- `sampled_observed_fraction_std` should remain close to uniform, roughly 0.288.
- `constrained_observed_mass_l1` should stay near 0.
- `adaptive_policy_max_over_uniform` should move above 1 without collapse.
- `train_nelbo`, `eval_nelbo`, `eval_nelbo_policy_naive`, and `eval_fid` decide whether it helps.
