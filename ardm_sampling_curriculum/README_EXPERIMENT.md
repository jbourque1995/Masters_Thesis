# ==============================================================================
# Modifications made by Jordan Bourque for master thesis research.
# University of New Brunswick, 2026.
# Purpose: Guidance and Scheduling ARDM experimentation and evaluation.
# ==============================================================================

# Curriculum-to-uniform observed-mask sampling

Top-level Python package: `ardm_sampling_curriculum`

This package is based on the fixed ARDM architecture and changes only the training subproblem sampler. It does **not** increase training data, UNet width/depth, tensor shapes, or trainable model capacity.

Main config value:

```python
config.elbo_mode = 'curriculum_t'
```

New TensorBoard metrics include sampled timestep/observed-fraction diagnostics plus scheduler-specific metrics for adaptive/curriculum runs.
