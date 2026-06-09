#!/bin/bash
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

# Runs the main image experiment.
python -m ardm_basicdisc_middle_end.experiments.images.main --work_unit_dir=results/images --config ardm_basicdisc_middle_end/experiments/images/config.py \
  --config.num_epochs 1 --config.architecture.n_channels 64 --config.architecture.num_res_blocks 0

# Runs the main language experiment.
python -m ardm_basicdisc_middle_end.experiments.language.main --work_unit_dir=results/language --config ardm_basicdisc_middle_end/experiments/language/configs/default.py \
  --config.num_train_steps 1 --config.num_layers 0

# Runs the main audio experiment.
python -m ardm_basicdisc_middle_end.experiments.audio.main --work_unit_dir=results/audio --config ardm_basicdisc_middle_end/experiments/audio/configs/sc09.py \
  --config.num_train_steps 1 --config.arch.config.num_blocks 1 --executable_name train_and_evaluate


