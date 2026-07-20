cd ~/projects || exit 1
source ~/venvs/ardm/bin/activate

BASE_RUN_DIR="/home/jordan/runs/ardm"
find "$BASE_RUN_DIR" -name '*:Zone.Identifier' -type f -delete 2>/dev/null || true
set +e

run_exp () {
  RUN_NAME="$1"
  shift
  RUN_DIR="${BASE_RUN_DIR}/${RUN_NAME}"
  mkdir -p "$RUN_DIR"

  echo ""
  echo "============================================================"
  echo "STARTING: ${RUN_NAME}"
  echo "RUN_DIR:  ${RUN_DIR}"
  echo "TIME:     $(date)"
  echo "============================================================"
  echo ""

  "$@"
  EXIT_CODE=$?

  echo ""
  echo "============================================================"
  echo "FINISHED: ${RUN_NAME}"
  echo "EXIT CODE: ${EXIT_CODE}"
  echo "TIME:      $(date)"
  echo "============================================================"
  echo ""
}


run_exp "baseline_ardm_base" \
python -m ardm_baseline.experiments.images.main \
  --config=ardm_baseline/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/baseline_ardm_base" \
  --config.num_epochs=200

run_exp "guidance_basicdisc_middle_dw1e-4" \
python -m ardm_basicdisc_middle.experiments.images.main \
  --config=ardm_basicdisc_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_basicdisc_middle_dw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.disc_weight_mid=0.0001 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_basicdisc_middle_dw5e-4" \
python -m ardm_basicdisc_middle.experiments.images.main \
  --config=ardm_basicdisc_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_basicdisc_middle_dw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.disc_weight_mid=0.0005 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_basicdisc_middle_dw2e-3" \
python -m ardm_basicdisc_middle.experiments.images.main \
  --config=ardm_basicdisc_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_basicdisc_middle_dw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.disc_weight_mid=0.002 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_basicdisc_end_dw1e-4" \
python -m ardm_basicdisc_end.experiments.images.main \
  --config=ardm_basicdisc_end/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_basicdisc_end_dw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_end \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0001 \
  --config.disc_dropout=0.0

run_exp "guidance_basicdisc_end_dw5e-4" \
python -m ardm_basicdisc_end.experiments.images.main \
  --config=ardm_basicdisc_end/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_basicdisc_end_dw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_end \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0005 \
  --config.disc_dropout=0.0

run_exp "guidance_basicdisc_end_dw2e-3" \
python -m ardm_basicdisc_end.experiments.images.main \
  --config=ardm_basicdisc_end/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_basicdisc_end_dw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_end \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.002 \
  --config.disc_dropout=0.0

run_exp "guidance_basicdisc_middle_end_dw1e-4" \
python -m ardm_basicdisc_middle_end.experiments.images.main \
  --config=ardm_basicdisc_middle_end/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_basicdisc_middle_end_dw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.disc_weight_mid=0.0001 \
  --config.disc_weight_end=0.0001 \
  --config.disc_dropout=0.0

run_exp "guidance_basicdisc_middle_end_dw5e-4" \
python -m ardm_basicdisc_middle_end.experiments.images.main \
  --config=ardm_basicdisc_middle_end/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_basicdisc_middle_end_dw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.disc_weight_mid=0.0005 \
  --config.disc_weight_end=0.0005 \
  --config.disc_dropout=0.0

run_exp "guidance_basicdisc_middle_end_dw2e-3" \
python -m ardm_basicdisc_middle_end.experiments.images.main \
  --config=ardm_basicdisc_middle_end/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_basicdisc_middle_end_dw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.disc_weight_mid=0.002 \
  --config.disc_weight_end=0.002 \
  --config.disc_dropout=0.0



run_exp "guidance_advanceddisc_middle_dw1e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_advanceddisc_middle_dw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0001 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_advanceddisc_middle_dw5e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_advanceddisc_middle_dw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0005 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_advanceddisc_middle_dw2e-3" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_advanceddisc_middle_dw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.002 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_advanceddisc_middle_end_dw1e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_advanceddisc_middle_end_dw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0001 \
  --config.disc_weight_end=0.0001 \
  --config.disc_dropout=0.0

run_exp "guidance_advanceddisc_middle_end_dw5e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_advanceddisc_middle_end_dw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0005 \
  --config.disc_weight_end=0.0005 \
  --config.disc_dropout=0.0

run_exp "guidance_advanceddisc_middle_end_dw2e-3" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_advanceddisc_middle_end_dw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.002 \
  --config.disc_weight_end=0.002 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_maskmatched_rw1e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_maskmatched_rw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0001 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_maskmatched_rw5e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_maskmatched_rw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0005 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_maskmatched_rw2e-3" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_maskmatched_rw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.002 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_end_maskmatched_rw1e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_end_maskmatched_rw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0001 \
  --config.reason_weight_end=0.0001 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_end_maskmatched_rw5e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_end_maskmatched_rw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.0005 \
  --config.reason_weight_end=0.0005 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_end_maskmatched_rw2e-3" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_end_maskmatched_rw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.reason_reference_mode=mask_matched_ema \
  --config.reason_weight_mid=0.002 \
  --config.reason_weight_end=0.002 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_samemaskfull_rw1e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_samemaskfull_rw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_reference_mode=same_mask_full_ema \
  --config.reason_weight_mid=0.0001 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_samemaskfull_rw5e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_samemaskfull_rw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_reference_mode=same_mask_full_ema \
  --config.reason_weight_mid=0.0005 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_samemaskfull_rw2e-3" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_samemaskfull_rw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_reference_mode=same_mask_full_ema \
  --config.reason_weight_mid=0.002 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_end_samemaskfull_rw1e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_end_samemaskfull_rw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.reason_reference_mode=same_mask_full_ema \
  --config.reason_weight_mid=0.0001 \
  --config.reason_weight_end=0.0001 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_end_samemaskfull_rw5e-4" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_end_samemaskfull_rw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.reason_reference_mode=same_mask_full_ema \
  --config.reason_weight_mid=0.0005 \
  --config.reason_weight_end=0.0005 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0

run_exp "guidance_reasoning_middle_end_samemaskfull_rw2e-3" \
python -m ardm_advanceddisc_reasoning_middle.experiments.images.main \
  --config=ardm_advanceddisc_reasoning_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/guidance_reasoning_middle_end_samemaskfull_rw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_mid_end \
  --config.reason_reference_mode=same_mask_full_ema \
  --config.reason_weight_mid=0.002 \
  --config.reason_weight_end=0.002 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.disc_dropout=0.0



run_exp "counterfactual_scalarcf_middle_cfw1e-4" \
python -m ardm_gradalign_scalarcf_middle.experiments.images.main \
  --config=ardm_gradalign_scalarcf_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/counterfactual_scalarcf_middle_cfw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.align_aux_guidance=False \
  --config.cf_mid_weight=0.0001 \
  --config.cf_mid_drop_rate=0.25 \
  --config.cf_mid_delta_clip=0.05 \
  --config.disc_generator_train=False \
  --config.disc_dropout=0.0

run_exp "counterfactual_scalarcf_middle_cfw5e-4" \
python -m ardm_gradalign_scalarcf_middle.experiments.images.main \
  --config=ardm_gradalign_scalarcf_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/counterfactual_scalarcf_middle_cfw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.align_aux_guidance=False \
  --config.cf_mid_weight=0.0005 \
  --config.cf_mid_drop_rate=0.25 \
  --config.cf_mid_delta_clip=0.05 \
  --config.disc_generator_train=False \
  --config.disc_dropout=0.0

run_exp "counterfactual_scalarcf_middle_cfw2e-3" \
python -m ardm_gradalign_scalarcf_middle.experiments.images.main \
  --config=ardm_gradalign_scalarcf_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/counterfactual_scalarcf_middle_cfw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.align_aux_guidance=False \
  --config.cf_mid_weight=0.002 \
  --config.cf_mid_drop_rate=0.25 \
  --config.cf_mid_delta_clip=0.05 \
  --config.disc_generator_train=False \
  --config.disc_dropout=0.0

run_exp "counterfactual_gradalign_scalarcf_middle_cfw1e-4" \
python -m ardm_gradalign_scalarcf_middle.experiments.images.main \
  --config=ardm_gradalign_scalarcf_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/counterfactual_gradalign_scalarcf_middle_cfw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.align_aux_guidance=True \
  --config.align_aux_mode=soft \
  --config.cf_mid_weight=0.0001 \
  --config.cf_mid_drop_rate=0.25 \
  --config.cf_mid_delta_clip=0.05 \
  --config.disc_generator_train=False \
  --config.disc_dropout=0.0

run_exp "counterfactual_gradalign_scalarcf_middle_cfw5e-4" \
python -m ardm_gradalign_scalarcf_middle.experiments.images.main \
  --config=ardm_gradalign_scalarcf_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/counterfactual_gradalign_scalarcf_middle_cfw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.align_aux_guidance=True \
  --config.align_aux_mode=soft \
  --config.cf_mid_weight=0.0005 \
  --config.cf_mid_drop_rate=0.25 \
  --config.cf_mid_delta_clip=0.05 \
  --config.disc_generator_train=False \
  --config.disc_dropout=0.0

run_exp "counterfactual_gradalign_scalarcf_middle_cfw2e-3" \
python -m ardm_gradalign_scalarcf_middle.experiments.images.main \
  --config=ardm_gradalign_scalarcf_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/counterfactual_gradalign_scalarcf_middle_cfw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.align_aux_guidance=True \
  --config.align_aux_mode=soft \
  --config.cf_mid_weight=0.002 \
  --config.cf_mid_drop_rate=0.25 \
  --config.cf_mid_delta_clip=0.05 \
  --config.disc_generator_train=False \
  --config.disc_dropout=0.0

run_exp "counterfactual_utilitycf_middle_cfw1e-4" \
python -m ardm_selectivecf_utility_middle.experiments.images.main \
  --config=ardm_selectivecf_utility_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/counterfactual_utilitycf_middle_cfw1e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.align_aux_guidance=False \
  --config.cf_mid_weight=0.0001 \
  --config.cf_mid_drop_rate=0.125 \
  --config.cf_mid_credit_mode=utility_ema \
  --config.cf_mid_utility_estimator=score \
  --config.cf_mid_utility_momentum=0.99 \
  --config.cf_mid_delta_clip=0.05 \
  --config.cf_mid_current_credit_mix=0.25 \
  --config.disc_dropout=0.0

run_exp "counterfactual_utilitycf_middle_cfw5e-4" \
python -m ardm_selectivecf_utility_middle.experiments.images.main \
  --config=ardm_selectivecf_utility_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/counterfactual_utilitycf_middle_cfw5e-4" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.align_aux_guidance=False \
  --config.cf_mid_weight=0.0005 \
  --config.cf_mid_drop_rate=0.125 \
  --config.cf_mid_credit_mode=utility_ema \
  --config.cf_mid_utility_estimator=score \
  --config.cf_mid_utility_momentum=0.99 \
  --config.cf_mid_delta_clip=0.05 \
  --config.cf_mid_current_credit_mix=0.25 \
  --config.disc_dropout=0.0

run_exp "counterfactual_utilitycf_middle_cfw2e-3" \
python -m ardm_selectivecf_utility_middle.experiments.images.main \
  --config=ardm_selectivecf_utility_middle/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/counterfactual_utilitycf_middle_cfw2e-3" \
  --config.num_epochs=200 \
  --config.guidance_mode=unet_middle \
  --config.reason_weight_mid=0.0 \
  --config.reason_weight_end=0.0 \
  --config.disc_weight_mid=0.0 \
  --config.disc_weight_end=0.0 \
  --config.align_aux_guidance=False \
  --config.cf_mid_weight=0.002 \
  --config.cf_mid_drop_rate=0.125 \
  --config.cf_mid_credit_mode=utility_ema \
  --config.cf_mid_utility_estimator=score \
  --config.cf_mid_utility_momentum=0.99 \
  --config.cf_mid_delta_clip=0.05 \
  --config.cf_mid_current_credit_mix=0.25 \
  --config.disc_dropout=0.0



run_exp "scheduler_adaptive_alpha0.5_mix0.25" \
python -m ardm_sampling_adaptive.experiments.images.main \
  --config=ardm_sampling_adaptive/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_adaptive_alpha0.5_mix0.25" \
  --config.num_epochs=200 \
  --config.elbo_mode=adaptive_t \
  --config.adaptive_start_epoch=10 \
  --config.adaptive_num_buckets=128 \
  --config.adaptive_alpha=0.5 \
  --config.adaptive_uniform_mix=0.25 \
  --config.adaptive_max_prob_ratio=8.0 \
   

run_exp "scheduler_adaptive_alpha0.75_mix0.15" \
python -m ardm_sampling_adaptive.experiments.images.main \
  --config=ardm_sampling_adaptive/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_adaptive_alpha0.75_mix0.15" \
  --config.num_epochs=200 \
  --config.elbo_mode=adaptive_t \
  --config.adaptive_start_epoch=10 \
  --config.adaptive_num_buckets=128 \
  --config.adaptive_alpha=0.75 \
  --config.adaptive_uniform_mix=0.15 \
  --config.adaptive_max_prob_ratio=12.0 \
   

run_exp "scheduler_adaptive_alpha1.0_mix0.10" \
python -m ardm_sampling_adaptive.experiments.images.main \
  --config=ardm_sampling_adaptive/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_adaptive_alpha1.0_mix0.10" \
  --config.num_epochs=200 \
  --config.elbo_mode=adaptive_t \
  --config.adaptive_start_epoch=10 \
  --config.adaptive_num_buckets=128 \
  --config.adaptive_alpha=1.0 \
  --config.adaptive_uniform_mix=0.10 \
  --config.adaptive_max_prob_ratio=16.0 \
   

run_exp "scheduler_constrained_adaptive_alpha0.5_mix0.25" \
python -m ardm_sampling_constrained_adaptive.experiments.images.main \
  --config=ardm_sampling_constrained_adaptive/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_constrained_adaptive_alpha0.5_mix0.25" \
  --config.num_epochs=200 \
  --config.elbo_mode=constrained_adaptive_t \
  --config.adaptive_start_epoch=10 \
  --config.adaptive_num_buckets=128 \
  --config.adaptive_alpha=0.5 \
  --config.adaptive_uniform_mix=0.25 \
  --config.adaptive_max_prob_ratio=8.0 \
  --config.constrained_num_mask_buckets=64 \ 

run_exp "scheduler_constrained_adaptive_alpha0.75_mix0.15" \
python -m ardm_sampling_constrained_adaptive.experiments.images.main \
  --config=ardm_sampling_constrained_adaptive/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_constrained_adaptive_alpha0.75_mix0.15" \
  --config.num_epochs=200 \
  --config.elbo_mode=constrained_adaptive_t \
  --config.adaptive_start_epoch=10 \
  --config.adaptive_num_buckets=128 \
  --config.adaptive_alpha=0.75 \
  --config.adaptive_uniform_mix=0.15 \
  --config.adaptive_max_prob_ratio=12.0 \
  --config.constrained_num_mask_buckets=64 \ 

run_exp "scheduler_constrained_adaptive_alpha1.0_mix0.10" \
python -m ardm_sampling_constrained_adaptive.experiments.images.main \
  --config=ardm_sampling_constrained_adaptive/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_constrained_adaptive_alpha1.0_mix0.10" \
  --config.num_epochs=200 \
  --config.elbo_mode=constrained_adaptive_t \
  --config.adaptive_start_epoch=10 \
  --config.adaptive_num_buckets=128 \
  --config.adaptive_alpha=1.0 \
  --config.adaptive_uniform_mix=0.10 \
  --config.adaptive_max_prob_ratio=16.0 \
  --config.constrained_num_mask_buckets=64 \ 

run_exp "scheduler_curriculum_obs0.50_to_uniform" \
python -m ardm_sampling_curriculum.experiments.images.main \
  --config=ardm_sampling_curriculum/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_curriculum_obs0.50_to_uniform" \
  --config.num_epochs=200 \
  --config.elbo_mode=curriculum_t \
  --config.curriculum_start_observed_fraction=0.50 \
  --config.curriculum_end_observed_fraction=0.0 \
  --config.curriculum_anneal_epochs=100

run_exp "scheduler_curriculum_obs0.75_to_uniform" \
python -m ardm_sampling_curriculum.experiments.images.main \
  --config=ardm_sampling_curriculum/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_curriculum_obs0.75_to_uniform" \
  --config.num_epochs=200 \
  --config.elbo_mode=curriculum_t \
  --config.curriculum_start_observed_fraction=0.75 \
  --config.curriculum_end_observed_fraction=0.0 \
  --config.curriculum_anneal_epochs=100

run_exp "scheduler_curriculum_obs0.90_to_uniform" \
python -m ardm_sampling_curriculum.experiments.images.main \
  --config=ardm_sampling_curriculum/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_curriculum_obs0.90_to_uniform" \
  --config.num_epochs=200 \
  --config.elbo_mode=curriculum_t \
  --config.curriculum_start_observed_fraction=0.90 \
  --config.curriculum_end_observed_fraction=0.0 \
  --config.curriculum_anneal_epochs=100

run_exp "scheduler_stratified_timestep" \
python -m ardm_sampling_stratified.experiments.images.main \
  --config=ardm_sampling_stratified/experiments/images/config.py \
  --work_unit_dir="${BASE_RUN_DIR}/scheduler_stratified_timestep" \
  --config.num_epochs=200 \
  --config.elbo_mode=stratified_t


echo ""
echo "============================================================"
echo "ALL FINAL ARDM THESIS ABLATIONS COMPLETE."
echo "Expected set: 1 baseline + 46 ablations = 47 runs."
echo "Run root: ${BASE_RUN_DIR}"
echo "============================================================"
