#!/usr/bin/env bash
#
# GRPO + DeepSpeed (ZeRO) quickstart on GSM8K with Qwen2.5-0.5B

set -euo pipefail

ZERO_STAGE=${ZERO_STAGE:-2}          # 0/1/2/3
GRAD_ACCUM=${GRAD_ACCUM:-1}
MICRO_BSZ=${MICRO_BSZ:-1}
ROLLOUT_N=${ROLLOUT_N:-2}
TOTAL_STEPS=${TOTAL_STEPS:-50}
SAVE_FREQ=${SAVE_FREQ:-50}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

if [ ! -f "${TRAIN_FILES}" ] || [ ! -f "${VAL_FILES}" ]; then
  echo "[GRPO][prep] Preparing GSM8K parquet under ${HOME}/data/gsm8k ..."
  python3 examples/data_preprocess/gsm8k.py --local_save_dir "${HOME}/data/gsm8k"
fi

python3 -m verl.trainer.main_ppo \
  actor@actor_rollout_ref.actor=deepspeed_actor \
  ref@actor_rollout_ref.ref=deepspeed_ref \
  critic=deepspeed_critic \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size=32 \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.zero_stage=${ZERO_STAGE} \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.deepspeed_config.mixed_precision=bf16 \
  actor_rollout_ref.actor.deepspeed_config.param_offload=False \
  actor_rollout_ref.actor.deepspeed_config.optimizer_offload=False \
  actor_rollout_ref.ref.zero_stage=0 \
  actor_rollout_ref.ref.deepspeed_config.mixed_precision=bf16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  critic.zero_stage=${ZERO_STAGE} \
  critic.train_batch_size=32 \
  critic.ppo_mini_batch_size=16 \
  critic.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
  critic.grad_clip=1.0 \
  critic.deepspeed_config.mixed_precision=bf16 \
  critic.deepspeed_config.param_offload=False \
  critic.deepspeed_config.optimizer_offload=False \
  trainer.logger=console \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=-1 \
  trainer.project_name=verl_grpo_deepspeed_gsm8k \
  trainer.experiment_name=qwen2_5_0_5b_ds_zero${ZERO_STAGE} "$@"
