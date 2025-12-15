#!/usr/bin/env bash
set -euo pipefail

# Three DeepSpeed PPO runs (Qwen2.5-0.5B, zero stage 2):
#   1) dp=2, sp=1 on 2 GPUs
#   2) dp=2, sp=2 on 4 GPUs
#   3) dp=1, sp=2 on 2 GPUs
# All use total_epochs=2 and total_training_steps=116 with equal global batch (128 tokens)
# micro=16, grad_accum=8 (per-rank); per-rank mini is normalized by dp inside the worker.

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONHASHSEED=42
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

PPO_DATA_DIR="${PPO_DATA_DIR:-$HOME/data/gsm8k_ppo}"
TRAIN_FILES="['$PPO_DATA_DIR/train.parquet']"
VAL_FILES="['$PPO_DATA_DIR/test.parquet']"

if [ ! -f "$PPO_DATA_DIR/train.parquet" ] || [ ! -f "$PPO_DATA_DIR/test.parquet" ]; then
  echo "[prepare] GSM8K PPO parquet not found at $PPO_DATA_DIR; please preprocess via examples/data_preprocess/gsm8k.py"
  exit 1
fi

CFG_PATH="$ROOT_DIR/examples/benchmarks/fsdp_vs_deepspeed_ppo/config"
CFG_NAME="deepspeed_ppo_benchmark"

LOG_DIR="$ROOT_DIR/outputs/logs/deepspeed"
mkdir -p "$LOG_DIR"
ts() { date +"%Y%m%d-%H%M%S"; }

base_overrides=(
  --config-path "$CFG_PATH"
  --config-name "$CFG_NAME"
  data.train_files="$TRAIN_FILES"
  data.val_files="$VAL_FILES"
  data.seed=42
  actor_rollout_ref.rollout.seed=42
  data.train_batch_size=128
  actor_rollout_ref.actor.ppo_mini_batch_size=128
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
  actor_rollout_ref.actor.gradient_accumulation_steps=8
  actor_rollout_ref.actor.zero_stage=2
  actor_rollout_ref.actor.train_batch_size=128
  actor_rollout_ref.actor.train_micro_batch_size_per_gpu=16
  actor_rollout_ref.actor.deepspeed_config.model_dtype=bf16
  actor_rollout_ref.actor.deepspeed_config.mixed_precision=bf16
  actor_rollout_ref.actor.deepspeed_config.param_offload=false
  actor_rollout_ref.actor.deepspeed_config.optimizer_offload=false
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.12
  critic.ppo_mini_batch_size=128
  critic.ppo_micro_batch_size_per_gpu=16
  critic.gradient_accumulation_steps=8
  critic.zero_stage=2
  critic.train_batch_size=128
  critic.train_micro_batch_size_per_gpu=16
  critic.deepspeed_config.model_dtype=bf16
  critic.deepspeed_config.mixed_precision=bf16
  critic.deepspeed_config.param_offload=false
  critic.deepspeed_config.optimizer_offload=false
  critic.model.use_remove_padding=True
  trainer.total_training_steps=116
  trainer.total_epochs=2
  trainer.logger='["console","file"]'
  trainer.resume_mode=disable
)

echo "[run] dp=2, sp=1 on GPUs 0,1"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
python3 -m verl.trainer.main_ppo \
  "${base_overrides[@]}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.deepspeed_config.ulysses_sequence_parallel_size=1 \
  critic.ulysses_sequence_parallel_size=1 \
  critic.deepspeed_config.ulysses_sequence_parallel_size=1 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  +ray_kwargs.ray_init.num_gpus=2 \
  +ray_kwargs.ray_init.num_cpus=8 \
  > "$LOG_DIR/dp2_sp1_2gpu_zero2_$(ts).log" 2>&1 &

echo "[run] dp=2, sp=2 on GPUs 2,3,4,5"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_4GPU:-2,3,4,5} \
python3 -m verl.trainer.main_ppo \
  "${base_overrides[@]}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.actor.deepspeed_config.ulysses_sequence_parallel_size=2 \
  critic.ulysses_sequence_parallel_size=2 \
  critic.deepspeed_config.ulysses_sequence_parallel_size=2 \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  +ray_kwargs.ray_init.num_gpus=4 \
  +ray_kwargs.ray_init.num_cpus=16 \
  > "$LOG_DIR/dp2_sp2_4gpu_zero2_$(ts).log" 2>&1 &

echo "[run] dp=1, sp=2 on GPUs 6,7"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SP2:-6,7} \
python3 -m verl.trainer.main_ppo \
  "${base_overrides[@]}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.actor.deepspeed_config.ulysses_sequence_parallel_size=2 \
  critic.ulysses_sequence_parallel_size=2 \
  critic.deepspeed_config.ulysses_sequence_parallel_size=2 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  +ray_kwargs.ray_init.num_gpus=2 \
  +ray_kwargs.ray_init.num_cpus=8 \
  > "$LOG_DIR/dp1_sp2_2gpu_zero2_$(ts).log" 2>&1 &

echo "[submit] Launched three jobs in background. Logs under $LOG_DIR"
