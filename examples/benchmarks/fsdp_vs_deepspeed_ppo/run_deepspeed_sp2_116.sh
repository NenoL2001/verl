#!/usr/bin/env bash
set -euo pipefail

# Run DeepSpeed PPO benchmark with SP=2 in two modes:
#   1) 2 GPUs (SP=2)
#   2) 4 GPUs (DP=2 x SP=2 hybrid)
# Both runs use 116 training steps with deterministic rollout (temperature=0, do_sample=False).

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

common_args=(
  --config-path "$CFG_PATH"
  --config-name "$CFG_NAME"
  data.train_files="$TRAIN_FILES"
  data.val_files="$VAL_FILES"
  data.seed=42
  actor_rollout_ref.rollout.seed=42
  data.train_batch_size=128
  actor_rollout_ref.actor.ppo_mini_batch_size=128
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
  actor_rollout_ref.actor.gradient_accumulation_steps=8
  actor_rollout_ref.actor.zero_stage=0
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
  actor_rollout_ref.actor.deepspeed_config.ulysses_sequence_parallel_size=2
  actor_rollout_ref.actor.deepspeed_config.model_dtype=bf16
  actor_rollout_ref.actor.deepspeed_config.mixed_precision=bf16
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.12
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.temperature=0
  actor_rollout_ref.rollout.do_sample=False
  critic.ulysses_sequence_parallel_size=2
  critic.deepspeed_config.ulysses_sequence_parallel_size=2
  critic.ppo_mini_batch_size=128
  critic.ppo_micro_batch_size_per_gpu=8
  critic.gradient_accumulation_steps=8
  critic.deepspeed_config.model_dtype=bf16
  critic.deepspeed_config.mixed_precision=bf16
  trainer.total_training_steps=116
  trainer.total_epochs=1
  trainer.logger='["console","file"]'
  trainer.resume_mode=disable
)

echo "[run] 2-GPU actor with SP=2 (116 steps)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
python3 -m verl.trainer.main_ppo \
  "${common_args[@]}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  +ray_kwargs.ray_init.num_gpus=2 \
  +ray_kwargs.ray_init.num_cpus=8 \
  > "$LOG_DIR/actor_sp2_2gpu_116_$(ts).log" 2>&1

echo "[run] 4-GPU actor with DP=2 x SP=2 (116 steps)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_4GPU:-0,1,2,3}
python3 -m verl.trainer.main_ppo \
  "${common_args[@]}" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  +ray_kwargs.ray_init.num_gpus=4 \
  +ray_kwargs.ray_init.num_cpus=16 \
  > "$LOG_DIR/actor_sp2_dp2_4gpu_116_$(ts).log" 2>&1

echo "[done] Logs saved to $LOG_DIR"
