# DeepSpeed Integration (ZeRO path)

This note captures the minimal DeepSpeed support ported onto the latest `main` and how to run PPO/GRPO with ZeRO on VERL.

## What Works
- Training backend: DeepSpeed for actor/critic (ZeRO-0/1/2/3) with optional CPU offload for params/optimizer.
- Parallelism: data parallel (DP) only; sequence/tensor/pipeline parallel paths are removed in this minimal subset.
- Precision: bf16/fp16 via `deepspeed_config.mixed_precision`.
- Rollout: async vLLM or HF backends; SGLang support is disabled.
- Checkpointing: `DeepSpeedCheckpointManager` saves/loads ZeRO shards for actor/critic.
- Algorithms: PPO and GRPO (batch normalization accounts for `rollout.n`).

## Key Components
- `verl/workers/deepspeed_workers.py` – builds actor/ref/critic with DeepSpeed and normalizes batches by DP size only.
- `verl/workers/deepspeed_parallel.py` – DP layouts and batch-size normalization (sequence parallel blocked).
- `verl/utils/deepspeed_utils.py` – config helper + engine launcher.
- `verl/utils/checkpoint/deepspeed_checkpoint_manager.py` – ZeRO-aware checkpoints.
- Configs: `trainer/config/actor|critic|engine|optim` include DeepSpeed variants (`zero_stage`, `param_offload`, `optimizer_offload`, `mixed_precision`).
- Rollout configs: `actor_rollout_ref.rollout.mode` must be `async` with `rollout.name` set to `vllm` or `hf`.

## Quickstart: GRPO on GSM8K (8×A100, Qwen2.5-0.5B)
Use the bundled recipe:
```bash
ZERO_STAGE=2 TOTAL_STEPS=50 SAVE_FREQ=50 \
  bash examples/deepspeed/run_qwen2_5_0_5b_grpo_ds_gsm8k.sh
```
Key knobs:
- `actor@actor_rollout_ref.actor=deepspeed_actor` / `critic=deepspeed_critic` (set inside the script).
- `actor_rollout_ref.actor.zero_stage` / `critic.zero_stage` – ZeRO stage (0/1/2/3).
- `actor_rollout_ref.actor.ppo_mini_batch_size`, `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` – drive GAS; values are per-DP rank (no SP scaling).
- `trainer.save_freq` – checkpoint cadence (handled by DeepSpeedCheckpointManager).

## Tips & Limits
- LoRA/PEFT and sequence parallel are disabled for DeepSpeed in this subset; configs with `lora_rank>0` or `ulysses_sequence_parallel_size>1` will error.
- Rollout engines: vLLM async server mode is recommended; HF works for small models. SGLang is not supported.
- Keep `rollout.n > 1` for GRPO; the batch normalizer multiplies mini-batch size accordingly.
- Run with `VERL_LOGGING_LEVEL=INFO` to see DeepSpeed init and checkpoint traces if debugging.
