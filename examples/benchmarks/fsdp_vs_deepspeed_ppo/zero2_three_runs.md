# DeepSpeed Zero-2 三组对比任务说明

本地脚本 `run_deepspeed_zero2_three.sh` 会同时启动三组 Qwen2.5‑0.5B PPO 训练，数据集为 `$HOME/data/gsm8k_ppo`，总步数 116、总 epoch=2，global train_batch_size=128，micro=16，grad_accum=8，zero stage 2，bf16。

三组配置（对应日志前缀）：

- `dp2_sp1_2gpu_zero2_*`: dp=2, sp=1，使用 GPU 0,1
- `dp2_sp2_4gpu_zero2_*`: dp=2, sp=2，使用 GPU 2,3,4,5
- `dp1_sp2_2gpu_zero2_*`: dp=1, sp=2，使用 GPU 6,7

其他要点：
- actor/critic 都启用 `use_remove_padding=True`
- `ulysses_sequence_parallel_size` 按组设置（sp=1 或 sp=2）
- DeepSpeed `train_batch_size` 已按 dp*sp 配平以满足 DS 断言，语义等价于 dp 缩放后的 global mini。

日志位置：`outputs/logs/deepspeed/`。可配合 `plot_deepspeed_metrics.py` 生成指标曲线。
