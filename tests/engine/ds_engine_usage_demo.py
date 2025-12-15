"""
Minimal DeepSpeed engine usage demo for sanity-checking backward/step/zero_grad.

Run on 1 GPU (or CPU if CUDA is unavailable):
  python tests/engine/ds_engine_usage_demo.py

Custom envs:
  GAS=2             # gradient_accumulation_steps
  MICRO=2           # micro batch size
  STEPS=2           # number of training steps
  LR=1e-3           # learning rate
  TORCH_OPT=1       # pass a torch.optim.AdamW into initialize (instead of DS-built)
  MANUAL_SCALE=1    # use manual loss scaling (loss/gas, scale_wrt_gas=False)
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import deepspeed
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"DeepSpeed not installed: {exc}")


def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DemoConfig:
    gas: int
    micro: int
    steps: int
    lr: float
    torch_opt: bool
    manual_scale: bool


def build_model(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )
    return model


def build_data(steps: int, batch: int, in_dim: int, out_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(steps, batch, in_dim)
    y = torch.randn(steps, batch, out_dim)
    return x, y


def param_hash(model: nn.Module) -> float:
    vals = []
    with torch.no_grad():
        for p in model.parameters():
            flat = p.detach().float().reshape(-1)
            if flat.numel() == 0:
                continue
            vals.append(flat[:4096])
    if not vals:
        return 0.0
    flat_cat = torch.cat(vals)
    flat64 = flat_cat.to(torch.float64)
    return float(torch.sum(flat64 * flat64).cpu())  # simple L2 sum hash


def run_torch_ref(cfg: DemoConfig, model: nn.Module, data: Tuple[torch.Tensor, torch.Tensor]) -> nn.Module:
    model = copy.deepcopy(model)
    model.to(device())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    x_all, y_all = data
    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        start = step
        xb = x_all[start]
        yb = y_all[start]
        # simulate micro-batching: split by micro size
        micro_splits = torch.split(torch.arange(xb.shape[0]), cfg.micro)
        for micro_idx, idxs in enumerate(micro_splits):
            if idxs.numel() == 0:
                continue
            x = xb[idxs].to(device())
            y = yb[idxs].to(device())
            out = model(x)
            loss = criterion(out, y)
            loss = loss / cfg.gas  # manual scale to mimic DS scale_wrt_gas
            loss.backward()
            print(
                f"[torch] step={step} micro={micro_idx} loss={loss.item():.6f} "
                f"norm={float(torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))):.4f}"
            )
        opt.step()
        print(f"[torch] step={step} param_hash={param_hash(model):.6e}")
    return model


def run_deepspeed(cfg: DemoConfig, model: nn.Module, data: Tuple[torch.Tensor, torch.Tensor]) -> nn.Module:
    model = copy.deepcopy(model)
    model.to(device())
    os.environ.setdefault("DEEPSPEED_ENABLE_MPI", "0")  # avoid mpi4py dependency
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    # Initialize a single-process process group so DeepSpeed can run without torchrun.
    import torch.distributed as dist
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        port = os.environ.get("DS_DEMO_PORT", "29500")
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
        )
    ds_config = {
        "train_batch_size": cfg.micro * cfg.gas,
        "train_micro_batch_size_per_gpu": cfg.micro,
        "gradient_accumulation_steps": cfg.gas,
        "zero_optimization": {"stage": 0},
        "bf16": {"enabled": False},
        "fp16": {"enabled": False},
        "optimizer": {"type": "AdamW", "params": {"lr": cfg.lr}},
        "steps_per_print": 1,
    }
    torch_opt = None
    if cfg.torch_opt:
        torch_opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        optimizer=torch_opt,
        dist_init_required=False,  # single-process demo
    )
    criterion = nn.MSELoss()
    x_all, y_all = data
    for step in range(cfg.steps):
        try:
            engine.zero_grad(set_to_none=True)
        except TypeError:
            engine.zero_grad()
        xb = x_all[step]
        yb = y_all[step]
        # simple contiguous micro batches
        for micro_idx in range(cfg.gas):
            start = micro_idx * cfg.micro
            end = start + cfg.micro
            x = xb[start:end].to(device())
            y = yb[start:end].to(device())
            out = engine(x)
            loss = criterion(out, y)
            if cfg.manual_scale:
                loss = loss / cfg.gas
            engine.set_gradient_accumulation_boundary(micro_idx == cfg.gas - 1)
            engine.backward(loss, scale_wrt_gas=not cfg.manual_scale)
            print(
                f"[ds] step={step} micro={micro_idx} loss={loss.item():.6f} "
                f"ga={engine.gradient_accumulation_steps()}, "
                f"boundary={engine.is_gradient_accumulation_boundary()}"
            )
        engine.step()
        print(
            f"[ds] step={step} param_hash={param_hash(engine.module):.6e} "
            f"opt_class={type(engine.optimizer).__name__} base_opt={type(getattr(engine.optimizer, 'optimizer', None)).__name__}"
        )
    return engine.module


def main():
    set_seed(1234)
    cfg = DemoConfig(
        gas=int(os.getenv("GAS", "1")),
        micro=int(os.getenv("MICRO", "2")),
        steps=int(os.getenv("STEPS", "2")),
        lr=float(os.getenv("LR", "1e-3")),
        torch_opt=bool(int(os.getenv("TORCH_OPT", "0"))),
        manual_scale=bool(int(os.getenv("MANUAL_SCALE", "0"))),
    )
    in_dim, hidden, out_dim = 8, 16, 4
    batch = cfg.gas * cfg.micro
    data = build_data(cfg.steps, batch, in_dim, out_dim)
    base_model = build_model(in_dim, hidden, out_dim)
    torch_model = run_torch_ref(cfg, base_model, data)
    ds_model = run_deepspeed(cfg, base_model, data)
    # Compare end params
    ref_params = torch.cat([p.detach().float().reshape(-1) for p in torch_model.parameters()])
    ds_params = torch.cat([p.detach().float().reshape(-1) for p in ds_model.parameters()])
    diff = (ref_params - ds_params).abs()
    print(
        f"[compare] L2={float(torch.linalg.vector_norm(diff).cpu()):.6e} "
        f"max={float(diff.max().cpu()):.6e} "
        f"torch_hash={param_hash(torch_model):.6e} ds_hash={param_hash(ds_model):.6e}"
    )


if __name__ == "__main__":
    main()
