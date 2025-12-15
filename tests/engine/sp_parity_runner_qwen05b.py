"""
Parity runner for DeepSpeed vs FSDP (critic/actor).

Run example (sp=2, two GPUs, bf16, one step):
  CUDA_VISIBLE_DEVICES=0,1 IMPL=fsdp PARITY_RUN_ROLE=critic PARITY_RUN_ID=crit_sp2 \
  PARITY_SP_SIZE=2 PARITY_BATCH_SIZE=64 PARITY_MICRO_BATCH=8 PARITY_GAS=8 \
  PARITY_PROMPT_LEN=32 PARITY_RESPONSE_LEN=64 PARITY_DTYPE=bf16 PARITY_DETERMINISTIC_TRAIN=1 \
  torchrun --nproc_per_node=2 tests/engine/sp_parity_runner_qwen05b.py
Then run the same with IMPL=deepspeed and compare rank0.pt using compare_parity_artifacts.py.
"""

from __future__ import annotations

import copy
import os
import time
import uuid
from typing import Dict, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, StateDictType, FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer
from flash_attn.bert_padding import unpad_input

# Default to python clip for DeepSpeed parity runs unless explicitly overridden.
os.environ.setdefault("PARITY_DS_PYTHON_CLIP", "1")

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.workers.config import (
    DeepSpeedActorConfig,
    DeepSpeedCriticConfig,
    DeepSpeedEngineConfig,
    FSDPActorConfig,
    FSDPCriticConfig,
    FSDPCriticModelCfg,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    OptimizerConfig,
)
from verl.workers.critic.dp_critic import _hash_tensor
from verl.workers.deepspeed_workers import ActorRolloutRefWorker as DSActorWorker
from verl.workers.deepspeed_workers import CriticWorker as DSCriticWorker
from verl.workers.fsdp_workers import ActorRolloutRefWorker as FSDPActorWorker
from verl.workers.fsdp_workers import CriticWorker as FSDPCriticWorker


def _init_dist():
    if dist.is_initialized():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = os.environ.get("PARITY_BACKEND")
    if not backend:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_id = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
    dist.init_process_group(backend=backend, init_method="env://", device_id=device_id)
    return device


def _seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_batch(
    batch_size: int,
    prompt_len: int,
    response_len: int,
    dtype: torch.dtype,
    seed: int,
    role: str = "critic",
    real_data: bool = False,
    real_case: str | None = None,
    real_case_id: int = 0,
    tokenizer=None,
) -> DataProto:
    total_len = prompt_len + response_len
    torch.manual_seed(seed)
    if not real_data:
        input_ids = torch.randint(low=10, high=200, size=(batch_size, total_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, total_len, dtype=torch.long)
        responses = input_ids[:, -response_len:]
        response_mask = torch.ones(batch_size, response_len, dtype=torch.float32)
        prompts = input_ids[:, :prompt_len]
        lengths = [response_len for _ in range(batch_size)]
    else:
        assert tokenizer is not None, "tokenizer required for real_data"
        if real_case not in (None, "varlen_pad"):
            raise ValueError(f"Unsupported real_case: {real_case}")
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        prompts_txt = [
            "Hello, how are you?",
            "Please summarize the text.",
            "What is the capital of France?",
            "List three colors.",
            "Explain quantum mechanics briefly.",
            "Describe your favorite book.",
            "Tell me a joke.",
            "Give me a recipe idea.",
        ]
        prompts_txt = (prompts_txt * ((batch_size + len(prompts_txt) - 1) // len(prompts_txt)))[:batch_size]
        enc_p = tokenizer(
            prompts_txt,
            padding="max_length",
            max_length=prompt_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        case0 = [64, 48, 40, 56, 64, 48, 40, 56]
        case1 = [60, 44, 36, 52, 60, 44, 36, 52]
        lengths_pool = case0 if real_case_id % 2 == 0 else case1
        lengths = [max(1, min(response_len, lengths_pool[i % len(lengths_pool)])) for i in range(batch_size)]
        resp_token_lists = []
        for i, eff_len in enumerate(lengths):
            torch.manual_seed(seed + 100 + i)
            resp_token_lists.append(torch.randint(low=10, high=200, size=(eff_len,), dtype=torch.long).tolist())
        enc_r = tokenizer.pad(
            {"input_ids": resp_token_lists},
            padding="max_length",
            max_length=response_len,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # replace any pad set by tokenizer to a known id to avoid accidental negatives
        enc_r.input_ids = enc_r.input_ids.masked_fill(enc_r.attention_mask == 0, pad_id)
        prompts = enc_p.input_ids
        responses = enc_r.input_ids
        response_mask = enc_r.attention_mask.to(torch.float32)
        input_ids = torch.cat([prompts, responses], dim=1)
        attention_mask = torch.cat([enc_p.attention_mask, enc_r.attention_mask], dim=1)
    position_ids = compute_position_id_with_mask(attention_mask)

    torch.manual_seed(seed + 1)
    values = torch.randn(batch_size, response_len, dtype=dtype)
    torch.manual_seed(seed + 2)
    returns = torch.randn(batch_size, response_len, dtype=dtype) * 0.5

    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
    }
    if role == "critic":
        tensors.update({"values": values, "returns": returns})
    else:
        torch.manual_seed(seed + 3)
        old_log_probs = torch.randn(batch_size, response_len, dtype=dtype) * 0.1
        torch.manual_seed(seed + 4)
        advantages = torch.randn(batch_size, response_len, dtype=dtype) * 0.2
        dp_size_tensor = torch.full((batch_size,), 1, dtype=torch.int32)
        batch_num_tokens = response_mask.sum(dim=-1).to(torch.int32)
        global_batch_tensor = torch.full((batch_size,), batch_size, dtype=torch.int32)
        tensors.update(
            {
                "old_log_probs": old_log_probs,
                "advantages": advantages,
                "dp_size": dp_size_tensor,
                "batch_num_tokens": batch_num_tokens,
                "global_batch_size": global_batch_tensor,
            }
        )

    meta_info = {
        "global_token_num": lengths,
        "rng_seed": seed,
        "temperature": 1.0,
    }
    return DataProto.from_dict(tensors=tensors, meta_info=meta_info)


def _broadcast_batch(batch: DataProto, device: torch.device) -> DataProto:
    payload = {"tensors": batch.batch, "meta": batch.meta_info} if dist.get_rank() == 0 else None
    obj_list = [payload]
    dist.broadcast_object_list(obj_list, src=0)
    tensors = obj_list[0]["tensors"]
    meta = obj_list[0]["meta"]
    for k, v in tensors.items():
        tensors[k] = v.to(device)
    batch.batch = tensors
    batch.meta_info = meta
    return batch


def _gather_fsdp_state_dict(module) -> Dict[str, torch.Tensor]:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg):
        state = module.state_dict()
    if dist.get_rank() != 0:
        return {}
    return {k.replace("_fsdp_wrapped_module.", ""): v.cpu() for k, v in state.items()}


def _gather_ds_state_dict(module) -> Dict[str, torch.Tensor]:
    if dist.get_rank() != 0:
        return {}
    state = module.state_dict()
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _choose_keys(common: Iterable[str], max_keys: int = 5) -> list[str]:
    common = sorted(common)
    preferred = [k for k in common if any(pat in k for pat in ("embed", "layers.0", "v_head", "value_head", "lm_head"))]
    chosen = preferred[:max_keys] or common[:max_keys]
    if not chosen:
        raise RuntimeError("No overlapping parameter keys found for comparison.")
    return chosen


def _compute_grad_stats(module) -> dict:
    total_sq = 0.0
    stats = []
    none_cnt = 0
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            none_cnt += 1
            continue
        g32 = g.detach().float()
        total_sq += float(torch.sum(g32 * g32).item())
        norm_val = float(torch.norm(g32).item())
        max_abs = float(torch.max(torch.abs(g32)).item())
        mean_abs = float(torch.mean(torch.abs(g32)).item())
        stats.append((name, norm_val, max_abs, mean_abs))
    stats.sort(key=lambda x: x[1], reverse=True)
    return {
        "grad_norm": float(total_sq ** 0.5),
        "grad_none": none_cnt,
        "grad_topk": stats[: min(10, len(stats))],
    }


def _clone_to_cpu_dict(tensors: Dict[str, torch.Tensor] | None) -> Dict[str, torch.Tensor] | None:
    if tensors is None:
        return None
    out: Dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def _unwrap_model(worker, role: str, impl: str):
    module = worker.critic_module if role == "critic" else (worker.actor_module_fsdp if impl == "fsdp" else worker.actor_module)

    def _walk(mod):
        changed = True
        while changed:
            changed = False
            for attr in ("module", "_fsdp_wrapped_module", "module_wrapped"):
                child = getattr(mod, attr, None)
                if isinstance(child, torch.nn.Module):
                    mod = child
                    changed = True
                    break
        for attr in ("model", "module", "base_model"):
            child = getattr(mod, attr, None)
            if isinstance(child, torch.nn.Module):
                mod = child
                break
        return mod

    return _walk(module)


def _init_hook_store() -> Dict[str, torch.Tensor | dict | None]:
    return {
        "layer0_vproj_in": None,
        "layer0_vproj_out": None,
        "layer0_vproj_in_meta": None,
        "layer0_vproj_out_meta": None,
    }


def _make_tensor_entry(tensor: torch.Tensor, token_dim: int = 1, max_tokens: int = 4, max_hidden: int = 128) -> dict:
    if tensor is None:
        return {"meta": None, "slice": None}
    slc_tokens = min(tensor.shape[token_dim], max_tokens) if tensor.dim() > token_dim else tensor.shape[0]
    hidden_dim = tensor.dim() - 1
    slc_hidden = min(tensor.shape[hidden_dim], max_hidden) if tensor.dim() > 1 else 1
    slicer = [slice(None)] * tensor.dim()
    slicer[token_dim] = slice(0, slc_tokens)
    slicer[hidden_dim] = slice(0, slc_hidden)
    part = tensor[tuple(slicer)].reshape(-1).detach().to("cpu").clone()
    return {
        "meta": {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)},
        "slice": part,
    }


def _param_hash(module, keys: Iterable[str]) -> dict:
    out = {}
    for name, p in module.named_parameters():
        if name in keys:
            out[name] = float(p.detach().reshape(-1)[:4096].sum().item())
    return out


PROBE_NAMES = [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "v_head.weight",
    "value_head.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]


def _capture_param_slice(module, device):
    out = {}
    for name, p in module.named_parameters():
        if name in PROBE_NAMES:
            out[name] = p.detach().to("cpu").reshape(-1)[:4096].clone()
    out["rng_cpu"] = torch.get_rng_state()
    if torch.cuda.is_available():
        out["rng_cuda"] = torch.cuda.get_rng_state(device=device)
    return out


def _attach_grad_hooks(module, grad_store: dict):
    handles = []

    def make_hook(name):
        def hook(grad):
            grad_store[name] = grad.detach().to("cpu").reshape(-1)[:4096].clone()

        return hook

    for name, p in module.named_parameters():
        if name in PROBE_NAMES and p.requires_grad:
            handles.append(p.register_hook(make_hook(name)))
    return handles


def _register_attn_hooks(model, hook_store: dict, hook_handles: list, rank: int):
    slice_len = 4096
    target_vproj = None
    target_self_attn = None
    for name, module in model.named_modules():
        if target_vproj is None and "layers.0" in name and "self_attn.v_proj" in name:
            target_vproj = module
        if target_self_attn is None and "layers.0" in name and "self_attn" in name:
            target_self_attn = module
        if target_vproj is not None and target_self_attn is not None:
            break
    target_in = target_vproj or target_self_attn
    target_out = target_vproj
    if target_in is None and target_out is None:
        if rank == 0:
            print("[hook] layer0 self_attn/v_proj modules not found, available modules:", flush=True)
            for name, _ in model.named_modules():
                print(name, flush=True)
        return

    def in_hook(mod, inp, _out):
        if not inp:
            return
        x = inp[0]
        hook_store["layer0_vproj_in"] = x.detach().to("cpu").reshape(-1)[:slice_len].clone()
        hook_store["layer0_vproj_in_meta"] = {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }

    def out_hook(_mod, _inp, out):
        y = out
        if not torch.is_tensor(y):
            return
        hook_store["layer0_vproj_out"] = y.detach().to("cpu").reshape(-1)[:slice_len].clone()
        hook_store["layer0_vproj_out_meta"] = {
            "shape": tuple(y.shape),
            "dtype": str(y.dtype),
            "device": str(y.device),
        }

    if target_in is not None:
        hook_handles.append(target_in.register_forward_hook(in_hook))
    if target_out is not None:
        hook_handles.append(target_out.register_forward_hook(out_hook))


def _register_forward_chain_hooks(model, chain_store: dict, hook_handles: list, rank: int):
    target_embed = None
    target_attn = None
    target_vproj = None
    target_mlp = None
    for name, module in model.named_modules():
        if target_embed is None and name.endswith("embed_tokens"):
            target_embed = module
        if target_attn is None and "layers.0" in name and name.endswith("self_attn"):
            target_attn = module
        if target_vproj is None and "layers.0" in name and "self_attn.v_proj" in name:
            target_vproj = module
        if target_mlp is None and "layers.0" in name and name.endswith("mlp"):
            target_mlp = module
        if target_embed and target_attn and target_vproj and target_mlp:
            break
    if rank == 0 and target_embed is None:
        print("[fwd_chain] embed_tokens not found", flush=True)
    if target_embed is not None:
        def embed_hook(_m, _inp, out):
            if chain_store.get("embed_out") is None and torch.is_tensor(out):
                chain_store["embed_out"] = out.detach().to("cpu")
        hook_handles.append(target_embed.register_forward_hook(embed_hook))
    if target_attn is not None:
        def attn_hook(mod, inp, out):
            if chain_store.get("layer0_attn_in") is None and inp:
                x = inp[0]
                if torch.is_tensor(x):
                    chain_store["layer0_attn_in"] = x.detach().to("cpu")
            y = out[0] if isinstance(out, (tuple, list)) else out
            if chain_store.get("layer0_attn_out") is None and torch.is_tensor(y):
                chain_store["layer0_attn_out"] = y.detach().to("cpu")
        hook_handles.append(target_attn.register_forward_hook(attn_hook))
    if target_vproj is not None:
        def vproj_hook(_m, _inp, out):
            if chain_store.get("vproj_out") is None and torch.is_tensor(out):
                chain_store["vproj_out"] = out.detach().to("cpu")
        hook_handles.append(target_vproj.register_forward_hook(vproj_hook))
    if target_mlp is not None:
        def mlp_hook(_m, inp, out):
            if chain_store.get("layer0_mlp_in") is None and inp:
                x = inp[0]
                if torch.is_tensor(x):
                    chain_store["layer0_mlp_in"] = x.detach().to("cpu")
            y = out[0] if isinstance(out, (tuple, list)) else out
            if chain_store.get("layer0_mlp_out") is None and torch.is_tensor(y):
                chain_store["layer0_mlp_out"] = y.detach().to("cpu")
        hook_handles.append(target_mlp.register_forward_hook(mlp_hook))


def main():
    impl = os.environ.get("IMPL", "fsdp")
    assert impl in ("fsdp", "deepspeed")
    role = os.environ.get("PARITY_RUN_ROLE", "critic")
    assert role in ("critic", "actor")
    device = _init_dist()
    world = dist.get_world_size()
    sp_size = int(os.environ.get("PARITY_SP_SIZE", "2"))
    assert world % sp_size == 0
    dp_size = world // sp_size

    run_id = os.environ.get("PARITY_RUN_ID", time.strftime("%Y%m%d-%H%M%S")) or uuid.uuid4().hex
    out_root = os.environ.get("PARITY_OUT_DIR", "/tmp/verl_sp_parity")
    out_dir = os.path.join(out_root, run_id, impl)
    os.makedirs(out_dir, exist_ok=True)
    rank = dist.get_rank()

    seed = 1234
    _seed_everything(seed)
    steps = int(os.environ.get("PARITY_STEPS", "1"))
    deterministic_train = os.environ.get("PARITY_DETERMINISTIC_TRAIN", "0") == "1"
    grad_accum_override = int(os.environ.get("PARITY_GAS", "1"))
    dump_sp_io = os.environ.get("PARITY_DUMP_SP_IO", "0") == "1"
    dump_attn_hook = os.environ.get("PARITY_DUMP_ATTN_HOOK", "0") == "1"
    dump_forward_chain = os.environ.get("PARITY_DUMP_FORWARD_CHAIN", "0") == "1"
    dump_steps = os.environ.get("PARITY_DUMP_STEPS", "0")
    real_data = os.environ.get("PARITY_REAL_DATA", "0") == "1"
    real_case = os.environ.get("PARITY_REAL_CASE", "varlen_pad")
    real_case_id = int(os.environ.get("PARITY_REAL_CASE_ID", "0"))

    batch_size = int(os.environ.get("PARITY_BATCH_SIZE", "1"))
    prompt_len = int(os.environ.get("PARITY_PROMPT_LEN", "2"))
    response_len = int(os.environ.get("PARITY_RESPONSE_LEN", "4"))
    micro_batch_env = int(os.environ.get("PARITY_MICRO_BATCH", str(batch_size)))
    micro_bsz_per_gpu = micro_batch_env
    local_batch = batch_size // dp_size
    if micro_batch_env * grad_accum_override != local_batch:
        print(
            f"[warn] micro*gas mismatch: micro={micro_batch_env} gas={grad_accum_override} local_batch={local_batch} dp_size={dp_size}",
            flush=True,
        )
    total_len = prompt_len + response_len
    dtype_env = os.environ.get("PARITY_DTYPE", "fp32").lower()
    if dtype_env == "bf16":
        dtype = torch.bfloat16
    elif dtype_env == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    mp_conf = {
        "param_dtype": "fp32" if dtype == torch.float32 else ("bf16" if dtype == torch.bfloat16 else "fp16"),
        "reduce_dtype": "fp32",
        "buffer_dtype": "fp32",
    }

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    if impl == "fsdp":
        # Pass global mini batch and let worker normalize by DP/SP, matching FSDP defaults.
        per_rank_mini_batch = batch_size

        fsdp_model_cfg = FSDPCriticModelCfg(
            path=model_name,
            tokenizer_path=model_name,
            use_remove_padding=True,
            trust_remote_code=True,
            enable_gradient_checkpointing=False,
            fsdp_config=FSDPEngineConfig(
                fsdp_size=-1,
                forward_prefetch=False,
                reshard_after_forward=False,
                model_dtype="bf16" if dtype == torch.bfloat16 else "fp32",
                use_orig_params=True,
                mixed_precision=mp_conf,
            ),
            override_config={"attn_implementation": "eager"},
        )
        fsdp_cfg = FSDPCriticConfig(
            strategy="fsdp",
            ppo_mini_batch_size=per_rank_mini_batch,
            ppo_micro_batch_size_per_gpu=micro_bsz_per_gpu,
            ppo_epochs=1,
            rollout_n=1,
            ppo_max_token_len_per_gpu=total_len,
            ulysses_sequence_parallel_size=sp_size,
            loss_agg_mode="token-mean",
            grad_clip=1e9,
            model=fsdp_model_cfg,
            optim=FSDPOptimizerConfig(
                lr=5e-4,
                weight_decay=0.0,
                betas=(0.9, 0.999),
                optimizer="AdamW",
                optimizer_impl="torch.optim",
                lr_scheduler_type="constant",
            ),
        )
        if role == "critic":
            worker = FSDPCriticWorker(fsdp_cfg)
        else:
            fsdp_actor_cfg = FSDPActorConfig(
                strategy="fsdp",
                ppo_mini_batch_size=per_rank_mini_batch,
                ppo_micro_batch_size_per_gpu=micro_bsz_per_gpu,
                ppo_epochs=1,
                rollout_n=1,
                ppo_max_token_len_per_gpu=total_len,
                ulysses_sequence_parallel_size=sp_size,
                loss_agg_mode="token-mean",
                grad_clip=1e9,
                optim=FSDPOptimizerConfig(
                    lr=5e-4,
                    weight_decay=0.0,
                    betas=(0.9, 0.999),
                    optimizer="AdamW",
                    optimizer_impl="torch.optim",
                    lr_scheduler_type="constant",
                    grad_clip=1e9,
                ),
                fsdp_config=fsdp_model_cfg.fsdp_config,
            )
            fsdp_actor_cfg.model = HFModelConfig(
                path=model_name,
                tokenizer_path=model_name,
                use_remove_padding=True,
                trust_remote_code=True,
                enable_gradient_checkpointing=False,
                override_config={"attn_implementation": "eager"},
            )
            from types import SimpleNamespace

            actor_cfg = SimpleNamespace(
                actor=fsdp_actor_cfg, rollout=SimpleNamespace(n=1), model=fsdp_actor_cfg.model, nccl_timeout=600
            )
            worker = FSDPActorWorker(actor_cfg, role="actor")
    else:
        ds_engine_cfg = DeepSpeedEngineConfig(
            ulysses_sequence_parallel_size=sp_size,
            model_dtype="bf16" if dtype == torch.bfloat16 else "fp32",
            param_offload=False,
            optimizer_offload=False,
            mixed_precision=mp_conf,
        )
        ds_model_cfg = HFModelConfig(
            path=model_name,
            tokenizer_path=model_name,
            use_remove_padding=True,
            trust_remote_code=True,
            enable_gradient_checkpointing=False,
            override_config={"attn_implementation": "eager"},
        )
        # Pass global mini batch and let worker normalize by DP/SP, matching FSDP defaults.
        per_rank_mini_batch = batch_size

        ds_cfg = DeepSpeedCriticConfig(
            strategy="deepspeed",
            zero_stage=0,
            gradient_accumulation_steps=grad_accum_override,
            ppo_mini_batch_size=per_rank_mini_batch,
            ppo_micro_batch_size_per_gpu=micro_bsz_per_gpu,
            ppo_epochs=1,
            rollout_n=1,
            ppo_max_token_len_per_gpu=total_len,
            ulysses_sequence_parallel_size=sp_size,
            grad_clip=1e9,
            model=ds_model_cfg,
            optim=FSDPOptimizerConfig(
                lr=5e-4,
                weight_decay=0.0,
                betas=(0.9, 0.999),
                optimizer="AdamW",
                optimizer_impl="torch.optim",
                grad_clip=1e9,
            ),
            deepspeed_config=ds_engine_cfg,
        )
        if role == "critic":
            worker = DSCriticWorker(ds_cfg)
        else:
            ds_actor_cfg = DeepSpeedActorConfig(
                strategy="deepspeed",
                zero_stage=0,
                gradient_accumulation_steps=grad_accum_override,
                ppo_mini_batch_size=per_rank_mini_batch,
                ppo_micro_batch_size_per_gpu=micro_bsz_per_gpu,
                ppo_epochs=1,
                rollout_n=1,
                ppo_max_token_len_per_gpu=total_len,
                ulysses_sequence_parallel_size=sp_size,
                grad_clip=1e9,
                optim=OptimizerConfig(lr=5e-4, weight_decay=0.0, betas=(0.9, 0.999), grad_clip=1e9),
                deepspeed_config=ds_engine_cfg,
            )
            ds_actor_cfg.model = ds_model_cfg
            from types import SimpleNamespace

            actor_cfg = SimpleNamespace(actor=ds_actor_cfg, rollout=SimpleNamespace(n=1), model=ds_actor_cfg.model, nccl_timeout=600)
            worker = DSActorWorker(actor_cfg, role="actor")

    worker.init_model()
    # No explicit cross-impl weight sync; rely on normal initialization paths
    init_hash = {}

    if deterministic_train and role == "critic":
        worker.critic_module.eval()
        for m in worker.critic_module.modules():
            if hasattr(m, "p"):
                try:
                    m.p = 0.0
                except Exception:
                    pass
    if deterministic_train and role == "actor":
        module = worker.actor_module_fsdp if impl == "fsdp" else worker.actor_module
        module.eval()
        for m in module.modules():
            if hasattr(m, "p"):
                try:
                    m.p = 0.0
                except Exception:
                    pass

    tokenizer = None
    global_batch_cpu = None
    if real_data and dist.get_rank() == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if dist.get_rank() == 0:
        batch = _build_batch(
            batch_size,
            prompt_len,
            response_len,
            dtype=dtype,
            seed=seed,
            role=role,
            real_data=real_data,
            real_case=real_case,
            real_case_id=real_case_id,
            tokenizer=tokenizer,
        )
        global_batch_cpu = _clone_to_cpu_dict(batch.batch)
    else:
        dummy = {
            "input_ids": torch.empty(batch_size, total_len, dtype=torch.long),
            "attention_mask": torch.empty(batch_size, total_len, dtype=torch.long),
            "position_ids": torch.empty(batch_size, total_len, dtype=torch.long),
            "prompts": torch.empty(batch_size, prompt_len, dtype=torch.long),
            "responses": torch.empty(batch_size, response_len, dtype=torch.long),
            "response_mask": torch.empty(batch_size, response_len, dtype=torch.float32),
        }
        if role == "critic":
            dummy.update(
                {
                    "values": torch.empty(batch_size, response_len, dtype=dtype),
                    "returns": torch.empty(batch_size, response_len, dtype=dtype),
                }
            )
        else:
            dummy.update(
                {
                    "old_log_probs": torch.empty(batch_size, response_len, dtype=dtype),
                    "advantages": torch.empty(batch_size, response_len, dtype=dtype),
                    "dp_size": torch.empty(batch_size, dtype=torch.int32),
                    "batch_num_tokens": torch.empty(batch_size, dtype=torch.int32),
                    "global_batch_size": torch.empty(batch_size, dtype=torch.int32),
                }
            )
        batch = DataProto.from_dict(
            tensors=dummy,
            meta_info={
                "global_token_num": [response_len for _ in range(batch_size)],
                "rng_seed": seed,
                "temperature": 1.0,
            },
        )
    batch = _broadcast_batch(batch, device=device)
    # attach parity info for worker-side dump
    batch.meta_info["parity_out_dir"] = out_dir
    batch.meta_info["parity_dump_steps"] = dump_steps
    batch.meta_info["parity_dump"] = os.environ.get("PARITY_DUMP", "0")
    batch.meta_info["impl"] = impl
    if role == "actor":
        batch.batch["dp_size"] = torch.full((batch_size,), dp_size, device=device, dtype=torch.int32)
        per_sample_tokens = batch.batch["response_mask"].sum(dim=-1).to(torch.int32)
        batch.batch["batch_num_tokens"] = per_sample_tokens
        batch.batch["global_batch_size"] = torch.full((batch_size,), batch_size, device=device, dtype=torch.int32)
    checksum = {
        "local_input_ids_sum": float(batch.batch["input_ids"].sum().item()),
        "local_attn_sum": float(batch.batch["attention_mask"].sum().item()),
        "local_mask_sum": float(batch.batch["response_mask"].sum().item()),
    }
    if dp_size == 1:
        checksum_tensor = torch.tensor(
            [checksum["local_input_ids_sum"], checksum["local_attn_sum"], checksum["local_mask_sum"]],
            device=device,
            dtype=torch.float64,
        )
        checksum_gather = [torch.zeros_like(checksum_tensor) for _ in range(world)]
        dist.all_gather(checksum_gather, checksum_tensor)
        if rank == 0:
            base = checksum_gather[0]
            for idx, other in enumerate(checksum_gather[1:], start=1):
                if not torch.allclose(other, base):
                    raise RuntimeError(f"checksum mismatch across ranks at rank {idx}: {other} vs {base}")

    before_state = (
        _gather_fsdp_state_dict(worker.critic_module if role == "critic" else worker.actor_module_fsdp)
        if impl == "fsdp"
        else _gather_ds_state_dict(worker.critic_module if role == "critic" else worker.actor_module)
    )
    model_for_hook = _unwrap_model(worker, role, impl)
    init_param_hash = _param_hash(model_for_hook, PROBE_NAMES)
    hook_store: Dict[str, torch.Tensor | dict | None] = _init_hook_store()
    hook_handles = []
    if dump_attn_hook:
        _register_attn_hooks(model_for_hook, hook_store, hook_handles, rank)
    chain_store: Dict[str, torch.Tensor] = {}
    chain_handles: list = []
    if dump_forward_chain:
        _register_forward_chain_hooks(model_for_hook, chain_store, chain_handles, rank)

    metrics_history = []
    debug_tensors = {}
    try:
        values = None
        if role == "critic":
            values = worker.compute_values(batch).batch["values"].detach()
        for step_idx in range(steps):
            _seed_everything(seed + step_idx)
            batch.meta_info["step_id"] = step_idx
            if role == "critic":
                metrics = worker.update_critic(batch)
            else:
                metrics = worker.update_actor(batch)
            if isinstance(metrics, DataProto):
                metrics = metrics.meta_info.get("metrics", {})
            metrics = dict(metrics)
            mod = worker.critic_module if role == "critic" else (worker.actor_module_fsdp if impl == "fsdp" else worker.actor_module)
            grad_stats = _compute_grad_stats(mod)
            metrics.update(
                {
                    "debug/grad_norm": grad_stats["grad_norm"],
                    "debug/grad_none": grad_stats["grad_none"],
                    "debug/grad_topk": grad_stats["grad_topk"],
                }
            )
            if role == "critic":
                with torch.no_grad():
                    vpreds = worker.compute_values(batch).batch["values"].detach()
                    returns = batch.batch["returns"].to(vpreds.device)
                    old_values = batch.batch["values"].to(vpreds.device)
                    response_mask = batch.batch["response_mask"].to(vpreds.device)
                    num = ((vpreds - returns) ** 2 * response_mask).sum()
                    denom = response_mask.sum()
                    vf_token_mean = 0.5 * num / denom
                    micro_divisor = max(1, batch_size // sp_size)
                    grad_accum = max(1, (batch_size * world // sp_size) // micro_divisor)
                    loss_for_backward = vf_token_mean / grad_accum
                    metrics.update(
                        {
                            "unified/raw_num": num.detach().cpu(),
                            "unified/raw_denom": denom.detach().cpu(),
                            "unified/vf_token_mean": vf_token_mean.detach().cpu(),
                            "unified/loss_for_backward": loss_for_backward.detach().cpu(),
                            "unified/grad_accum": grad_accum,
                        }
                    )
                    if os.environ.get("PARITY_DUMP_TENSORS", "0") == "1" and dist.get_rank() == 0:
                        debug_tensors = {
                            "vpreds": vpreds.detach().cpu(),
                            "returns": returns.detach().cpu(),
                            "values": old_values.detach().cpu(),
                            "response_mask": response_mask.detach().cpu(),
                        }
            metrics_history.append(metrics)
    except Exception as exc:
        print(f"[rank{rank}] runner failure: {exc}", flush=True)
        success = torch.tensor(0, device=device)
        dist.all_reduce(success)
        raise
    else:
        success = torch.tensor(1, device=device)
        dist.all_reduce(success)
        if success.item() != dist.get_world_size():
            raise RuntimeError("Some ranks failed during run")
    forward_chain_dump = None
    if dump_forward_chain:
        forward_chain_dump = {"impl": impl, "role": role, "rank": rank, "sp_rank": rank % sp_size, "dp_rank": rank // sp_size}
        if chain_store.get("embed_out") is not None:
            embed_out = chain_store["embed_out"]
            forward_chain_dump["embed_out"] = _make_tensor_entry(embed_out, token_dim=1)
            forward_chain_dump["pre_pack_hidden"] = _make_tensor_entry(embed_out, token_dim=1)
            attn_mask_cpu = batch.batch["attention_mask"].detach().cpu()
            flat = embed_out.reshape(-1, embed_out.shape[-1])
            mask_flat = attn_mask_cpu.reshape(-1)
            if flat.shape[0] == mask_flat.numel():
                indices = torch.nonzero(mask_flat, as_tuple=False).reshape(-1)
                if indices.numel() > 0 and indices.max() < flat.shape[0]:
                    packed = flat.index_select(0, indices)
                    forward_chain_dump["post_pack_hidden"] = _make_tensor_entry(packed, token_dim=0)
                    forward_chain_dump["token_map"] = indices[:64].detach().cpu()
                    pad_size = (sp_size - (packed.shape[0] % sp_size)) % sp_size
                    forward_chain_dump["pad_size"] = pad_size
                    if pad_size > 0:
                        pad_shape = list(packed.shape)
                        pad_shape[0] = pad_size
                        pad = torch.zeros(pad_shape, dtype=packed.dtype)
                        packed_for_sp = torch.cat([packed, pad], dim=0)
                    else:
                        packed_for_sp = packed
                    shards = torch.chunk(packed_for_sp, sp_size, dim=0)
                    shard = shards[rank % sp_size]
                    forward_chain_dump["post_ulysses_scatter"] = _make_tensor_entry(shard, token_dim=0)
            else:
                # fallback: treat embed_out as already packed (rmpad) and chunk
                packed = embed_out.squeeze(0) if embed_out.dim() > 1 else embed_out
                forward_chain_dump["post_pack_hidden"] = _make_tensor_entry(packed, token_dim=0)
                pad_size = (sp_size - (packed.shape[0] % sp_size)) % sp_size
                forward_chain_dump["pad_size"] = pad_size
                forward_chain_dump["token_map"] = None
                if pad_size > 0:
                    pad_shape = list(packed.shape)
                    pad_shape[0] = pad_size
                    pad = torch.zeros(pad_shape, dtype=packed.dtype)
                    packed_for_sp = torch.cat([packed, pad], dim=0)
                else:
                    packed_for_sp = packed
                shards = torch.chunk(packed_for_sp, sp_size, dim=0)
                shard = shards[rank % sp_size]
                forward_chain_dump["post_ulysses_scatter"] = _make_tensor_entry(shard, token_dim=0)
                if chain_store.get("layer0_attn_in") is None:
                    chain_store["layer0_attn_in"] = shard.detach().cpu()
        for key_name, tensor_name in [
            ("layer0_attn_in", "layer0_attn_in"),
            ("vproj_out", "vproj_out"),
            ("layer0_attn_out", "layer0_attn_out"),
            ("layer0_mlp_in", "layer0_mlp_in"),
            ("layer0_mlp_out", "layer0_mlp_out"),
        ]:
            if tensor_name in chain_store:
                forward_chain_dump[key_name] = _make_tensor_entry(chain_store.get(tensor_name), token_dim=1)

    after_state = (
        _gather_fsdp_state_dict(worker.critic_module if role == "critic" else worker.actor_module_fsdp)
        if impl == "fsdp"
        else _gather_ds_state_dict(worker.critic_module if role == "critic" else worker.actor_module)
    )

    dump_opt_state = os.environ.get("PARITY_DUMP_OPT", "0") == "1"
    opt_state = None
    if dump_opt_state:
        opt = None
        if role == "critic":
            opt = worker.critic_optimizer
        elif role == "actor":
            opt = worker.actor_optimizer if impl == "fsdp" else worker.actor_optimizer
        try:
            if opt is not None:
                opt_state = opt.state_dict()
        except Exception:
            opt_state = None

    if dist.get_rank() == 0:
        keys = _choose_keys(set(before_state.keys()) & set(after_state.keys())) if before_state and after_state else []
        delta = {k: (after_state[k] - before_state[k]).cpu() for k in keys} if keys else {}
        torch.save(
            {
                "impl": impl,
                "values": values.cpu() if values is not None else torch.tensor(0),
                "before_state": before_state,
                "after_state": after_state,
                "delta": delta,
                "metrics": metrics_history,
                "steps": steps,
                "debug_tensors": debug_tensors,
                "optimizer_state": opt_state,
            },
            os.path.join(out_dir, f"rank{dist.get_rank()}.pt"),
        )
    if dump_sp_io:
        # detach metrics to cpu for reproducible dump
        metrics_history_cpu: list[dict] = []
        for entry in metrics_history:
            metrics_history_cpu.append(
                {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in entry.items()}
            )
        local_cpu = _clone_to_cpu_dict(batch.batch)
        for key in ("input_ids", "attention_mask", "position_ids"):
            if key in local_cpu:
                local_cpu[f"hash_{key}"] = _hash_tensor(local_cpu[key])
        run_dir = run_id if run_id.startswith("sp_io_") else f"sp_io_{run_id}"
        sp_dir = os.path.join(out_root, run_dir, impl)
        os.makedirs(sp_dir, exist_ok=True)
        dump_obj = {
            "impl": impl,
            "role": role,
            "world": world,
            "sp_size": sp_size,
            "dp_size": dp_size,
            "rank": rank,
            "sp_rank": rank % sp_size,
            "dp_rank": rank // sp_size,
            "global": _clone_to_cpu_dict(global_batch_cpu) if rank == 0 else None,
            "local": local_cpu,
            "hooks": hook_store,
            "checksums": checksum,
            "init_param_hash": init_param_hash,
            "metrics": metrics_history_cpu,
        }
        out_path = os.path.join(sp_dir, f"rank{rank}_sp_io.pt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if dist.get_rank() == 0:
            torch.save(dump_obj, out_path)
            dist.barrier()
        else:
            dist.barrier()
            torch.save(dump_obj, out_path)
    if dump_forward_chain and forward_chain_dump is not None:
        run_dir = run_id if run_id.startswith("sp_io_") else f"sp_io_{run_id}"
        chain_dir = os.path.join(out_root, run_dir, impl)
        os.makedirs(chain_dir, exist_ok=True)
        forward_chain_dump["init_param_hash"] = init_param_hash
        forward_chain_dump["checksums"] = checksum
        out_path = os.path.join(chain_dir, f"rank{rank}_fwd_chain.pt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if dist.get_rank() == 0:
            torch.save(forward_chain_dump, out_path)
            dist.barrier()
        else:
            dist.barrier()
            torch.save(forward_chain_dump, out_path)
    for h in hook_handles:
        h.remove()
    for h in chain_handles:
        h.remove()
    dist.barrier()


if __name__ == "__main__":
    main()
