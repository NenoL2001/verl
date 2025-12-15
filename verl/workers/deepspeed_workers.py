# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Clean DeepSpeed-based Workers for PPO Training (Native DeepSpeed API)

This module provides worker implementations using native DeepSpeed API,
similar to how FSDP workers use native PyTorch FSDP API.
"""

import asyncio
import datetime
import logging
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Dict, List, Any, Tuple

import psutil
import torch

# Performance optimization: conditionally enable debug logging
_DEBUG_ENABLED = os.getenv("VERL_DEBUG", "0") == "1"


def _get_or_create_event_loop():
    """Return a usable asyncio event loop.

    - If no current loop, create and set a new one (prefer uvloop when available).
    - If the existing loop is closed (e.g., uvloop policy teardown), recreate it.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No current event loop; set policy and create one
        try:
            import uvloop  # type: ignore

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except Exception:
            pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
import torch.distributed
from tensordict import TensorDict
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.third_party.vllm import vllm_version
from verl.utils import hf_processor, hf_tokenizer
from verl.workers.deepspeed_parallel import (
    ParallelLayout,
    build_parallel_layout,
    normalize_actor_batches,
    normalize_critic_batches,
)
from verl.workers.critic.dp_critic import _hash_tensor, _flatten_slice, _step_enabled, _dump_debug
from verl.utils.checkpoint.deepspeed_checkpoint_manager import DeepSpeedCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.deepspeed_utils import (
    get_deepspeed_config,
    initialize_deepspeed_engine,
    load_deepspeed_model_to_gpu,
    offload_deepspeed_model_to_cpu,
)
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import (
    compute_position_id_with_mask,
    convert_weight_keys,
    get_generation_config,
    load_valuehead_model,
    print_model_size,
    update_model_config,
)
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.profiler.performance import reduce_timing, simple_timer, topk_reduce_ratio_min_max
from verl.utils.py_functional import append_to_dict, convert_to_regular_types
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.workers.config import DeepSpeedCriticConfig, DeepSpeedEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.actor import DataParallelPPOActor
from verl.workers.critic import DataParallelPPOCritic
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.deepspeed_ulysses import DeepSpeedUlyssesShardingManager
from verl.utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def _parse_mixed_precision_config(mixed_precision):
    """Parse mixed_precision config to determine fp16/bf16 flags.

    Args:
        mixed_precision: Can be None, str ("fp16"/"bf16"), or dict {"param_dtype": "bf16", ...}

    Returns:
        tuple: (fp16_enabled, bf16_enabled)
    """
    if mixed_precision is None:
        return False, False
    elif isinstance(mixed_precision, str):
        return mixed_precision == "fp16", mixed_precision == "bf16"
    elif isinstance(mixed_precision, dict):
        param_dtype = mixed_precision.get("param_dtype", "fp32")
        return param_dtype == "fp16", param_dtype == "bf16"
    else:
        return False, False


def _derive_ds_batch_params(per_rank_mini: int, micro_bsz: int, dp_size: int) -> Tuple[int, int]:
    """
    Match FSDP semantics: per-rank mini/micro are already normalized by DP/SP.
    Returns (grad_accum_steps, train_batch_size).
    """
    if per_rank_mini <= 0 or micro_bsz <= 0:
        raise ValueError(f"Invalid mini ({per_rank_mini}) or micro ({micro_bsz}) batch size")
    if per_rank_mini % micro_bsz != 0:
        raise ValueError(f"per-rank mini {per_rank_mini} must be divisible by micro {micro_bsz}")
    grad_accum = per_rank_mini // micro_bsz
    # train_batch_size should only scale with DP (not SP) to mirror FSDP divisor
    train_batch = micro_bsz * grad_accum * max(1, dp_size)
    return grad_accum, train_batch




class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    Clean DeepSpeed-based worker using native DeepSpeed API.

    Similar to FSDP worker structure but uses DeepSpeed for training.
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        self.actor_sharding_manager = None
        self.ref_sharding_manager = None
        self.actor_layout: ParallelLayout | None = None
        self.ref_layout: ParallelLayout | None = None

        rollout_cfg = self.config.get("rollout", {}) if isinstance(self.config, DictConfig) else {}
        self._skip_rollout = rollout_cfg.get("skip_rollout", False)
        load_format = rollout_cfg.get("load_format", "")
        self._dummy_rollout = isinstance(load_format, str) and load_format.startswith("dummy")

        # Ensure CUDA device is bound to LOCAL_RANK before init_process_group
        try:
            if torch.cuda.is_available():
                local_rank_env = os.environ.get("LOCAL_RANK")
                if local_rank_env is not None:
                    lr = int(local_rank_env)
                    if torch.cuda.current_device() != lr:
                        torch.cuda.set_device(lr)
        except Exception:
            pass

        # Initialize distributed environment
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                device_id=get_device_id() if torch.cuda.is_available() else None,
            )

        # Parse role
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # Setup profiler
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(f"Invalid role {self.role}")

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        # Ensure the CUDA device matches LOCAL_RANK to avoid NCCL hangs in new groups
        try:
            if torch.cuda.is_available():
                local_rank_env = os.environ.get("LOCAL_RANK")
                if local_rank_env is not None:
                    lr = int(local_rank_env)
                    if torch.cuda.current_device() != lr:
                        torch.cuda.set_device(lr)
        except Exception:
            pass

        # Setup offload flags
        self._is_offload_param = False
        if self._is_actor:
            deepspeed_config = self.config.actor.deepspeed_config
            self._is_offload_param = deepspeed_config.get("param_offload", False)
        elif self._is_ref:
            deepspeed_config = self.config.ref.deepspeed_config
            self._is_offload_param = deepspeed_config.get("param_offload", False)

        # Build parallel layouts and normalize configs
        if self._is_actor:
            tp_size = rollout_cfg.get("tensor_model_parallel_size", 1) if isinstance(rollout_cfg, dict) else 1
            self.actor_layout = build_parallel_layout(self.config.actor, tp_size=tp_size)
            normalize_actor_batches(
                self.config.actor, self.config.rollout.n, self.actor_layout.dp_size, sp_size=self.actor_layout.sp_size
            )
            self.actor_ulysses_sequence_parallel_size = self.actor_layout.sp_size
            self.ulysses_sequence_parallel_size = self.actor_layout.sp_size  # backward compat
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.actor_layout.dp_rank, is_collect=self.actor_layout.collect
            )
        else:
            self.actor_ulysses_sequence_parallel_size = 1
            self.ulysses_sequence_parallel_size = 1

        if self._is_ref:
            self.ref_layout = build_parallel_layout(self.config.ref)
            self.ref_ulysses_sequence_parallel_size = self.ref_layout.sp_size
            self._register_dispatch_collect_info(
                "ref", dp_rank=self.ref_layout.dp_rank, is_collect=self.ref_layout.collect
            )
        else:
            self.ref_ulysses_sequence_parallel_size = 1

        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0

    def _build_model_optimizer(
        self,
        model_path: str,
        deepspeed_config: DeepSpeedEngineConfig,
        optim_config: Optional[dict],
        override_model_config: dict,
        use_remove_padding: bool = False,
        use_fused_kernels: bool = False,
        enable_gradient_checkpointing: bool = False,
        trust_remote_code: bool = False,
        use_liger: bool = False,
        role: str = "actor",
        layout: ParallelLayout | None = None,
    ):
        """
        Build model and optimizer using native DeepSpeed API.

        Returns:
            tuple: (deepspeed_engine, model, optimizer, lr_scheduler, model_config)
        """
        # Load model config
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )

        # Override config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)

        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # Determine torch dtype
        torch_dtype = deepspeed_config.get("model_dtype", "fp32")
        if torch_dtype == "fp32":
            torch_dtype = torch.float32
        elif torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Create model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)

            # Apply monkey patches
            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            # Initialize Ulysses SP group for DeepSpeed-HF if requested
            if layout is not None:
                sp_size = layout.sp_size
            elif role == "actor":
                sp_size = self.actor_ulysses_sequence_parallel_size
                self.ulysses_sequence_parallel_size = sp_size
            elif role == "ref":
                sp_size = self.ref_ulysses_sequence_parallel_size
            else:
                sp_size = int(getattr(deepspeed_config, "ulysses_sequence_parallel_size", 1) or 1)
            sp_group = None
            prev_sp_group = get_ulysses_sequence_parallel_group()
            if sp_size > 1 and torch.distributed.is_initialized():
                # Use layout to build per-DP SP group to avoid cross-role pollution
                if layout is None:
                    world = torch.distributed.get_world_size()
                    assert world % sp_size == 0, f"world_size {world} must be divisible by ulysses sp_size {sp_size}"
                    rank = torch.distributed.get_rank()
                    group_id = rank // sp_size
                    ranks = list(range(group_id * sp_size, (group_id + 1) * sp_size))
                else:
                    ranks = list(
                        range(layout.dp_rank * layout.sp_size, (layout.dp_rank + 1) * layout.sp_size)
                    )
                sp_group = torch.distributed.new_group(ranks=ranks, backend=get_nccl_backend())
                set_ulysses_sequence_parallel_group(sp_group)
                # synchronize all ranks inside the SP group before patching
                torch.distributed.barrier(group=sp_group)

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=sp_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            actor_module.to(torch_dtype)

            # Gradient checkpointing
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            # LoRA
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        if sp_group is not None:
            torch.distributed.barrier(group=sp_group)
        set_ulysses_sequence_parallel_group(prev_sp_group)

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        if role == "actor":
            self.actor_sharding_manager = DeepSpeedUlyssesShardingManager(sp_group)
        elif role == "ref":
            self.ref_sharding_manager = DeepSpeedUlyssesShardingManager(sp_group)

        # Initialize DeepSpeed
        if optim_config is not None and role == "actor":
            # Build DeepSpeed config
            # Parse mixed precision config (supports str or dict)
            fp16_enabled, bf16_enabled = _parse_mixed_precision_config(
                deepspeed_config.get("mixed_precision")
            )

            zero_stage = getattr(self.config.actor, "zero_stage", deepspeed_config.get("zero_stage", 2))

            dp_size = layout.dp_size if layout is not None else (
                torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            )
            world_size = layout.world_size if layout is not None else (
                torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            )
            sp_size = layout.sp_size if layout is not None else 1
            if world_size != max(1, dp_size) * max(1, sp_size):
                raise AssertionError(f"world_size({world_size}) must equal dp_size({dp_size}) * sp_size({sp_size})")
            per_rank_mini = self.config.actor.ppo_mini_batch_size
            micro_bsz = self.config.actor.get("ppo_micro_batch_size_per_gpu", 1) or 1
            ds_grad_accum, ds_train_batch_size = _derive_ds_batch_params(per_rank_mini, micro_bsz, dp_size)
            # Persist into config for downstream logging/debug consistency
            try:
                self.config.actor.deepspeed_config["train_batch_size"] = ds_train_batch_size
                self.config.actor.deepspeed_config["gradient_accumulation_steps"] = ds_grad_accum
            except Exception:
                pass

            if self.rank == 0:
                print(
                    f"[ds-actor-config] sp_size={sp_size}, dp_size={dp_size}, world_size={world_size}, "
                    f"per_rank_mini={per_rank_mini}, micro_bsz={micro_bsz}, grad_accum={ds_grad_accum}, "
                    f"train_batch_size={ds_train_batch_size}"
                )
                print(
                    f"[ds-actor-init] train_batch_size={ds_train_batch_size}, micro_bsz={micro_bsz}, "
                    f"grad_accum={ds_grad_accum}, ds_config_grad_accum={ds_config.get('gradient_accumulation_steps')}"
                )
                print(
                    f"[ds-actor-init] train_batch_size={ds_train_batch_size}, micro_bsz={micro_bsz}, "
                    f"grad_accum={ds_grad_accum}, ds_config_grad_accum={ds_config.get('gradient_accumulation_steps')}"
                )

            python_clip = bool(int(os.getenv("PARITY_DS_PYTHON_CLIP", "0")))
            ds_config = get_deepspeed_config(
                optimizer_type=optim_config.get("optimizer", "AdamW"),
                train_batch_size=ds_train_batch_size,
                train_micro_batch_size_per_gpu=micro_bsz,
                gradient_accumulation_steps=ds_grad_accum,
                zero_stage=zero_stage,
                lr=optim_config.get("lr", 1e-5),
                betas=optim_config.get("betas", [0.9, 0.999]),
                eps=optim_config.get("eps", 1e-8),
                weight_decay=optim_config.get("weight_decay", 0.01),
                fp16_enabled=fp16_enabled,
                bf16_enabled=bf16_enabled,
                cpu_offload=deepspeed_config.get("param_offload", False),
                offload_optimizer=deepspeed_config.get("optimizer_offload", False),
                gradient_clipping=0.0 if python_clip else self.config.actor.get("grad_clip", None),
            )
            if bool(int(os.getenv("PARITY_DS_TORCH_OPT", "0"))):
                ds_config["optimizer"]["params"]["torch_adam"] = True
                ds_config["optimizer"]["params"]["fused"] = False

            # Initialize DeepSpeed engine
            ds_engine, optimizer, _, lr_scheduler = initialize_deepspeed_engine(
                model=actor_module,
                config=ds_config,
                model_parameters=actor_module.parameters(),
            )

            return ds_engine, ds_engine.module, optimizer, lr_scheduler, actor_model_config
        else:
            # No optimizer for ref or rollout
            return None, actor_module, None, None, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        """Build rollout engine (vLLM/SGLang)."""

        # Initialize RNG snapshots (needed even for dummy mode)
        device = get_torch_device()
        self.torch_random_states = device.get_rng_state()
        self.gen_random_states = self.torch_random_states.clone()

        rollout_config = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)
        model_config = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        # Auto-fix: ensure vLLM TP divides model attention heads
        try:
            from transformers import AutoConfig as HF_AutoConfig
            from verl.utils.fs import copy_to_local

            local_path = copy_to_local(model_config.path, use_shm=model_config.get("use_shm", False))
            hf_cfg = HF_AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

            def _extract_num_heads(cfg):
                for name in ("num_attention_heads", "num_heads", "n_head"):
                    if hasattr(cfg, name) and isinstance(getattr(cfg, name), (int,)):  # type: ignore
                        return int(getattr(cfg, name))
                # nested text_config (e.g., some multimodal models)
                tc = getattr(cfg, "text_config", None)
                if tc is not None:
                    for name in ("num_attention_heads", "num_heads", "n_head"):
                        if hasattr(tc, name) and isinstance(getattr(tc, name), (int,)):
                            return int(getattr(tc, name))
                return None

            def _extract_num_kv_heads(cfg):
                for name in ("num_key_value_heads", "num_kv_heads"):
                    if hasattr(cfg, name) and isinstance(getattr(cfg, name), (int,)):
                        return int(getattr(cfg, name))
                tc = getattr(cfg, "text_config", None)
                if tc is not None:
                    for name in ("num_key_value_heads", "num_kv_heads"):
                        if hasattr(tc, name) and isinstance(getattr(tc, name), (int,)):
                            return int(getattr(tc, name))
                return None

            num_heads = _extract_num_heads(hf_cfg)
            num_kv_heads = _extract_num_kv_heads(hf_cfg)
            # Use DP world (not counting SP) for rollout TP validation when actor uses Ulysses
            world_size = (
                self.actor_layout.dp_size
                if self.actor_layout is not None
                else (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
            )
            req_tp = int(getattr(rollout_config, "tensor_model_parallel_size", 1) or 1)
            def _divides_all(tp):
                cond1 = (num_heads is None) or (num_heads % tp == 0)
                cond2 = (num_kv_heads is None) or (num_kv_heads % tp == 0)
                return cond1 and cond2

            if (num_heads is not None or num_kv_heads is not None) and (not _divides_all(req_tp) or req_tp > world_size):
                # pick the largest divisor of num_heads that does not exceed world_size
                divisors = [d for d in range(world_size, 0, -1) if _divides_all(d)]
                new_tp = divisors[0] if divisors else 1
                if self.rank == 0:
                    print(
                        f"[AutoFix] Adjust vLLM TP from {req_tp} to {new_tp} so that heads kv_heads divisible "
                        f"(world_size={world_size})."
                    )
                rollout_config.tensor_model_parallel_size = new_tp
        except Exception as e:  # best-effort; don't block rollout build
            if self.rank == 0:
                print(f"[AutoFix] Skip TP auto-adjust due to: {e}")

        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)

        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=None
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Register dispatch info so Ray routing knows how to gather rollout outputs
        if self.actor_layout is not None:
            self._register_dispatch_collect_info(
                "rollout", dp_rank=self.actor_layout.dp_rank, is_collect=self.actor_layout.collect
            )
        else:
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)

        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format

        # Switch to trainer mode for sync rollout
        if rollout_config.mode == "sync" and self._is_actor:
            loop = _get_or_create_event_loop()
            loop.run_until_complete(self.trainer_mode())


    async def rollout_mode(self):
        """Context switch to rollout mode."""
        if self._skip_rollout:
            # When rollout is skipped, no weight syncing is required.
            self.base_sync_done = True
            return

        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_deepspeed_model_to_gpu", logger=logger)
        if self._is_offload_param and self.actor_engine is not None:
            load_deepspeed_model_to_gpu(self.actor_engine)
        log_gpu_memory_usage("After load_deepspeed_model_to_gpu", logger=logger)

        # Get model parameters for rollout - ensure we get full tensors
        if self.actor_engine is not None:
            # DeepSpeed engine - need to handle ZeRO partitioned parameters
            # For ZeRO-2, weights are not partitioned, only optimizer states
            # For ZeRO-3, weights ARE partitioned and need gathering

            # Use deepspeed's method to get full state dict if available
            if hasattr(self.actor_engine, 'get_full_state_dict'):
                params = self.actor_engine.get_full_state_dict()
            else:
                # Fallback: get from module directly
                params = self.actor_engine.module.state_dict()

            # Log weight shapes for debugging
            if self.rank == 0:
                for key in list(params.keys())[:3]:  # Check first 3 weights
                    logger.info(f"DeepSpeed weight {key} shape: {params[key].shape}, dtype: {params[key].dtype}")
        else:
            params = self.actor_module.state_dict()

        # Critical: Convert weight keys to match vLLM expectations (like FSDP does)
        if self.actor_engine is not None:
            params = convert_weight_keys(params, self.actor_engine.module)
        else:
            params = convert_weight_keys(params, self.actor_module)

        log_gpu_memory_usage("Before offload_deepspeed_model_to_cpu", logger=logger)
        if self._is_offload_param and self.actor_engine is not None:
            offload_deepspeed_model_to_cpu(self.actor_engine)
        log_gpu_memory_usage("After offload_deepspeed_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        device = get_device_id()

        # Use generator like FSDP does - avoid eager list creation
        # This may help with memory management and weight format compatibility
        def _yield_params():
            for name, param in params.items():
                tensor = param.to(device, non_blocking=True)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                yield name, tensor

        per_tensor_param = _yield_params()

        # Ensure all transfers and memory operations are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Critical fix for DeepSpeed dummy mode compatibility
        # Unlike FSDP, DeepSpeed's weights in dummy mode cause CUDA errors in vLLM
        # Root cause: vLLM's dummy weight initialization is incompatible with DeepSpeed weight format
        # Solution: Skip weight update in dummy mode, let vLLM use its own dummy weights
        if self._dummy_rollout:
            # Dummy mode: Skip weight update entirely
            logger.info("Dummy mode: Skipping weight update, vLLM will use its own dummy-initialized weights")
            del params, per_tensor_param
            aggressive_empty_cache(force_sync=True)
            # Mark as synced to prevent future update attempts
            self.base_sync_done = True
        else:
            # Normal mode: Update weights as usual (this works for DeepSpeed)
            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=["weights"])
            log_gpu_memory_usage("After resume weights", logger=logger)

            await self.rollout.update_weights(per_tensor_param, base_sync_done=self.base_sync_done)
            log_gpu_memory_usage("After update_weights", logger=logger)
            del params, per_tensor_param
            aggressive_empty_cache(force_sync=True)

            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=["kv_cache"])
            log_gpu_memory_usage("After resume kv_cache", logger=logger)

            # Set base_sync_done to True after first sync
            if not self.base_sync_done:
                self.base_sync_done = True

        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch to trainer mode."""
        if self._skip_rollout:
            return

        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        if self.actor_engine is not None:
            self.actor_engine.module.train()
        else:
            self.actor_module.train()

        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(True)

        # Restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize models and engines."""
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        # Load local model path
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)

        # Initialize tokenizer/processor (before model creation)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Load generation config
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        # Build actor model with DeepSpeed
        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                deepspeed_config = omega_conf_to_dataclass(self.config.actor.deepspeed_config)
            else:
                optim_config = None
                deepspeed_config = DeepSpeedEngineConfig()

            (
                self.actor_engine,
                self.actor_module,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                deepspeed_config=deepspeed_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=trust_remote_code,
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                layout=self.actor_layout,
            )

            if self._is_offload_param and self.actor_engine is not None:
                offload_deepspeed_model_to_cpu(self.actor_engine)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

        # Build actor wrapper
        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DeepSpeedPPOActor(config=actor_cfg, actor_module=self.actor_module, engine=self.actor_engine)

        # Build rollout
        if self._is_rollout:
            self._build_rollout(trust_remote_code=trust_remote_code)

        # Build reference policy
        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)

            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            (
                self.ref_engine,
                self.ref_module,
                _,
                _,
                _,
            ) = self._build_model_optimizer(
                model_path=local_path,
                deepspeed_config=omega_conf_to_dataclass(self.config.ref.deepspeed_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=trust_remote_code,
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
                layout=self.ref_layout,
            )

            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module)

        # Create checkpoint manager and flops counter
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)

            # DeepSpeedCheckpointManager expects worker object (accesses self.engine.engine)
            # Store engine reference for checkpoint manager to access
            self.engine = self.actor_engine
            self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)

        if not self._is_actor and self._is_rollout:
            # Standalone rollout checkpoint manager (load only)
            # Store engine reference for checkpoint manager to access
            self.engine = self.actor_engine if self.actor_engine is not None else None
            if self.engine is not None:
                self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step: int = 0, max_ckpt_to_keep: int | None = None):
        """Save actor (DeepSpeed) checkpoint using native DeepSpeed format.

        Directory layout mirrors FSDP manager style: <root>/step_<global_step>/
        """
        import torch

        if not self._is_actor or self.actor_engine is None:
            return

        # Expose engine handle for checkpoint manager
        self.engine = self.actor_engine

        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)

        self.checkpoint_manager.save(
            root=local_path, global_step=global_step, hdfs_path=hdfs_path, max_ckpt_to_keep=max_ckpt_to_keep
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, step: int | None = None, del_local_after_load: bool = True):
        """Load actor (DeepSpeed) checkpoint saved by `save_checkpoint`."""
        import torch

        if not self._is_actor or self.actor_engine is None:
            return {}

        # Compatibility with trainer resume logic: when caller passes None, treat as no-op.
        # This mirrors FSDP manager behavior and avoids TypeError inside the DS checkpoint manager.
        if local_path is None or (isinstance(local_path, str) and local_path.strip() == ""):
            return {}

        # Expose engine handle for checkpoint manager
        self.engine = self.actor_engine

        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)

        state = self.checkpoint_manager.load(
            root=local_path, step=step, hdfs_path=None, del_local_after_load=del_local_after_load
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)

        return state

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        """Update actor policy using PPO."""
        assert self._is_actor
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)

        data = data.to("cpu")

        manager = self.actor_sharding_manager
        ctx = manager if manager is not None else nullcontext()
        with ctx:
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        """Generate sequences using vLLM/SGLang rollout."""
        assert self._is_rollout

        prompts = prompts.to(get_device_id())

        eos_id = (
            self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id
        )
        pad_id = (
            self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id
        )

        if prompts.meta_info is None:
            prompts.meta_info = {}
        prompts.meta_info.setdefault("eos_token_id", eos_id)
        prompts.meta_info.setdefault("pad_token_id", pad_id)

        timing_generate: dict[str, float] = {}

        # Critical fix for dummy mode: Skip actual generation, craft lightweight placeholder outputs
        # This keeps the PPO training stack exercised without invoking vLLM kernels.
        if self._dummy_rollout:
            import time
            # Dummy mode also needs mode switching for RNG consistency
            if self._is_actor:
                import asyncio
                loop = _get_or_create_event_loop()
                loop.run_until_complete(self.rollout_mode())

            start = time.perf_counter()

            idx = prompts.batch["input_ids"]
            batch_size = idx.size(0)
            device = idx.device

            if "attention_mask" in prompts.batch.keys():
                attention_mask = prompts.batch["attention_mask"]
            else:
                attention_mask = torch.ones_like(idx, dtype=torch.int64, device=device)

            if "position_ids" in prompts.batch.keys():
                position_ids = prompts.batch["position_ids"]
            else:
                seq_len = idx.size(-1)
                base_positions = torch.arange(seq_len, device=device, dtype=torch.int64)
                position_ids = base_positions.unsqueeze(0).expand_as(idx)

            eos_token_meta = prompts.meta_info.get("eos_token_id", eos_id)
            if isinstance(eos_token_meta, (list, tuple)):
                eos_token_value = eos_token_meta[0]
            elif isinstance(eos_token_meta, torch.Tensor):
                eos_token_value = eos_token_meta.view(-1)[0].item()
            elif eos_token_meta is None:
                eos_token_value = pad_id
            else:
                eos_token_value = eos_token_meta

            try:
                eos_token_value = int(eos_token_value)
            except (TypeError, ValueError):
                eos_token_value = int(pad_id)

            response_length = 1
            responses = torch.full(
                (batch_size, response_length), eos_token_value, dtype=idx.dtype, device=device
            )
            seq = torch.cat([idx, responses], dim=-1)

            delta_position_id = torch.arange(
                1, response_length + 1, device=position_ids.device, dtype=position_ids.dtype
            )
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            if position_ids.dim() == 3:  # e.g. mrope (batch, num_heads, seq_len)
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                    batch_size, position_ids.size(1), -1
                )
            response_position_ids = position_ids[..., -1:] + delta_position_id
            extended_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = torch.ones(
                (batch_size, response_length), dtype=attention_mask.dtype, device=attention_mask.device
            )
            extended_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)

            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": responses,
                    "input_ids": seq,
                    "attention_mask": extended_attention_mask,
                    "position_ids": extended_position_ids,
                },
                batch_size=batch_size,
            )

            if hasattr(prompts.batch, "get") and prompts.batch.get("rollout_log_probs") is not None:
                batch["rollout_log_probs"] = prompts.batch["rollout_log_probs"]

            non_tensor_batch = (
                prompts.non_tensor_batch.copy()
                if isinstance(prompts.non_tensor_batch, dict)
                else dict(prompts.non_tensor_batch or {})
            )
            non_tensor_batch.pop("raw_prompt_ids", None)
            non_tensor_batch.pop("multi_modal_data", None)

            timing_generate["generate_sequences"] = time.perf_counter() - start

            meta_info = dict(prompts.meta_info or {})
            timing_meta = dict(meta_info.get("timing", {}))
            timing_meta.update(timing_generate)
            meta_info["timing"] = timing_meta

            output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
            logger.info("Dummy mode: Skipped vLLM generation, returned prompts as placeholder responses")

            # Switch back to trainer mode after dummy generation
            if self._is_actor:
                import asyncio
                loop = _get_or_create_event_loop()
                loop.run_until_complete(self.trainer_mode())

            return output

        # Normal mode: Use vLLM for actual generation
        if self._is_actor:
            import asyncio
            loop = _get_or_create_event_loop()
            loop.run_until_complete(self.rollout_mode())

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            import asyncio
            loop = _get_or_create_event_loop()
            loop.run_until_complete(self.trainer_mode())

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )

        output.meta_info.setdefault("timing", {}).update(timing_generate)
        output = output.to("cpu")
        get_torch_device().empty_cache()

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)

        manager = self.actor_sharding_manager
        ctx = manager if manager is not None else nullcontext()

        with ctx:
            is_lora = data.meta_info.pop("is_lora", False)
            adapter_ctx = self.actor.disable_adapter() if is_lora else nullcontext()

            data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
            data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
            data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
            data.meta_info["temperature"] = self.config.rollout.temperature

            with adapter_ctx:
                log_probs, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)

            output = DataProto.from_dict(
                tensors={"old_log_probs": log_probs, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data

        assert self._is_ref

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        manager = self.ref_sharding_manager
        ctx = manager if manager is not None else nullcontext()
        with ctx:
            data = data.to("cpu")
            log_prob, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": log_prob})

        return output.to("cpu")


class DeepSpeedPPOActor(DataParallelPPOActor):
    """PPO actor that integrates DeepSpeed optimizer/ZeRO workflow."""

    def __init__(self, config, actor_module, engine):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=engine.optimizer)
        self.deepspeed_engine = engine
        self._use_manual_backward = False
        self._last_grad_layout: list[tuple[str, int]] = []
        assert (
            self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0
        ), (
            "ppo_mini_batch_size must be divisible by ppo_micro_batch_size_per_gpu "
            "and should already be per-rank (same semantics as FSDP)."
        )
        base_opt = getattr(engine.optimizer, "optimizer", None)
        if torch.distributed.get_rank() == 0:
            print(
                f"[DEBUG][DS Actor] optimizer={type(engine.optimizer).__name__}, "
                f"base_optimizer={type(base_opt).__name__ if base_opt else 'None'}, "
                f"use_manual_backward={self._use_manual_backward}"
            )

    def _get_grad_accum_steps(self) -> int:
        engine_attr = getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
        steps = engine_attr() if callable(engine_attr) else engine_attr
        if steps is None:
            steps = max(1, self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu)
        return steps

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        # Optional per-step RNG seed for reproducibility
        if "rng_seed" in data.meta_info:
            rng_seed = int(data.meta_info["rng_seed"])
            if torch.distributed.get_rank() == 0:
                print(f"[DS Actor] Setting RNG seed: {rng_seed}")
            torch.manual_seed(rng_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rng_seed)

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        tis_cap = getattr(self.config, "tis_imp_ratio_cap", 0)
        if tis_cap > 0:
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        if tis_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires `actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        meta = data.meta_info
        impl = meta.get("impl", "deepspeed") if isinstance(meta, dict) else "deepspeed"
        step_id = int(meta.get("step_id", 0)) if isinstance(meta, dict) else 0
        out_dir = meta.get("parity_out_dir") if isinstance(meta, dict) else None
        dump_this_step = _step_enabled(meta, step_id) if isinstance(meta, dict) else False
        debug_env = bool(int(os.getenv("PARITY_DS_MINI_DEBUG", "0")))
        force_zero_grad = bool(int(os.getenv("PARITY_DS_FORCE_ZERO_GRAD", "0")))
        debug_enabled = dump_this_step or debug_env
        trace_steps = bool(int(os.getenv("PARITY_DS_TRACE_STEPS", "0")))
        fp16_bucket_debug = bool(int(os.getenv("PARITY_DS_FP16_BUCKET_DEBUG", "0")))
        fp16_bucket_clear = bool(int(os.getenv("PARITY_DS_FP16_BUCKET_CLEAR", "0")))

        def _collect_fp16_bucket_stats(opt, label: str) -> Dict[str, Dict[str, float]]:
            stats: Dict[str, Dict[str, float]] = {}
            if opt is None:
                return stats
            try:
                buckets = getattr(opt, "fp16_groups_flat", None)
                if buckets:
                    for i, buf in enumerate(buckets[:3]):  # limit to first few to avoid heavy cost
                        if buf is None:
                            continue
                        stats[f"{label}_fp16_flat_{i}"] = {
                            "norm": float(torch.linalg.vector_norm(buf.float()).cpu()),
                            "max_abs": float(torch.max(buf.abs()).cpu()),
                        }
                main_grad = getattr(opt, "main_grad", None)
                if isinstance(main_grad, torch.Tensor):
                    stats[f"{label}_main_grad"] = {
                        "norm": float(torch.linalg.vector_norm(main_grad.float()).cpu()),
                        "max_abs": float(torch.max(main_grad.abs()).cpu()),
                    }
            except Exception:
                pass
            return stats

        def _clear_fp16_buckets(opt):
            if opt is None:
                return
            try:
                buckets = getattr(opt, "fp16_groups_flat", None)
                if buckets:
                    for buf in buckets:
                        if buf is not None:
                            buf.zero_()
                main_grad = getattr(opt, "main_grad", None)
                if isinstance(main_grad, torch.Tensor):
                    main_grad.zero_()
            except Exception:
                pass
        probe_pre = {}
        probe_grad: Dict[str, torch.Tensor] = {}
        probe_hooks: List = []
        input_dump = {}
        mini_debug_records: List[Dict] = []
        probes = self._probe_params() if debug_enabled else {}
        probe_subset_names = sorted(list(probes.keys()))[:3]
        probe_subset = {n: probes[n] for n in probe_subset_names}
        trace_steps = bool(int(os.getenv("PARITY_DS_TRACE_STEPS", "0")))

        def _hash_stats(t: torch.Tensor) -> Dict[str, float]:
            flat = t.detach().float().reshape(-1)[:4096]
            if flat.numel() == 0:
                return {"abs_sum": 0.0, "sq_sum": 0.0, "max_abs": 0.0}
            flat64 = flat.to(torch.float64)
            return {
                "abs_sum": float(torch.sum(flat64.abs()).cpu()),
                "sq_sum": float(torch.sum(flat64 * flat64).cpu()),
                "max_abs": float(torch.max(flat64.abs()).cpu()),
            }

        def _grad_state(params: Dict[str, torch.nn.Parameter]) -> Dict[str, Dict[str, float | str]]:
            state: Dict[str, Dict[str, float | str]] = {}
            for name, p in params.items():
                g = p.grad
                if g is None:
                    state[name] = {"state": "none"}
                else:
                    norm_val = float(torch.linalg.vector_norm(g.detach().float()).cpu())
                    max_abs = float(torch.max(g.detach().abs()).cpu())
                    state[name] = {"state": "zero" if norm_val == 0.0 else "nonzero", "norm": norm_val, "max_abs": max_abs}
            return state



        if debug_enabled:
            for name, p in probes.items():
                probe_pre[name] = _flatten_slice(p)
            probe_hooks = self._attach_grad_hooks(probes, probe_grad)
            _dump_debug(out_dir, impl, step_id, "pre_fwd", {"probe_pre": probe_pre})
        meta = data.meta_info
        impl = meta.get("impl", "deepspeed") if isinstance(meta, dict) else "deepspeed"
        step_id = int(meta.get("step_id", 0)) if isinstance(meta, dict) else 0
        out_dir = meta.get("parity_out_dir") if isinstance(meta, dict) else None
        dump_this_step = _step_enabled(meta, step_id) if isinstance(meta, dict) else False
        debug_env = bool(int(os.getenv("PARITY_DS_MINI_DEBUG", "0")))
        force_zero_grad = bool(int(os.getenv("PARITY_DS_FORCE_ZERO_GRAD", "0")))
        debug_enabled = dump_this_step or debug_env
        trace_steps = bool(int(os.getenv("PARITY_DS_TRACE_STEPS", "0")))
        fp16_bucket_debug = bool(int(os.getenv("PARITY_DS_FP16_BUCKET_DEBUG", "0")))
        fp16_bucket_clear = bool(int(os.getenv("PARITY_DS_FP16_BUCKET_CLEAR", "0")))
        probe_pre = {}
        probe_grad: Dict[str, torch.Tensor] = {}
        probe_hooks: List = []
        input_dump = {}
        mini_debug_records: List[Dict] = []
        probes = self._probe_params() if debug_enabled else {}
        probe_subset_names = sorted(list(probes.keys()))[:3]
        probe_subset = {n: probes[n] for n in probe_subset_names}

        def _hash_stats(t: torch.Tensor) -> Dict[str, float]:
            flat = t.detach().float().reshape(-1)[:4096]
            if flat.numel() == 0:
                return {"abs_sum": 0.0, "sq_sum": 0.0, "max_abs": 0.0}
            flat64 = flat.to(torch.float64)
            return {
                "abs_sum": float(torch.sum(flat64.abs()).cpu()),
                "sq_sum": float(torch.sum(flat64 * flat64).cpu()),
                "max_abs": float(torch.max(flat64.abs()).cpu()),
            }

        def _grad_state(params: Dict[str, torch.nn.Parameter]) -> Dict[str, Dict[str, float | str]]:
            state: Dict[str, Dict[str, float | str]] = {}
            for name, p in params.items():
                g = p.grad
                if g is None:
                    state[name] = {"state": "none"}
                else:
                    norm_val = float(torch.linalg.vector_norm(g.detach().float()).cpu())
                    max_abs = float(torch.max(g.detach().abs()).cpu())
                    state[name] = {"state": "zero" if norm_val == 0.0 else "nonzero", "norm": norm_val, "max_abs": max_abs}
            return state

        if dump_this_step:
            probes = self._probe_params()
            for name, p in probes.items():
                probe_pre[name] = _flatten_slice(p)
            probe_hooks = self._attach_grad_hooks(probes, probe_grad)
            _dump_debug(out_dir, impl, step_id, "pre_fwd", {"probe_pre": probe_pre})
        meta = data.meta_info
        impl = meta.get("impl", "deepspeed") if isinstance(meta, dict) else "deepspeed"
        step_id = int(meta.get("step_id", 0)) if isinstance(meta, dict) else 0
        out_dir = meta.get("parity_out_dir") if isinstance(meta, dict) else None
        dump_this_step = _step_enabled(meta, step_id) if isinstance(meta, dict) else False
        probe_pre = {}
        probe_grad: Dict[str, torch.Tensor] = {}
        probe_hooks: List = []
        input_dump = {}
        if dump_this_step:
            probes = self._probe_params()
            for name, p in probes.items():
                probe_pre[name] = _flatten_slice(p)
            probe_hooks = self._attach_grad_hooks(probes, probe_grad)
            _dump_debug(out_dir, impl, step_id, "pre_fwd", {"probe_pre": probe_pre})
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                sp_factor = max(1, self.ulysses_sequence_parallel_size)
                # enforce per-rank semantics: mini batch size should match config
                any_tensor = next(iter(mini_batch.batch.values()))
                actual_mini = any_tensor.shape[0]
                expected_mini = self.config.ppo_mini_batch_size
                if actual_mini != expected_mini:
                    raise AssertionError(
                        f"per-rank mini batch mismatch: actual {actual_mini} vs config {expected_mini} "
                        "(config should already be per-rank, aligned with FSDP semantics)"
                    )
                assert (
                    actual_mini % self.config.ppo_micro_batch_size_per_gpu == 0
                ), f"mini {actual_mini} not divisible by micro {self.config.ppo_micro_batch_size_per_gpu}"
                if torch.distributed.get_rank() == 0:
                    print(
                        f"[ds-critic-batch] actual_mini={actual_mini}, config_mini={expected_mini}, "
                        f"micro_bsz={self.config.ppo_micro_batch_size_per_gpu}, "
                        f"use_dynamic_bsz={self.config.use_dynamic_bsz}, "
                        f"micro_batches_expected={actual_mini // self.config.ppo_micro_batch_size_per_gpu}"
                    )
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    assert len(micro_batches) > 0, "micro_batches empty"

                grad_accum_steps = len(micro_batches)
                if torch.distributed.get_rank() == 0:
                    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                    dp_sz = max(1, world // max(1, self.ulysses_sequence_parallel_size))
                    print(
                        f"[ds-critic-batch] mini={actual_mini}, config_mini={expected_mini}, "
                        f"micro_bsz={self.config.ppo_micro_batch_size_per_gpu}, "
                        f"micro_batches={len(micro_batches)}, grad_accum={grad_accum_steps}, dp={dp_sz}"
                    )
                if hasattr(self.deepspeed_engine, "set_gradient_accumulation_steps"):
                    try:
                        self.deepspeed_engine.set_gradient_accumulation_steps(grad_accum_steps)
                    except Exception as exc:
                        if torch.distributed.get_rank() == 0:
                            print(f"[ds-actor-scale] set_gradient_accumulation_steps failed: {exc}")
                if torch.distributed.get_rank() == 0:
                    dp_sz = torch.distributed.get_world_size() // sp_factor if torch.distributed.is_initialized() else 1
                    print(
                        f"[ds-actor-scale] sp={sp_factor}, dp={dp_sz}, grad_accum={grad_accum_steps}"
                    )

                self.deepspeed_engine.zero_grad()

                for idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    rollout_log_probs = model_inputs.get("rollout_log_probs")
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Vanilla/GSP0/GPG interfaces expect optional `rollout_is_weights`.
                    # We do not compute off-policy weights here; pass None by default.
                    policy_loss_out = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=None,
                    )

                    if (
                        isinstance(policy_loss_out, tuple)
                        and len(policy_loss_out) == 2
                        and isinstance(policy_loss_out[1], dict)
                    ):
                        pg_loss, pg_metrics = policy_loss_out
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_out
                        pg_metrics = {
                            "actor/pg_clipfrac": pg_clipfrac,
                            "actor/ppo_kl": ppo_kl,
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower,
                        }

                    # Normalize metric values for logging
                    pg_metrics = {
                        k: (v.detach().item() if torch.is_tensor(v) else float(v))
                        for k, v in pg_metrics.items()
                    }

                    # Diagnostic print to catch NaNs early
                    def _stat(x):
                        if x is None:
                            return {"min": 0.0, "max": 0.0, "mean": 0.0, "nan": False, "inf": False}
                        x_flat = x.float().detach().reshape(-1)
                        return {
                            "min": float(torch.nan_to_num(x_flat.min(), nan=0.0)),
                            "max": float(torch.nan_to_num(x_flat.max(), nan=0.0)),
                            "mean": float(torch.nan_to_num(x_flat.mean(), nan=0.0)),
                            "nan": bool(torch.isnan(x_flat).any()),
                            "inf": bool(torch.isinf(x_flat).any()),
                        }

                    if torch.distributed.get_rank() == 0:
                        log_info = {
                            "log_prob": _stat(log_prob),
                            "old_log_prob": _stat(old_log_prob),
                            "advantages": _stat(advantages),
                            "pg_loss": _stat(pg_loss),
                            "entropy": _stat(entropy),
                        }
                        if not torch.isfinite(pg_loss).all() or not torch.isfinite(log_prob).all():
                            print(f"[ds-actor-nan] micro_idx={idx}, stats={log_info}")

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                        micro_batch_metrics = {
                            "actor/kl_loss": kl_loss.detach().item(),
                            "actor/kl_coef": self.config.kl_loss_coef,
                        }
                    else:
                        micro_batch_metrics = {}

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1.0 / grad_accum_steps

                    # 使用 DeepSpeed 内部 GAS 缩放，避免外部二次缩放影响其他特性
                    is_last_micro = idx == len(micro_batches) - 1
                    self.deepspeed_engine.set_gradient_accumulation_boundary(is_last_micro)
                    loss = policy_loss
                    self.deepspeed_engine.backward(loss, scale_wrt_gas=True)

                    # Collect metrics (记录按外部缩放的数值，便于与 FSDP 对齐)
                    metric_scale_factor = loss_scale_factor

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * metric_scale_factor,
                            "actor/grad_accum_steps": float(grad_accum_steps),
                            "actor/sp_size": float(sp_factor),
                            "actor/dp_size": float(dp_sz if 'dp_sz' in locals() else 1),
                            "actor/micro_batches_per_step": float(len(micro_batches)),
                            "actor/sp_loss_divisor": 1.0,
                            "actor/loss_scale_factor": loss_scale_factor,
                            "actor/token_sum_local": float(response_mask.sum().detach().cpu()),
                        }
                    )
                    micro_batch_metrics.update(pg_metrics)
                    append_to_dict(metrics, micro_batch_metrics)

                python_clip = bool(int(os.getenv("PARITY_DS_PYTHON_CLIP", "0")))
                # Prefer python clip in parity mode or zero_stage=0 to mimic FSDP norm computation.
                if python_clip or self.config.zero_stage == 0:
                    grad_norm_val = float(
                        torch.nn.utils.clip_grad_norm_(
                            self.actor_module.parameters(),
                            max_norm=self.config.grad_clip,
                            norm_type=2.0,
                        )
                    )
                else:
                    ds_zero = getattr(self.config, "zero_stage", 0)
                    if ds_zero and hasattr(self.deepspeed_engine, "get_global_grad_norm"):
                        try:
                            grad_norm_val = float(self.deepspeed_engine.get_global_grad_norm())
                        except Exception:
                            grad_norm_val = 0.0
                    else:
                        grad_norm_val = float(
                            torch.nn.utils.clip_grad_norm_(
                                self.actor_module.parameters(),
                                max_norm=self.config.grad_clip,
                                norm_type=2.0,
                            )
                        )
                append_to_dict(metrics, {"actor/grad_norm": grad_norm_val})

                # Step only once after all micro batches
                self.deepspeed_engine.step()

        self.deepspeed_engine.zero_grad()
        return metrics


class DeepSpeedPPOCritic(DataParallelPPOCritic):
    """PPO critic that delegates backward/step to a DeepSpeed engine."""

    def __init__(self, config, critic_module, engine, ds_config: dict | None = None, force_torch_optimizer: bool = False):
        super().__init__(config=config, critic_module=critic_module, critic_optimizer=engine.optimizer)
        self.deepspeed_engine = engine
        self._ds_config_dict = ds_config or {}
        # Always use DeepSpeed-managed backward/step; manual backward is disabled.
        self._use_manual_backward = False
        assert (
            self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0
        ), (
            "ppo_mini_batch_size must be divisible by ppo_micro_batch_size_per_gpu "
            "and should already be per-rank (same semantics as FSDP)."
        )
        if torch.distributed.get_rank() == 0:
            base_opt = getattr(engine.optimizer, "optimizer", None)
            print(
                f"[DEBUG][DS Critic] optimizer={type(engine.optimizer).__name__}, "
                f"base_optimizer={type(base_opt).__name__ if base_opt else 'None'}, "
                f"use_manual_backward={self._use_manual_backward}"
            )
            print(
                f"[ds-critic-config] train_batch_size={self._ds_config_dict.get('train_batch_size')}, "
                f"micro_bsz={self._ds_config_dict.get('train_micro_batch_size_per_gpu')}, "
                f"grad_accum={self._ds_config_dict.get('gradient_accumulation_steps')}, "
                f"dp_size={getattr(self, 'layout', None).dp_size if getattr(self, 'layout', None) else 'n/a'}, "
                f"world_size={getattr(self, 'layout', None).world_size if getattr(self, 'layout', None) else 'n/a'}"
            )

    def _get_grad_accum_steps(self) -> int:
        engine_attr = getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
        steps = engine_attr() if callable(engine_attr) else engine_attr
        if steps is None:
            steps = max(1, self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu)
        return steps

    def update_critic(self, data: DataProto):
        self.critic_module.train()

        if 'rng_seed' in data.meta_info:
            rng_seed = data.meta_info['rng_seed']
            if torch.distributed.get_rank() == 0:
                print(f"[DS Critic] Setting RNG seed: {rng_seed}")
            torch.manual_seed(rng_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rng_seed)

        metrics = {}
        meta = data.meta_info
        impl = meta.get("impl", "deepspeed") if isinstance(meta, dict) else "deepspeed"
        step_id = int(meta.get("step_id", 0)) if isinstance(meta, dict) else 0
        out_dir = meta.get("parity_out_dir") if isinstance(meta, dict) else None
        dump_this_step = _step_enabled(meta, step_id) if isinstance(meta, dict) else False
        debug_env = bool(int(os.getenv("PARITY_DS_MINI_DEBUG", "0")))
        force_zero_grad = bool(int(os.getenv("PARITY_DS_FORCE_ZERO_GRAD", "0")))
        debug_enabled = dump_this_step or debug_env
        trace_steps = bool(int(os.getenv("PARITY_DS_TRACE_STEPS", "0")))
        fp16_bucket_debug = bool(int(os.getenv("PARITY_DS_FP16_BUCKET_DEBUG", "0")))
        fp16_bucket_clear = bool(int(os.getenv("PARITY_DS_FP16_BUCKET_CLEAR", "0")))
        probe_pre = {}
        probe_grad: Dict[str, torch.Tensor] = {}
        probe_hooks: List = []
        input_dump = {}
        mini_debug_records: List[Dict] = []
        probes = self._probe_params() if debug_enabled else {}
        probe_subset_names = sorted(list(probes.keys()))[:3]
        probe_subset = {n: probes[n] for n in probe_subset_names}

        def _collect_fp16_bucket_stats(opt, label: str) -> Dict[str, Dict[str, float]]:
            stats: Dict[str, Dict[str, float]] = {}
            if opt is None:
                return stats
            try:
                buckets = getattr(opt, "fp16_groups_flat", None)
                if buckets:
                    for i, buf in enumerate(buckets[:3]):  # limit to first few to avoid heavy cost
                        if buf is None:
                            continue
                        stats[f"{label}_fp16_flat_{i}"] = {
                            "norm": float(torch.linalg.vector_norm(buf.float()).cpu()),
                            "max_abs": float(torch.max(buf.abs()).cpu()),
                        }
                main_grad = getattr(opt, "main_grad", None)
                if isinstance(main_grad, torch.Tensor):
                    stats[f"{label}_main_grad"] = {
                        "norm": float(torch.linalg.vector_norm(main_grad.float()).cpu()),
                        "max_abs": float(torch.max(main_grad.abs()).cpu()),
                    }
            except Exception:
                pass
            return stats

        def _clear_fp16_buckets(opt):
            if opt is None:
                return
            try:
                buckets = getattr(opt, "fp16_groups_flat", None)
                if buckets:
                    for buf in buckets:
                        if buf is not None:
                            buf.zero_()
                main_grad = getattr(opt, "main_grad", None)
                if isinstance(main_grad, torch.Tensor):
                    main_grad.zero_()
            except Exception:
                pass

        def _hash_stats(t: torch.Tensor) -> Dict[str, float]:
            flat = t.detach().float().reshape(-1)[:4096]
            if flat.numel() == 0:
                return {"abs_sum": 0.0, "sq_sum": 0.0, "max_abs": 0.0}
            flat64 = flat.to(torch.float64)
            return {
                "abs_sum": float(torch.sum(flat64.abs()).cpu()),
                "sq_sum": float(torch.sum(flat64 * flat64).cpu()),
                "max_abs": float(torch.max(flat64.abs()).cpu()),
            }

        def _grad_state(params: Dict[str, torch.nn.Parameter]) -> Dict[str, Dict[str, float | str]]:
            state: Dict[str, Dict[str, float | str]] = {}
            for name, p in params.items():
                g = p.grad
                if g is None:
                    state[name] = {"state": "none"}
                else:
                    norm_val = float(torch.linalg.vector_norm(g.detach().float()).cpu())
                    max_abs = float(torch.max(g.detach().abs()).cpu())
                    state[name] = {"state": "zero" if norm_val == 0.0 else "nonzero", "norm": norm_val, "max_abs": max_abs}
            return state

        if debug_enabled:
            for name, p in probes.items():
                probe_pre[name] = _flatten_slice(p)
            probe_hooks = self._attach_grad_hooks(probes, probe_grad)
            _dump_debug(out_dir, impl, step_id, "pre_fwd", {"probe_pre": probe_pre})

        select_keys = [
            "input_ids",
            "responses",
            "response_mask",
            "attention_mask",
            "position_ids",
            "values",
            "returns",
        ]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for epoch_idx in range(self.config.ppo_epochs):
            for mini_idx, mini_batch in enumerate(mini_batches):
                any_tensor = next(iter(mini_batch.batch.values()))
                actual_mini = any_tensor.shape[0]
                expected_mini = self.config.ppo_mini_batch_size
                if actual_mini != expected_mini:
                    raise AssertionError(
                        f"per-rank mini batch mismatch: actual {actual_mini} vs config {expected_mini} "
                        "(config should already be per-rank, aligned with FSDP semantics)"
                    )
                assert (
                    actual_mini % self.config.ppo_micro_batch_size_per_gpu == 0
                ), f"mini {actual_mini} not divisible by micro {self.config.ppo_micro_batch_size_per_gpu}"
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    assert len(micro_batches) > 0, "micro_batches empty"

                grad_accum_steps = len(micro_batches)
                if hasattr(self.deepspeed_engine, "set_gradient_accumulation_steps"):
                    try:
                        self.deepspeed_engine.set_gradient_accumulation_steps(grad_accum_steps)
                    except Exception as exc:
                        if torch.distributed.get_rank() == 0:
                            print(f"[ds-critic-scale] set_gradient_accumulation_steps failed: {exc}")
                sp_factor = max(1, self.ulysses_sequence_parallel_size)
                debug_gas = bool(int(os.getenv("PARITY_DEBUG_GAS", "0")))
                if debug_gas and torch.distributed.get_rank() == 0:
                    ga_before = None
                    try:
                        ga_before = (
                            self.deepspeed_engine.gradient_accumulation_steps()
                            if callable(getattr(self.deepspeed_engine, "gradient_accumulation_steps", None))
                            else getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
                        )
                    except Exception:
                        pass
                    print(
                        f"[ds-critic-gas] grad_accum_expected={grad_accum_steps}, engine_before={ga_before}, "
                        f"micro_batches={len(micro_batches)}, "
                        f"cfg.grad_accum={self.config.deepspeed_config.get('gradient_accumulation_steps')}, "
                        f"cfg.micro_bsz={self.config.ppo_micro_batch_size_per_gpu}, "
                        f"cfg.train_batch_size={self.config.deepspeed_config.get('train_batch_size')}"
                    )
                # Re-read engine GAS after possible override
                engine_ga = grad_accum_steps
                try:
                    engine_ga = (
                        self.deepspeed_engine.gradient_accumulation_steps()
                        if callable(getattr(self.deepspeed_engine, "gradient_accumulation_steps", None))
                        else getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
                    )
                except Exception:
                    pass
                if engine_ga != grad_accum_steps:
                    raise AssertionError(
                        f"Engine gradient_accumulation_steps {engine_ga} != expected {grad_accum_steps}; "
                        "check DeepSpeed config for overrides."
                    )
                if torch.distributed.get_rank() == 0:
                    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                    print(
                        f"[ds-critic-scale] sp={sp_factor}, dp={world // sp_factor}, "
                        f"grad_accum={grad_accum_steps}, sp_loss_divisor=1.0, use_sp_loss_scale=False"
                    )

                mini_debug: Dict[str, Any] = {"epoch": epoch_idx, "mini_idx": mini_idx}
                if debug_enabled:
                    mini_debug["pre_param_hash"] = {n: _hash_stats(p) for n, p in probe_subset.items()}
                    mini_debug["pre_param_slice"] = {n: _flatten_slice(p) for n, p in probe_subset.items()}
                    mini_debug["pre_grad_state"] = _grad_state(probe_subset)
                    if trace_steps:
                        mini_debug["ops"] = []
                    leaks = {n: v for n, v in mini_debug["pre_grad_state"].items() if v.get("state") == "nonzero"}
                    if leaks and torch.distributed.get_rank() == 0:
                        print(f"[ds-critic-grad-leak] epoch={epoch_idx}, mini={mini_idx}, leaks={leaks}")

                self.deepspeed_engine.zero_grad()

                if force_zero_grad and debug_enabled:
                    mini_debug["force_zero_grad"] = _grad_state(probe_subset)
                if trace_steps and debug_enabled:
                    mini_debug.setdefault("ops", []).append({"op": "zero_grad", "where": "pre_mini", "mini_idx": mini_idx})
                if fp16_bucket_debug and debug_enabled:
                    mini_debug["fp16_bucket_pre"] = _collect_fp16_bucket_stats(
                        getattr(self.deepspeed_engine, "optimizer", None), "pre"
                    )
                if fp16_bucket_clear and debug_enabled:
                    _clear_fp16_buckets(getattr(self.deepspeed_engine, "optimizer", None))
                    mini_debug.setdefault("ops", []).append({"op": "clear_fp16_buckets", "mini_idx": mini_idx})

                last_micro_inputs = None
                for idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    if dump_this_step and not input_dump:
                        input_dump = {
                            "input_ids": model_inputs["input_ids"][:2, :16].detach().cpu(),
                            "responses": model_inputs["responses"][:2, :16].detach().cpu(),
                            "attention_mask": model_inputs["attention_mask"][:2, :16].detach().cpu(),
                            "position_ids": model_inputs["position_ids"][:2, :16].detach().cpu(),
                            "values": values[:2, :16].detach().cpu(),
                            "returns": returns[:2, :16].detach().cpu(),
                            "response_mask": response_mask[:2, :16].detach().cpu(),
                            "hash_input_ids": _hash_tensor(model_inputs["input_ids"]),
                        }
                        # extra debug for broadcast consistency
                        input_dump["hash_attention_mask"] = _hash_tensor(model_inputs["attention_mask"])
                        input_dump["hash_position_ids"] = _hash_tensor(model_inputs["position_ids"])

                    vpreds = self._forward_micro_batch(model_inputs)
                    vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                        vpreds=vpreds,
                        values=values,
                        returns=returns,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1.0 / grad_accum_steps
                    if torch.distributed.get_rank() == 0:
                        try:
                            eng_ga = (
                                self.deepspeed_engine.gradient_accumulation_steps()
                                if callable(getattr(self.deepspeed_engine, "gradient_accumulation_steps", None))
                                else getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
                            )
                        except Exception:
                            eng_ga = None
                        dp_sz_eff = max(
                            1,
                            (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
                            // sp_factor,
                        )
                        print(
                            f"[ds-critic-loss-debug] micro_idx={idx}, grad_accum={grad_accum_steps}, "
                            f"eng_grad_accum={eng_ga}, dp_size={dp_sz_eff}, "
                            f"token_sum={float(response_mask.sum().detach().cpu())}, "
                            f"vf_loss={float(vf_loss.detach().cpu())}, "
                            f"loss_scale_factor={loss_scale_factor}"
                        )

                    # 使用 DS 内部 GAS 缩放（scale_wrt_gas=True），与早期实现保持一致
                    loss = vf_loss
                    is_last_micro = idx == len(micro_batches) - 1
                    self.deepspeed_engine.set_gradient_accumulation_boundary(is_last_micro)
                    self.deepspeed_engine.backward(loss, scale_wrt_gas=True)

                    if debug_enabled and is_last_micro:
                        if trace_steps:
                            mini_debug.setdefault("ops", []).append(
                                {"op": "backward", "where": "last_micro", "mini_idx": mini_idx, "micro_idx": idx}
                            )
                        grad_snapshot = {}
                        for name, p in probe_subset.items():
                            g = p.grad
                            if g is None:
                                continue
                            grad_snapshot[name] = {
                                "norm": float(torch.linalg.vector_norm(g.detach().float()).cpu()),
                                "max_abs": float(torch.max(g.detach().abs()).cpu()),
                                "slice": _flatten_slice(g, limit=256),
                            }
                        try:
                            boundary_flag = bool(self.deepspeed_engine.is_gradient_accumulation_boundary())
                        except Exception:
                            boundary_flag = "NA"
                        try:
                            engine_ga_now = (
                                self.deepspeed_engine.gradient_accumulation_steps()
                                if callable(getattr(self.deepspeed_engine, "gradient_accumulation_steps", None))
                                else getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
                            )
                        except Exception:
                            engine_ga_now = "NA"
                        mini_debug["post_backward"] = {
                            "micro_idx": idx,
                            "grad": grad_snapshot,
                            "boundary": boundary_flag,
                            "engine_ga": engine_ga_now,
                        }
                        # Extra verification: compare recorded loss/grad norm with actual tensors
                        verify = bool(int(os.getenv("PARITY_DS_VERIFY_METRICS", "0")))
                        if verify:
                            try:
                                torch_grad_norm = float(
                                    torch.nn.utils.clip_grad_norm_(
                                        self.critic_module.parameters(),
                                        max_norm=float("inf"),
                                        norm_type=2.0,
                                    )
                                )
                            except Exception:
                                torch_grad_norm = None
                            mini_debug["verify"] = {
                                "loss_item": float(loss.detach().cpu()),
                                "torch_grad_norm_infclip": torch_grad_norm,
                            }
                        if fp16_bucket_debug and debug_enabled:
                            mini_debug["fp16_bucket_post_backward"] = _collect_fp16_bucket_stats(
                                getattr(self.deepspeed_engine, "optimizer", None), "post_bwd"
                            )
                        last_micro_inputs = model_inputs

                    micro_batch_metrics = {
                        "critic/vf_loss": vf_loss.detach().item() * loss_scale_factor,
                        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                        "critic/grad_accum_steps": grad_accum_steps,
                        "critic/token_sum_local": float(response_mask.sum().detach().cpu()),
                        "critic/per_token_mse_raw": float(
                            torch.nan_to_num(((vpreds - returns) ** 2 * response_mask).sum(), nan=0.0)
                            / max(1.0, float(response_mask.sum().detach().cpu()))
                        ),
                        "critic/loss_scale_factor": loss_scale_factor,
                        "critic/loss_unscaled": float(vf_loss.detach().cpu()),
                        "critic/loss_for_backward": float(loss.detach().cpu()),
                        "critic/vpreds_slice": vpreds[:2, :16].detach().cpu(),
                        "critic/returns_slice": returns[:2, :16].detach().cpu(),
                        "critic/sp_size": sp_factor,
                        "critic/dp_size": max(
                            1,
                            (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
                            // sp_factor,
                        ),
                        "critic/micro_batches_per_step": len(micro_batches),
                        "critic/sp_loss_divisor": 1.0,
                    }
                    append_to_dict(metrics, micro_batch_metrics)

                    if dump_this_step:
                        post_fwd_payload = {
                            "vpreds": vpreds[:2, :16].detach().cpu(),
                        "vf_loss": vf_loss.detach().cpu(),
                        "loss_for_log": (vf_loss * loss_scale_factor).detach().cpu(),
                        "loss_scale_factor": loss_scale_factor,
                        }
                        if input_dump:
                            post_fwd_payload["inputs"] = input_dump
                        _dump_debug(out_dir, impl, step_id, "post_fwd_pre_bwd", post_fwd_payload)

                python_clip = bool(int(os.getenv("PARITY_DS_PYTHON_CLIP", "0")))
                # Prefer python clip in parity mode or zero_stage=0 to mimic FSDP norm computation.
                if python_clip or self.config.zero_stage == 0:
                    grad_norm_val = float(
                        torch.nn.utils.clip_grad_norm_(
                            self.critic_module.parameters(),
                            max_norm=self.config.grad_clip,
                            norm_type=2.0,
                        )
                    )
                else:
                    ds_zero = getattr(self.config, "zero_stage", 0)
                    if ds_zero and hasattr(self.deepspeed_engine, "get_global_grad_norm"):
                        try:
                            grad_norm_val = float(self.deepspeed_engine.get_global_grad_norm())
                        except Exception:
                            grad_norm_val = 0.0
                    else:
                        grad_norm_val = float(
                            torch.nn.utils.clip_grad_norm_(
                                self.critic_module.parameters(),
                                max_norm=self.config.grad_clip,
                                norm_type=2.0,
                            )
                    )
                append_to_dict(metrics, {"critic/grad_norm": grad_norm_val})
                if dump_this_step:
                    post_bwd = {
                        "grad_norm": float(grad_norm_val),
                        "grad_store": probe_grad,
                    }
                    _dump_debug(out_dir, impl, step_id, "post_bwd_pre_step", post_bwd)
                if debug_enabled:
                    mini_debug["post_clip_grad_norm"] = float(grad_norm_val)
                    opt = getattr(self.deepspeed_engine, "optimizer", None)
                  

                # Step only once after all micro batches
                self.deepspeed_engine.step()
                if trace_steps and debug_enabled:
                    mini_debug.setdefault("ops", []).append({"op": "step", "mini_idx": mini_idx})

                if debug_enabled:
                    post_hash = {n: _hash_stats(p) for n, p in probe_subset.items()}
                    post_slice = {n: _flatten_slice(p) for n, p in probe_subset.items()}
                    delta = {}
                    for name, post in post_slice.items():
                        pre_slice = mini_debug.get("pre_param_slice", {}).get(name)
                        if pre_slice is not None and pre_slice.numel() == post.numel():
                            delta[name] = float(torch.max((post - pre_slice).abs()).cpu())
                    mini_debug["post_step_param_hash"] = post_hash
                    mini_debug["post_step_param_delta_max_abs"] = delta
                    fwd_probe = None
                    if last_micro_inputs is not None:
                        try:
                            with torch.no_grad():
                                fwd_out = self._forward_micro_batch(last_micro_inputs)
                            fwd_probe = float(torch.max(fwd_out.detach().abs()).cpu())
                        except Exception as exc:
                            fwd_probe = f"error:{exc}"
                    mini_debug["post_step_forward_max_abs"] = fwd_probe
                    # Capture optimizer state just before/after step for offline replay.
                    opt = getattr(self.deepspeed_engine, "optimizer", None)
                
                    if fp16_bucket_debug:
                        mini_debug["fp16_bucket_post_step"] = _collect_fp16_bucket_stats(
                            getattr(self.deepspeed_engine, "optimizer", None), "post_step"
                        )
                    mini_debug_records.append(mini_debug)

               
                self.deepspeed_engine.zero_grad()
                if trace_steps and debug_enabled:
                    mini_debug.setdefault("ops", []).append({"op": "zero_grad", "where": "post_step", "mini_idx": mini_idx})

        self.deepspeed_engine.zero_grad()
        if trace_steps and debug_enabled and mini_debug_records:
            # Append a final marker to the last mini for clarity
            mini_debug_records[-1].setdefault("ops", []).append({"op": "zero_grad", "where": "final_reset"})
        if dump_this_step:
            post_step = {}
            probes = self._probe_params()
            for name, p in probes.items():
                slice_now = _flatten_slice(p)
                prev = probe_pre.get(name)
                delta = slice_now - prev if prev is not None else slice_now
                post_step[name] = {"param": slice_now, "delta": delta}
            payload = {
                "inputs": input_dump,
                "probe_pre": probe_pre,
                "probe_grad": probe_grad,
                "post_step": post_step,
                "mini_debug": mini_debug_records,
                "step_id": step_id,
                "impl": impl,
            }
            _dump_debug(out_dir, impl, step_id, "post_step", payload)
        if probe_hooks:
            for h in probe_hooks:
                h.remove()
        return metrics



class CriticWorker(Worker, DistProfilerExtension):
    """Clean DeepSpeed-based Critic Worker."""

    def __init__(self, config: DictConfig | DeepSpeedCriticConfig, **kwargs):
        Worker.__init__(self)

        if isinstance(config, DictConfig):
            critic_config = omega_conf_to_dataclass(config, dataclass_type=DeepSpeedCriticConfig)
        else:
            critic_config = config

        self.config: DeepSpeedCriticConfig = critic_config
        self.critic_sharding_manager = None
        self.layout: ParallelLayout | None = None

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                device_id=get_device_id() if torch.cuda.is_available() else None,
            )

        # Setup profiler
        omega_profiler_config = self.config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=None)
        )

        # Build layout & register dispatch on DP dimension
        self.layout = build_parallel_layout(self.config)
        self._register_dispatch_collect_info("critic", dp_rank=self.layout.dp_rank, is_collect=self.layout.collect)

        self._is_offload_param = self.config.deepspeed_config.get("param_offload", False)
        # Ulysses SP for critic dynamic batching
        self.ulysses_sequence_parallel_size = self.layout.sp_size
        self._lora_rank = getattr(self.config.model, "lora_rank", 0)
        self._is_lora = self._lora_rank > 0

        normalize_critic_batches(self.config, self.layout.dp_size, sp_size=self.layout.sp_size)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize critic model with native DeepSpeed."""

        import torch  # local binding to avoid scope issues when checking distributed state

        import_external_libs(self.config.model.get("external_lib", None))

        trust_remote_code = getattr(self.config.model, "trust_remote_code", False)
        use_shm = getattr(self.config.model, "use_shm", False)
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)

        tokenizer_path = getattr(self.config.model, "tokenizer_path", None) or self.config.model.path
        tokenizer_local_path = copy_to_local(tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(tokenizer_local_path, trust_remote_code=trust_remote_code)

        custom_template = getattr(self.config.model, "custom_chat_template", None)
        if custom_template is not None:
            if self.processor is not None:
                self.processor.chat_template = custom_template
            elif self.tokenizer is not None:
                self.tokenizer.chat_template = custom_template

        override_config = getattr(self.config.model, "override_config", {}) or {}
        if isinstance(override_config, DictConfig):
            override_config = OmegaConf.to_container(override_config, resolve=True)
        override_config_kwargs = {}
        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )
        override_config_kwargs.update(override_config)

        attn_impl = override_config_kwargs.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
        )
        update_model_config(critic_model_config, override_config_kwargs=override_config_kwargs)
        critic_model_config.num_labels = 1
        critic_model_config.classifier_dropout = 0.0
        critic_model_config.hidden_dropout = 0.0
        critic_model_config.summary_dropout_prob = 0.0

        torch_dtype = PrecisionType.to_dtype(getattr(self.config.deepspeed_config, "model_dtype", "fp32"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_module = load_valuehead_model(
                local_path=local_path,
                torch_dtype=torch_dtype,
                model_config=critic_model_config,
                trust_remote_code=trust_remote_code,
            )

        use_remove_padding = getattr(self.config.model, "use_remove_padding", False)

        # Initialize Ulysses SP group for Critic if requested
        sp_size = int(getattr(self, "ulysses_sequence_parallel_size", 1))
        sp_group = None
        prev_sp_group = get_ulysses_sequence_parallel_group()
        if sp_size > 1 and torch.distributed.is_initialized():
            if self.layout is not None:
                ranks = list(range(self.layout.dp_rank * sp_size, (self.layout.dp_rank + 1) * sp_size))
            else:
                world = torch.distributed.get_world_size()
                assert world % sp_size == 0, f"world_size {world} must be divisible by ulysses sp_size {sp_size}"
                rank = torch.distributed.get_rank()
                group_id = rank // sp_size
                ranks = list(range(group_id * sp_size, (group_id + 1) * sp_size))
            sp_group = torch.distributed.new_group(ranks=ranks, backend=get_nccl_backend())
            set_ulysses_sequence_parallel_group(sp_group)
            torch.distributed.barrier(group=sp_group)

        apply_monkey_patch(
            model=critic_module,
            use_remove_padding=use_remove_padding,
            ulysses_sp_size=sp_size,
        )

        if sp_group is not None:
            torch.distributed.barrier(group=sp_group)
        set_ulysses_sequence_parallel_group(prev_sp_group)

        critic_module.to(torch_dtype)

        if getattr(self.config.model, "enable_gradient_checkpointing", False):
            critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self._lora_rank,
                "lora_alpha": getattr(self.config.model, "lora_alpha", 16),
                "target_modules": convert_to_regular_types(getattr(self.config.model, "target_modules", None)),
                "bias": "none",
            }
            exclude_modules = getattr(self.config.model, "exclude_modules", None)
            if exclude_modules is not None:
                lora_config["exclude_modules"] = convert_to_regular_types(exclude_modules)
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config
        self.critic_sharding_manager = DeepSpeedUlyssesShardingManager(sp_group)

        # Initialize DeepSpeed
        # Parse mixed precision config (supports str or dict)
        fp16_enabled, bf16_enabled = _parse_mixed_precision_config(
            self.config.deepspeed_config.get("mixed_precision")
        )

        zero_stage = getattr(self.config, "zero_stage", self.config.deepspeed_config.get("zero_stage", 2))

        dp_size = self.layout.dp_size if self.layout is not None else (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        )
        world_size = self.layout.world_size if self.layout is not None else (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        )
        sp_size = self.layout.sp_size if self.layout is not None else 1
        if world_size != max(1, dp_size) * max(1, sp_size):
            raise AssertionError(f"world_size({world_size}) must equal dp_size({dp_size}) * sp_size({sp_size})")
        per_rank_mini = self.config.ppo_mini_batch_size
        micro_bsz = self.config.get("ppo_micro_batch_size_per_gpu", 1) or 1
        ds_grad_accum, ds_train_batch_size = _derive_ds_batch_params(per_rank_mini, micro_bsz, dp_size)
        # Cache computed sizes explicitly to avoid stale/frozen config copies.
        setattr(self.config.deepspeed_config, "ds_train_batch_size", ds_train_batch_size)
        setattr(self.config.deepspeed_config, "ds_grad_accum", ds_grad_accum)
        setattr(self.config.deepspeed_config, "ds_micro_batch_size_per_gpu", micro_bsz)

        if self.rank == 0:
            print(
                f"[ds-critic-config] sp_size={sp_size}, dp_size={dp_size}, world_size={world_size}, "
                f"per_rank_mini={per_rank_mini}, micro_bsz={micro_bsz}, grad_accum={ds_grad_accum}, "
                f"train_batch_size={ds_train_batch_size}"
            )
            print(
                f"[ds-critic-config-detail] ds_cfg.train_batch_size={ds_train_batch_size}, "
                f"ds_cfg.grad_accum={ds_grad_accum}, "
                f"ppo_micro_batch_size_per_gpu={self.config.ppo_micro_batch_size_per_gpu}"
            )

        python_clip = bool(int(os.getenv("PARITY_DS_PYTHON_CLIP", "0")))
        ds_config = get_deepspeed_config(
            optimizer_type=self.config.optim.get("optimizer", "AdamW"),
            train_batch_size=ds_train_batch_size,
            train_micro_batch_size_per_gpu=micro_bsz,
            gradient_accumulation_steps=ds_grad_accum,
            zero_stage=zero_stage,
            lr=self.config.optim.lr,
            betas=self.config.optim.get("betas", [0.9, 0.999]),
            weight_decay=self.config.optim.get("weight_decay", 0.01),
            fp16_enabled=fp16_enabled,
            bf16_enabled=bf16_enabled,
            cpu_offload=self.config.deepspeed_config.get("param_offload", False),
            offload_optimizer=self.config.deepspeed_config.get("optimizer_offload", False),
            gradient_clipping=0.0 if python_clip else self.config.get("grad_clip", None),
        )
        # Force pure torch AdamW to bypass fused/FusedAdam path.
        ds_config["optimizer"]["params"]["torch_adam"] = True
        ds_config["optimizer"]["params"]["fused"] = False
        import torch.optim
        torch_optimizer = torch.optim.AdamW(
            critic_module.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.get("betas", [0.9, 0.999]),
            weight_decay=self.config.optim.get("weight_decay", 0.01),
        )

        # Static sanity checks before touching DeepSpeed to ensure we pass the intended dict.
        expected_train_batch = micro_bsz * ds_grad_accum * dp_size
        if ds_config.get("gradient_accumulation_steps") != ds_grad_accum:
            raise AssertionError(
                f"ds_config.gradient_accumulation_steps={ds_config.get('gradient_accumulation_steps')} "
                f"!= expected {ds_grad_accum}"
            )
        if ds_config.get("train_micro_batch_size_per_gpu") != micro_bsz:
            raise AssertionError(
                f"ds_config.train_micro_batch_size_per_gpu={ds_config.get('train_micro_batch_size_per_gpu')} "
                f"!= expected {micro_bsz}"
            )
        if ds_config.get("train_batch_size") != expected_train_batch:
            raise AssertionError(
                f"ds_config.train_batch_size={ds_config.get('train_batch_size')} "
                f"!= micro_bsz({micro_bsz}) * grad_accum({ds_grad_accum}) * dp_size({dp_size})"
            )
        if self.rank == 0:
            print(f"[ds-critic-config-final] {convert_to_regular_types(ds_config)}")

        # DeepSpeed 默认按 world_size 校验 train_batch_size；为保持与 FSDP 同步的 DP 语义（micro * GAS * dp_size），禁用该断言。
        try:
            import deepspeed.runtime.config as ds_cfg_mod

            if hasattr(ds_cfg_mod.DeepSpeedConfig, "_batch_assertion"):
                ds_cfg_mod.DeepSpeedConfig._batch_assertion = lambda self: None
        except Exception:
            pass

        self.critic_engine, optimizer, _, lr_scheduler = initialize_deepspeed_engine(
            model=critic_module,
            config=ds_config,
            model_parameters=critic_module.parameters(),
            optimizer=torch_optimizer,
        )

        self.critic_module = self.critic_engine.module
        self.critic_optimizer = optimizer
        self.critic_lr_scheduler = lr_scheduler

        self.critic = DeepSpeedPPOCritic(
            config=self.config,
            critic_module=self.critic_module,
            engine=self.critic_engine,
            ds_config=ds_config,
        )

        if self.rank == 0:
            opt = self.critic_engine.optimizer
            base_opt = getattr(opt, "optimizer", None)
            fused_flag = getattr(getattr(opt, "defaults", None), "get", lambda k, d=None: d)("fused", None)
            has_master = hasattr(opt, "fp32_param_groups")
            try:
                eng_ga = (
                    self.critic_engine.gradient_accumulation_steps()
                    if callable(getattr(self.critic_engine, "gradient_accumulation_steps", None))
                    else getattr(self.critic_engine, "gradient_accumulation_steps", None)
                )
            except Exception:
                eng_ga = "NA"
            print(
                f"[ds-critic-optimizer] class={type(opt).__name__}, "
                f"base_class={type(base_opt).__name__ if base_opt else 'None'}, "
                f"fused_flag={fused_flag}, has_fp32_master={has_master}, engine_grad_accum={eng_ga}"
            )

        # Setup checkpoint manager for critic (mirror actor behavior)
        # Expose engine handle for DeepSpeedCheckpointManager via self.engine
        self.engine = self.critic_engine
        try:
            self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)
        except Exception:
            # Defer creation if environment lacks DS helpers; save/load will fallback to engine API
            self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="critic_compute_values")
    def compute_values(self, data: DataProto):
        """Run critic forward pass to compute value estimates."""
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        micro_batch_size = getattr(self.config, "forward_micro_batch_size_per_gpu", None)
        if micro_batch_size is None:
            micro_batch_size = getattr(self.config, "ppo_micro_batch_size_per_gpu", 1)

        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = getattr(
            self.config, "forward_max_token_len_per_gpu", self.config.ppo_max_token_len_per_gpu
        )
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz

        data = data.to("cpu")  # tensors move to device inside critic.compute_values

        manager = self.critic_sharding_manager
        ctx = manager if manager is not None else nullcontext()
        with ctx:
            values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values}).to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="blue", role="critic_update")
    def update_critic(self, data: DataProto):
        """Update critic value function."""
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        data = data.to("cpu")

        manager = self.critic_sharding_manager
        ctx = manager if manager is not None else nullcontext()
        with ctx:
            with Timer(name="update_critic", logger=None):
                metrics = self.critic.update_critic(data=data)

        lr = self.critic_lr_scheduler.get_last_lr()[0]
        metrics["critic/lr"] = lr
        self.critic_lr_scheduler.step()

        output = DataProto(meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step: int = 0, max_ckpt_to_keep: int | None = None):
        """Save critic (DeepSpeed) checkpoint using native DeepSpeed format."""
        import torch

        # Ensure engine handle is set for checkpoint manager
        self.engine = getattr(self, "critic_engine", None)
        if self.engine is None:
            return

        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        self.checkpoint_manager.save(
            root=local_path, global_step=global_step, hdfs_path=hdfs_path, max_ckpt_to_keep=max_ckpt_to_keep
        )

        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, step: int | None = None, del_local_after_load: bool = True):
        """Load critic (DeepSpeed) checkpoint saved by `save_checkpoint`."""
        import torch

        # Treat None or empty path as no-op for compatibility
        if local_path is None or (isinstance(local_path, str) and local_path.strip() == ""):
            return {}

        # Ensure engine handle is set for checkpoint manager
        self.engine = getattr(self, "critic_engine", None)
        if self.engine is None:
            return {}

        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        state = self.checkpoint_manager.load(
            root=local_path, step=step, hdfs_path=None, del_local_after_load=del_local_after_load
        )

        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

        return state


class RolloutWorker(Worker):
    """Standalone vLLM/SGLang Rollout Worker."""

    def __init__(self, config: DictConfig, **kwargs):
        Worker.__init__(self)
        self.config = config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize rollout engine only."""
        rollout_config = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)
        model_config = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=None
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto):
        """Generate sequences."""
        prompts = prompts.to(get_device_id())
        output = self.rollout.generate_sequences(prompts=prompts)
        return output


# Async variants
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    """Async variant of ActorRolloutRefWorker."""
    pass


class RewardModelWorker(Worker, DistProfilerExtension):
    """
    DeepSpeed-based Reward Model Worker.

    Implements reward model inference using DeepSpeed ZeRO for memory efficiency.
    Supports AutoModelForTokenClassification models.
    """

    def __init__(self, config):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self.config = config
        self.layout: ParallelLayout | None = None
        self.reward_sharding_manager: DeepSpeedUlyssesShardingManager | None = None
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                device_id=get_device_id() if torch.cuda.is_available() else None,
            )

        # Build layout (supports DP; optional SP via ulysses_sequence_parallel_size)
        self.layout = build_parallel_layout(self.config)
        self.ulysses_sequence_parallel_size = self.layout.sp_size

        # Create training dispatch
        self._register_dispatch_collect_info("reward", dp_rank=self.layout.dp_rank, is_collect=self.layout.collect)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # Normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= self.layout.dp_size
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

        self._is_offload_param = self.config.deepspeed_config.get("param_offload", False)

    def _build_model(self, config):
        """Build reward model with DeepSpeed."""
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        # Handle chat template switching if needed
        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # Determine torch dtype
        torch_dtype = self.config.deepspeed_config.get("model_dtype", "fp32")
        if torch_dtype == "fp32":
            torch_dtype = torch.float32
        elif torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16  # default to bf16

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # Initialize Ulysses SP group if requested
            sp_size = self.ulysses_sequence_parallel_size
            sp_group = None
            prev_sp_group = get_ulysses_sequence_parallel_group()
            if sp_size > 1 and torch.distributed.is_initialized():
                ranks = list(range(self.layout.dp_rank * sp_size, (self.layout.dp_rank + 1) * sp_size))
                sp_group = torch.distributed.new_group(ranks=ranks, backend=get_nccl_backend())
                set_ulysses_sequence_parallel_group(sp_group)
                torch.distributed.barrier(group=sp_group)

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=sp_size,
            )

            reward_module.to(torch_dtype)

        if sp_group is not None:
            torch.distributed.barrier(group=sp_group)
        set_ulysses_sequence_parallel_group(prev_sp_group)
        self.reward_sharding_manager = DeepSpeedUlyssesShardingManager(sp_group)

        # Initialize DeepSpeed for inference (no optimizer)
        # Parse mixed precision config
        fp16_enabled, bf16_enabled = _parse_mixed_precision_config(
            self.config.deepspeed_config.get("mixed_precision")
        )

        zero_stage = getattr(self.config, "zero_stage", self.config.deepspeed_config.get("zero_stage", 2))

        ds_config = get_deepspeed_config(
            optimizer_type="AdamW",  # Not used but required by config generator
            train_batch_size=1,
            train_micro_batch_size_per_gpu=1,
            gradient_accumulation_steps=1,
            zero_stage=zero_stage,
            lr=1e-5,  # Not used but required
            fp16_enabled=fp16_enabled,
            bf16_enabled=bf16_enabled,
            cpu_offload=self.config.deepspeed_config.get("param_offload", False),
            offload_optimizer=False,  # No optimizer for inference
            disable_scheduler=True,  # No scheduler for inference
        )

        # Remove optimizer from config since this is inference only
        if "optimizer" in ds_config:
            del ds_config["optimizer"]

        # Initialize DeepSpeed engine without optimizer
        ds_engine, _, _, _ = initialize_deepspeed_engine(
            model=reward_module,
            config=ds_config,
            model_parameters=None,  # No parameters needed for inference
        )

        return ds_engine

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize reward model."""
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_engine = self._build_model(config=self.config)
        self.reward_module = self.reward_engine.module

    def _forward_micro_batch(self, micro_batch):
        """Forward pass for a single micro batch."""
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            sp_size = self.ulysses_sequence_parallel_size

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                pad_size = 0
                if sp_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=sp_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if sp_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        """Expand sentence-level scores to token-level."""
        batch_size = data.batch.batch_size[0]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        """Switch chat template if input_tokenizer is different from reward model tokenizer."""
        import numpy as np

        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                print(f"Switch template. chat: {prompt_with_chat_template}")

            prompt_ids = target_tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)

            # pad or truncate
            if len(prompt_ids) < src_max_length:
                prompt_ids = prompt_ids + [target_tokenizer.pad_token_id] * (src_max_length - len(prompt_ids))
                attn_mask = [1] * len(prompt_ids) + [0] * (src_max_length - len(prompt_ids))
            else:
                prompt_ids = prompt_ids[:src_max_length]
                attn_mask = [1] * src_max_length

            rm_input_ids.append(prompt_ids)
            rm_attention_mask.append(attn_mask)

        # convert to tensors
        rm_input_ids = torch.tensor(rm_input_ids, dtype=torch.long, device=data.batch["input_ids"].device)
        rm_attention_mask = torch.tensor(
            rm_attention_mask, dtype=torch.long, device=data.batch["attention_mask"].device
        )

        # update data
        data.batch["input_ids"] = rm_input_ids
        data.batch["attention_mask"] = rm_attention_mask
        # recompute position_ids
        data.batch["position_ids"] = compute_position_id_with_mask(rm_attention_mask)

        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    def compute_rm_score(self, data: DataProto):
        """Compute reward model scores."""
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.reward_engine)

        data = data.to("cpu")

        # Switch chat template if needed
        if self._do_switch_chat_template:
            data = self._switch_chat_template(data)

        # Move data to device
        data = data.to(get_device_id())

        # Compute scores for each micro batch
        micro_batch_size = self.config.get("micro_batch_size_per_gpu", 1)
        batch_size = data.batch.batch_size[0]
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        all_scores = []
        manager = self.reward_sharding_manager if self.reward_sharding_manager is not None else nullcontext()
        with manager:
            for i in range(num_micro_batches):
                start_idx = i * micro_batch_size
                end_idx = min((i + 1) * micro_batch_size, batch_size)

                micro_batch = {
                    "input_ids": data.batch["input_ids"][start_idx:end_idx],
                    "attention_mask": data.batch["attention_mask"][start_idx:end_idx],
                    "position_ids": data.batch["position_ids"][start_idx:end_idx],
                }

                scores = self._forward_micro_batch(micro_batch)
                all_scores.append(scores)

        # Concatenate all scores
        rm_scores = torch.cat(all_scores, dim=0)

        # Expand to token level if needed
        token_level_scores = self._expand_to_token_level(data, rm_scores)

        output = DataProto.from_dict(tensors={"token_level_scores": token_level_scores})

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.reward_engine)

        return output
