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
Implement a multiprocess PPOCritic
"""

import logging
import os
import hashlib
from contextlib import nullcontext
from typing import Dict, List, Tuple

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

PROBE_NAMES: Tuple[str, ...] = (
    "model.embed_tokens.weight",
    "lm_head.weight",
    "v_head.weight",
    "value_head.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
)


def _hash_tensor(t: torch.Tensor) -> str:
    try:
        return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
    except Exception:
        return "NA"


def _flatten_slice(t: torch.Tensor, limit: int = 4096) -> torch.Tensor:
    return t.detach().float().reshape(-1)[:limit].cpu()


def _should_dump(meta: Dict) -> bool:
    if not meta:
        return False
    return str(meta.get("parity_dump", "0")) == "1"


def _step_enabled(meta: Dict, step_id: int) -> bool:
    if not _should_dump(meta):
        return False
    raw = str(meta.get("parity_dump_steps", "0"))
    try:
        steps = {int(x.strip()) for x in raw.split(",") if x.strip() != ""}
    except Exception:
        steps = {0}
    return step_id in steps


def _dump_debug(out_dir: str, impl: str, step_id: int, tag: str, payload: Dict):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"debug_rank0_step{step_id}_{tag}.pt")
    torch.save(payload, path)


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

    def _probe_params(self) -> Dict[str, nn.Parameter]:
        probes = {}
        for name, p in self.critic_module.named_parameters():
            norm_name = name.replace("_fsdp_wrapped_module.", "")
            if norm_name in PROBE_NAMES and p.requires_grad:
                probes[norm_name] = p
        return probes

    def _attach_grad_hooks(self, probes: Dict[str, nn.Parameter], store: Dict[str, torch.Tensor]) -> List:
        hooks: List = []

        def _make(name):
            def fn(grad):
                store[name] = _flatten_slice(grad)

            return fn

        for n, p in probes.items():
            if p.grad_fn is None and not p.requires_grad:
                continue
            hooks.append(p.register_hook(_make(n)))
        return hooks

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

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
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outputs_and_unpad(
                        values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                values = values[:, -response_length - 1 : -1]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values = output[2]
                else:
                    values = output.logits
                values = values[:, -response_length - 1 : -1].squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = (
            ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
            if "response_mask" in data.batch
            else ["responses", "input_ids", "attention_mask", "position_ids"]
        )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                values = self._forward_micro_batch(model_inputs)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)

        if "response_mask" in data.batch:
            response_mask = data.batch["response_mask"]
            response_mask = response_mask.to(values.device)
            values = values * response_mask  # Only action tokens have values
        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()

        # Align RNG with DS parity runs if provided in meta.
        if isinstance(data.meta_info, dict) and "rng_seed" in data.meta_info:
            rng_seed = data.meta_info["rng_seed"]
            if torch.distributed.get_rank() == 0:
                print(f"[FSDP Critic] Setting RNG seed: {rng_seed}")
            torch.manual_seed(rng_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rng_seed)
        metrics = {}
        loss_debug = bool(int(os.getenv("PARITY_FSDP_LOSS_DEBUG", "0")))
        meta = data.meta_info
        impl = meta.get("impl", "fsdp") if isinstance(meta, dict) else "fsdp"
        step_id = int(meta.get("step_id", 0)) if isinstance(meta, dict) else 0
        out_dir = meta.get("parity_out_dir") if isinstance(meta, dict) else None
        dump_this_step = _step_enabled(meta, step_id) if isinstance(meta, dict) else False
        probe_pre = {}
        probe_grad: Dict[str, torch.Tensor] = {}
        probe_hooks: List = []
        input_dump = {}
        probes = {}
        if dump_this_step:
            # capture small slices of inputs for first micro-batch later
            probes = self._probe_params()
            for name, p in probes.items():
                probe_pre[name] = _flatten_slice(p)
            probe_hooks = self._attach_grad_hooks(probes, probe_grad)
            _dump_debug(out_dir, impl, step_id, "pre_fwd", {"probe_pre": probe_pre})

        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    any_tensor = next(iter(mini_batch.batch.values()))
                    actual_mini = any_tensor.shape[0]
                    print(
                        f"[fsdp-critic-batch] mini={actual_mini}, config_mini={self.config.ppo_mini_batch_size}, "
                        f"micro_bsz={self.config.ppo_micro_batch_size_per_gpu}, "
                        f"micro_batches={len(micro_batches)}, grad_accum={self.gradient_accumulation}"
                    )
                micro_batch_count = len(micro_batches)

                self.critic_optimizer.zero_grad()

                for micro_idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    if dump_this_step and not input_dump:
                        # record a small snapshot of inputs
                        input_dump = {
                            "input_ids": model_inputs["input_ids"][:2, :16].detach().cpu(),
                            "responses": model_inputs["responses"][:2, :16].detach().cpu(),
                            "attention_mask": model_inputs["attention_mask"][:2, :16].detach().cpu(),
                            "position_ids": model_inputs["position_ids"][:2, :16].detach().cpu(),
                            "values": values[:2, :16].detach().cpu(),
                            "returns": returns[:2, :16].detach().cpu(),
                            "response_mask": response_mask[:2, :16].detach().cpu(),
                            "hash_input_ids": _hash_tensor(model_inputs["input_ids"]),
                            "hash_attention_mask": _hash_tensor(model_inputs["attention_mask"]),
                            "hash_position_ids": _hash_tensor(model_inputs["position_ids"]),
                        }

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
                        # relative to the dynamic bsz
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        loss = vf_loss * loss_scale_factor
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = vf_loss * loss_scale_factor

                    if loss_debug and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                        print(
                            f"[fsdp-critic-loss-debug] batch_idx={batch_idx}, micro_idx={micro_idx}, "
                            f"grad_accum={self.gradient_accumulation}, "
                            f"token_sum={float(response_mask.sum().detach().cpu())}, "
                            f"vf_loss={float(vf_loss.detach().cpu())}, "
                            f"loss_scale_factor={loss_scale_factor}, "
                            f"loss_for_backward={float(loss.detach().cpu())}"
                        )

                    loss.backward()

                    if dump_this_step and not micro_batch_metrics.get("debug/post_fwd"):
                        micro_batch_metrics["debug/post_fwd"] = {
                            "vpreds": vpreds[:2, :16].detach().cpu(),
                            "vf_loss": vf_loss.detach().cpu(),
                            "loss": loss.detach().cpu(),
                            "loss_scale_factor": loss_scale_factor,
                            "returns": returns[:2, :16].detach().cpu(),
                        }

                    micro_batch_metrics.update(
                        {
                            "critic/vf_loss": vf_loss.detach().item() * loss_scale_factor,
                            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                            "critic/token_sum_local": float(response_mask.sum().detach().cpu()),
                            "critic/per_token_mse_raw": float(
                                torch.nan_to_num(((vpreds - returns) ** 2 * response_mask).sum(), nan=0.0)
                                / max(1.0, float(response_mask.sum().detach().cpu()))
                            ),
                            "critic/grad_accum_steps": float(self.gradient_accumulation),
                            "critic/loss_scale_factor": float(loss_scale_factor),
                            "critic/loss_unscaled": float(vf_loss.detach().cpu()),
                            "critic/loss_for_backward": float(loss.detach().cpu()),
                            "critic/vpreds_slice": vpreds[:2, :16].detach().cpu(),
                            "critic/returns_slice": returns[:2, :16].detach().cpu(),
                        }
                    )

                    append_to_dict(metrics, micro_batch_metrics)

            grad_norm = self._optimizer_step()
            mini_batch_metrics = {
                "critic/grad_norm": grad_norm.detach().item(),
                "critic/micro_batches_per_step": float(micro_batch_count),
            }
            append_to_dict(metrics, mini_batch_metrics)
            if dump_this_step:
                # ensure gradients captured even if hooks missed
                if not probe_grad and probes:
                    for n, p in probes.items():
                        if p.grad is not None:
                            probe_grad[n] = _flatten_slice(p.grad)
                post_fwd_payload = micro_batch_metrics.get("debug/post_fwd", {})
                if input_dump:
                    post_fwd_payload.update({"inputs": input_dump})
                _dump_debug(out_dir, impl, step_id, "post_fwd_pre_bwd", post_fwd_payload)
                post_bwd = {
                    "grad_norm": grad_norm.detach().cpu(),
                    "grad_store": probe_grad,
                }
                _dump_debug(out_dir, impl, step_id, "post_bwd_pre_step", post_bwd)
            # optimizer step updates params
        self.critic_optimizer.zero_grad()
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
                "step_id": step_id,
                "impl": impl,
            }
            _dump_debug(out_dir, impl, step_id, "post_step", payload)
            for h in probe_hooks:
                h.remove()
        return metrics
