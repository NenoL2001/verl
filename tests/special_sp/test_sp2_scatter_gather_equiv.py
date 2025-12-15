import argparse
import os
import time

import torch
import torch.distributed as dist
from flash_attn.bert_padding import pad_input, unpad_input

from verl.utils.ulysses import gather_outputs_and_unpad, set_ulysses_sequence_parallel_group, slice_input_tensor


def _init_dist():
    if dist.is_initialized():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        return device
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_id = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
    dist.init_process_group(backend=backend, init_method="env://", device_id=device_id)
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    return device


def _build_hidden_states(batch_size: int, seqlen: int, hidden: int, device: torch.device):
    rank = dist.get_rank()
    base_lengths = [16, 12, 8, 14, 10, 6, 4, 2]
    lengths = [max(1, min(seqlen, base_lengths[i % len(base_lengths)])) for i in range(batch_size)]

    if rank == 0:
        torch.manual_seed(1234)
        hidden_states = torch.randn(batch_size, seqlen, hidden, device=device)
        attention_mask = torch.zeros(batch_size, seqlen, device=device, dtype=torch.long)
        for i, ln in enumerate(lengths):
            attention_mask[i, :ln] = 1
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
    else:
        hidden_states = torch.empty(batch_size, seqlen, hidden, device=device)
        attention_mask = torch.empty(batch_size, seqlen, device=device, dtype=torch.long)
        lengths_tensor = torch.empty(batch_size, device=device, dtype=torch.long)

    dist.broadcast(hidden_states, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(lengths_tensor, src=0)
    lengths = lengths_tensor.tolist()
    return hidden_states, attention_mask, lengths


def main():
    parser = argparse.ArgumentParser(description="Validate Ulysses SP scatter/gather equivalence.")
    parser.add_argument("--run_id", default=time.strftime("%Y%m%d_%H%M%S"), help="Run identifier for dump directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seqlen", type=int, default=32, help="Sequence length (including padding)")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension size")
    args = parser.parse_args()

    device = _init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    sp_size = world
    if world < 2:
        raise RuntimeError("This test is intended for sp_size>=2 (use torchrun with at least 2 ranks).")

    hidden_states, attention_mask, lengths = _build_hidden_states(args.batch_size, args.seqlen, args.hidden, device)

    hidden_rmpad, indices, *_ = unpad_input(hidden_states, attention_mask)
    hidden_rmpad_batched = hidden_rmpad.unsqueeze(0)
    pad_size = (sp_size - (hidden_rmpad_batched.size(1) % sp_size)) % sp_size
    if pad_size > 0:
        pad_tensor = torch.zeros((1, pad_size, hidden_rmpad_batched.size(-1)), device=device, dtype=hidden_rmpad_batched.dtype)
        hidden_padded = torch.cat([hidden_rmpad_batched, pad_tensor], dim=1)
    else:
        hidden_padded = hidden_rmpad_batched

    shard = slice_input_tensor(hidden_padded, dim=1, padding=False)
    gathered = gather_outputs_and_unpad(shard, gather_dim=1, unpad_dim=1, padding_size=pad_size)

    max_abs_gather = float((gathered - hidden_rmpad_batched).abs().max().item())
    reconstructed = pad_input(gathered.squeeze(0), indices, batch=args.batch_size, seqlen=args.seqlen)
    max_abs_reconstruct = float((reconstructed - hidden_states).abs().max().item())

    dump_root = "/tmp/verl_sp_parity"
    dump_dir = os.path.join(dump_root, f"sp_unit_{args.run_id}")
    os.makedirs(dump_dir, exist_ok=True)
    torch.save(
        {
            "rank": rank,
            "world": world,
            "sp_size": sp_size,
            "pad_size": pad_size,
            "lengths": lengths,
            "attention_mask": attention_mask.detach().cpu(),
            "hidden_full": hidden_states.detach().cpu(),
            "hidden_rmpad": hidden_rmpad.detach().cpu(),
            "hidden_padded": hidden_padded.detach().cpu(),
            "local_shard": shard.detach().cpu(),
            "gathered": gathered.detach().cpu(),
            "reconstructed": reconstructed.detach().cpu(),
            "max_abs_gather": max_abs_gather,
            "max_abs_reconstruct": max_abs_reconstruct,
        },
        os.path.join(dump_dir, f"rank{rank}.pt"),
    )

    gather_tensor = torch.tensor([max_abs_gather, max_abs_reconstruct], device=device)
    dist.all_reduce(gather_tensor, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"[scatter_gather_equiv] world={world} sp_size={sp_size} "
            f"max_abs_gather={gather_tensor[0].item()} max_abs_reconstruct={gather_tensor[1].item()} "
            f"dumps={dump_dir}"
        )

    dist.barrier()


if __name__ == "__main__":
    main()
