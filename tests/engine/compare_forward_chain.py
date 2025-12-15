import argparse
import os
from typing import Dict, Tuple

import torch

CHAIN_ORDER = [
    "embed_out",
    "pre_pack_hidden",
    "post_pack_hidden",
    "post_ulysses_scatter",
    "layer0_attn_in",
    "vproj_out",
    "layer0_attn_out",
    "layer0_mlp_in",
    "layer0_mlp_out",
]


def _collect_rank_files(directory: str, suffix: str) -> Dict[int, str]:
    files = {}
    for fname in os.listdir(directory):
        if not fname.startswith("rank") or not fname.endswith(suffix):
            continue
        try:
            rank = int(fname.split("rank", 1)[1].split("_", 1)[0])
        except Exception:
            continue
        files[rank] = os.path.join(directory, fname)
    return files


def _compare_entry(fsdp_ent: dict, ds_ent: dict) -> Tuple[bool, float, float, torch.Tensor]:
    if fsdp_ent is None or ds_ent is None or fsdp_ent.get("slice") is None or ds_ent.get("slice") is None:
        return False, float("inf"), float("inf"), torch.tensor([])
    f = fsdp_ent["slice"].float()
    d = ds_ent["slice"].float()
    if f.shape != d.shape:
        return False, float("inf"), float("inf"), torch.tensor([])
    diff = (f - d).abs()
    return diff.max().item() == 0, diff.max().item(), diff.mean().item(), diff


def main():
    parser = argparse.ArgumentParser(description="Compare forward chain dumps between FSDP and DeepSpeed.")
    parser.add_argument("--fsdp_dir", required=True, help="Directory containing fsdp rank*_fwd_chain.pt")
    parser.add_argument("--ds_dir", required=True, help="Directory containing ds rank*_fwd_chain.pt")
    parser.add_argument("--topk", type=int, default=10, help="TopK diff indices to print")
    args = parser.parse_args()

    fsdp_files = _collect_rank_files(args.fsdp_dir, "_fwd_chain.pt")
    ds_files = _collect_rank_files(args.ds_dir, "_fwd_chain.pt")
    shared = sorted(set(fsdp_files) & set(ds_files))
    if not shared:
        raise SystemExit("No overlapping rank forward-chain dumps.")

    for r in shared:
        f_obj = torch.load(fsdp_files[r], map_location="cpu")
        d_obj = torch.load(ds_files[r], map_location="cpu")
        token_map_f = f_obj.get("token_map")
        token_map_d = d_obj.get("token_map")
        pad_f = f_obj.get("pad_size")
        pad_d = d_obj.get("pad_size")
        print(f"=== rank {r} ===")
        if token_map_f is not None and token_map_d is not None:
            tok_ok = torch.equal(token_map_f, token_map_d)
            print(f"token_map: {'OK' if tok_ok else 'MISMATCH'}")
        if pad_f is not None or pad_d is not None:
            print(f"pad_size fsdp={pad_f} ds={pad_d}")
        first_diff = None
        for name in CHAIN_ORDER:
            ent_f = f_obj.get(name)
            ent_d = d_obj.get(name)
            ok, max_abs, mean_abs, diff = _compare_entry(ent_f, ent_d)
            status = "OK" if ok else "MISMATCH"
            print(f"{name}: {status} max_abs={max_abs} mean_abs={mean_abs}")
            if not ok and first_diff is None:
                first_diff = name
                if diff.numel():
                    vals, idx = torch.topk(diff.flatten(), k=min(args.topk, diff.numel()))
                    print(" top diff indices/values:")
                    for i in range(len(vals)):
                        print(f"  idx={idx[i].item()} fsdp={ent_f['slice'][idx[i]].item()} ds={ent_d['slice'][idx[i]].item()} diff={vals[i].item()}")
        if first_diff is None:
            print("All chain nodes match.")
        else:
            print(f"First diff at: {first_diff}")


if __name__ == "__main__":
    main()
