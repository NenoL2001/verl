import argparse
import os
from typing import Dict, List, Tuple

import torch


def _collect_rank_files(directory: str) -> Dict[int, str]:
    files = {}
    for fname in os.listdir(directory):
        if not fname.startswith("rank") or not fname.endswith("_sp_io.pt"):
            continue
        try:
            rank_part = fname.split("rank", 1)[1]
            rank_str = rank_part.split("_", 1)[0]
            rank = int(rank_str)
        except Exception:
            continue
        files[rank] = os.path.join(directory, fname)
    return files


def _top_diffs(fsdp_t: torch.Tensor, ds_t: torch.Tensor, max_print: int) -> List[Tuple[Tuple[int, ...], float, float, float]]:
    diff = (fsdp_t.to(torch.float64) - ds_t.to(torch.float64)).abs()
    if diff.numel() == 0:
        return []
    k = min(max_print, diff.numel())
    if k == 0:
        return []
    vals, idx = torch.topk(diff.flatten(), k)
    coords = torch.unravel_index(idx, diff.shape)
    out = []
    for i in range(k):
        coord = tuple(int(coords[d][i]) for d in range(len(coords)))
        out.append((coord, float(fsdp_t[coord]), float(ds_t[coord]), float(vals[i])))
    return out


def _compare_tensor(name: str, fsdp_t, ds_t, max_print: int):
    if fsdp_t is None or ds_t is None:
        ok = fsdp_t is None and ds_t is None
        info = "both None" if ok else "one is None"
        return ok, info, []
    if fsdp_t.shape != ds_t.shape or fsdp_t.dtype != ds_t.dtype:
        info = f"shape/dtype fsdp={fsdp_t.shape}/{fsdp_t.dtype} ds={ds_t.shape}/{ds_t.dtype}"
        return False, info, []
    diff = (fsdp_t.to(torch.float64) - ds_t.to(torch.float64)).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    if max_abs == 0:
        return True, "OK", []
    top_entries = _top_diffs(fsdp_t, ds_t, max_print)
    info = f"max_abs={max_abs}"
    return False, info, top_entries


def _compare_meta(fsdp_meta, ds_meta):
    if fsdp_meta is None or ds_meta is None:
        return fsdp_meta is None and ds_meta is None
    return fsdp_meta == ds_meta


def _print_result(prefix: str, ok: bool, info: str, top_entries, max_print: int):
    status = "OK" if ok else "MISMATCH"
    print(f"{prefix}: {status} {info}")
    if not ok and top_entries:
        print(f"  top{min(max_print, len(top_entries))} diffs (idx, fsdp, ds, abs_diff):")
        for coord, f_val, d_val, diff in top_entries:
            print(f"    {coord}: {f_val} vs {d_val} (diff={diff})")


def main():
    parser = argparse.ArgumentParser(description="Compare SP IO dumps between FSDP and DeepSpeed.")
    parser.add_argument("--fsdp_dir", required=True, help="Directory containing fsdp rank*_sp_io.pt dumps")
    parser.add_argument("--ds_dir", required=True, help="Directory containing deepspeed rank*_sp_io.pt dumps")
    parser.add_argument("--max_print", type=int, default=10, help="Number of top diffs to print per mismatch")
    args = parser.parse_args()

    fsdp_files = _collect_rank_files(args.fsdp_dir)
    ds_files = _collect_rank_files(args.ds_dir)
    shared_ranks = sorted(set(fsdp_files) & set(ds_files))
    if not shared_ranks:
        raise RuntimeError("No overlapping rank dumps found to compare.")

    stage_priority = {"local": 0, "hook_in": 1, "hook_out": 2, "none": 3}
    first_mismatch_by_rank = {}

    for rank in shared_ranks:
        fsdp_dump = torch.load(fsdp_files[rank], map_location="cpu")
        ds_dump = torch.load(ds_files[rank], map_location="cpu")

        fsdp_local = fsdp_dump.get("local", {})
        ds_local = ds_dump.get("local", {})
        hooks_f = fsdp_dump.get("hooks", {}) or {}
        hooks_d = ds_dump.get("hooks", {}) or {}

        stage = "none"
        local_keys = ["input_ids", "attention_mask", "position_ids", "response_mask", "responses"]
        extra_keys = [
            k
            for k in ("old_log_probs", "advantages", "dp_size", "batch_num_tokens", "global_batch_size", "values", "returns")
            if k in fsdp_local and k in ds_local
        ]
        for key in local_keys + extra_keys:
            if key not in fsdp_local or key not in ds_local:
                continue
            ok, info, top_entries = _compare_tensor(key, fsdp_local[key], ds_local[key], args.max_print)
            _print_result(f"[rank{rank}] local/{key}", ok, info, top_entries, args.max_print)
            if not ok and stage == "none":
                stage = "local"

        hook_in_ok, hook_out_ok = True, True
        if "layer0_vproj_in" in hooks_f and "layer0_vproj_in" in hooks_d:
            ok, info, top_entries = _compare_tensor(
                "layer0_vproj_in", hooks_f.get("layer0_vproj_in"), hooks_d.get("layer0_vproj_in"), args.max_print
            )
            _print_result(f"[rank{rank}] hook/layer0_vproj_in", ok, info, top_entries, args.max_print)
            meta_ok = _compare_meta(hooks_f.get("layer0_vproj_in_meta"), hooks_d.get("layer0_vproj_in_meta"))
            if not meta_ok:
                print(f"[rank{rank}] hook/layer0_vproj_in_meta: MISMATCH {hooks_f.get('layer0_vproj_in_meta')} vs {hooks_d.get('layer0_vproj_in_meta')}")
            hook_in_ok = ok and meta_ok
            if not hook_in_ok and stage == "none":
                stage = "hook_in"

        if "layer0_vproj_out" in hooks_f and "layer0_vproj_out" in hooks_d:
            ok, info, top_entries = _compare_tensor(
                "layer0_vproj_out", hooks_f.get("layer0_vproj_out"), hooks_d.get("layer0_vproj_out"), args.max_print
            )
            _print_result(f"[rank{rank}] hook/layer0_vproj_out", ok, info, top_entries, args.max_print)
            meta_ok = _compare_meta(hooks_f.get("layer0_vproj_out_meta"), hooks_d.get("layer0_vproj_out_meta"))
            if not meta_ok:
                print(f"[rank{rank}] hook/layer0_vproj_out_meta: MISMATCH {hooks_f.get('layer0_vproj_out_meta')} vs {hooks_d.get('layer0_vproj_out_meta')}")
            hook_out_ok = ok and meta_ok
            if not hook_out_ok and stage == "none":
                stage = "hook_out"

        first_mismatch_by_rank[rank] = stage

    overall_stage = "none"
    for st in first_mismatch_by_rank.values():
        if stage_priority[st] < stage_priority[overall_stage]:
            overall_stage = st

    print("\n=== First mismatch summary ===")
    for rank in shared_ranks:
        print(f"rank{rank}: {first_mismatch_by_rank[rank]}")
    print(f"Overall first mismatch stage: {overall_stage}")


if __name__ == "__main__":
    main()
