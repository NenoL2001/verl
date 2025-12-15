import argparse
import os
from typing import Dict, Iterable, Tuple

import torch

TAGS = ("pre_fwd", "post_fwd_pre_bwd", "post_bwd_pre_step", "post_step")


def _load(path: str):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a - b).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (b.abs() + 1e-12)).max().item()
    return max_abs, max_rel


def _is_number(x):
    return isinstance(x, (int, float)) or (isinstance(x, torch.Tensor) and x.numel() == 1)


def _summarize_tensor(t: torch.Tensor) -> str:
    t = t.detach().cpu().float()
    flat = t.reshape(-1)
    preview = flat[:5].tolist()
    return f"shape={tuple(t.shape)}, dtype={t.dtype}, mean={t.mean().item():.6g}, std={t.std().item():.6g}, first={preview}"


def _as_tensor(x: torch.Tensor):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


def _compare_payload(fsdp_p: Dict, ds_p: Dict) -> Dict:
    report = {}
    common_keys = set(fsdp_p.keys()) & set(ds_p.keys())
    for k in sorted(common_keys):
        if isinstance(fsdp_p[k], torch.Tensor) and isinstance(ds_p[k], torch.Tensor):
            report[k] = _compare_tensors(fsdp_p[k], ds_p[k])
        elif _is_number(fsdp_p[k]) and _is_number(ds_p[k]):
            report[k] = _compare_tensors(_as_tensor(fsdp_p[k]).float(), _as_tensor(ds_p[k]).float())
        elif isinstance(fsdp_p[k], dict) and isinstance(ds_p[k], dict):
            # one level nested dict of tensors
            nested = {}
            for nk in sorted(set(fsdp_p[k].keys()) & set(ds_p[k].keys())):
                if isinstance(fsdp_p[k][nk], torch.Tensor) and isinstance(ds_p[k][nk], torch.Tensor):
                    nested[nk] = _compare_tensors(fsdp_p[k][nk], ds_p[k][nk])
                elif _is_number(fsdp_p[k][nk]) and _is_number(ds_p[k][nk]):
                    nested[nk] = _compare_tensors(
                        _as_tensor(fsdp_p[k][nk]).float(), _as_tensor(ds_p[k][nk]).float()
                    )
            report[k] = nested
    return report


def compare_dirs(fsdp_dir: str, ds_dir: str, steps: Iterable[int], show_samples: bool = False) -> None:
    any_fail = False
    for step in steps:
        print(f"\n=== Step {step} ===")
        for tag in TAGS:
            f_path = os.path.join(fsdp_dir, f"debug_rank0_step{step}_{tag}.pt")
            d_path = os.path.join(ds_dir, f"debug_rank0_step{step}_{tag}.pt")
            fsdp_p = _load(f_path)
            ds_p = _load(d_path)
            if fsdp_p is None or ds_p is None:
                print(f"{tag}: missing file (fsdp exists? {fsdp_p is not None}, ds exists? {ds_p is not None})")
                any_fail = True
                continue
            if isinstance(fsdp_p, dict):
                print(f"{tag}: fsdp keys={list(fsdp_p.keys())[:6]}")
            if isinstance(ds_p, dict):
                print(f"{tag}: ds   keys={list(ds_p.keys())[:6]}")
            if show_samples:
                for probe in ("vpreds", "vf_loss", "loss", "grad_norm"):
                    if isinstance(fsdp_p, dict) and probe in fsdp_p and isinstance(fsdp_p[probe], torch.Tensor):
                        print(f"  fsdp {probe}: {_summarize_tensor(fsdp_p[probe])}")
                    if isinstance(ds_p, dict) and probe in ds_p and isinstance(ds_p[probe], torch.Tensor):
                        print(f"  ds   {probe}: {_summarize_tensor(ds_p[probe])}")
            report = _compare_payload(fsdp_p, ds_p)
            if not report:
                print(f"{tag}: no common keys")
                continue
            worst = 0.0
            worst_k = None
            for k, v in report.items():
                if isinstance(v, tuple):
                    if v[0] > worst:
                        worst = v[0]
                        worst_k = k
                elif isinstance(v, dict):
                    for nk, vv in v.items():
                        if vv[0] > worst:
                            worst = vv[0]
                            worst_k = f"{k}.{nk}"
            print(f"{tag}: worst_diff={worst:.6g} at {worst_k}")
            if worst > 1e-6:
                any_fail = True
    if any_fail:
        raise SystemExit("Mismatch detected, see logs above")
    print("All compared tags within tolerance.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp_dir", required=True)
    parser.add_argument("--ds_dir", required=True)
    parser.add_argument("--steps", type=str, default="0")
    parser.add_argument("--show_samples", action="store_true", help="Print tensor summaries for common keys.")
    args = parser.parse_args()
    steps = [int(x) for x in args.steps.split(",") if x != ""]
    compare_dirs(args.fsdp_dir, args.ds_dir, steps, show_samples=args.show_samples)


if __name__ == "__main__":
    main()
