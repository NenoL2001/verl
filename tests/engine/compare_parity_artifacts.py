"""
Compare parity artefacts saved by sp_parity_runner_qwen05b.py.

Usage:
  python3 tests/engine/compare_parity_artifacts.py --fsdp /tmp/verl_sp_parity/<run>/fsdp/rank0.pt --ds /tmp/verl_sp_parity/<run>/deepspeed/rank0.pt
"""

import argparse
import torch


def load(path):
    return torch.load(path, map_location="cpu")


def compare_metrics(fsdp_m, ds_m, atol=1e-6, rtol=1e-7):
    keys = set(fsdp_m[0].keys()) & set(ds_m[0].keys()) if fsdp_m and ds_m else set()
    for step_idx, (m_f, m_d) in enumerate(zip(fsdp_m, ds_m)):
        for k in keys:
            if k.startswith("perf/") or k.startswith("debug/grad_topk"):
                continue
            v1, v2 = m_f.get(k), m_d.get(k)
            if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
                continue
            t1 = torch.tensor(v1)
            t2 = torch.tensor(v2)
            try:
                torch.testing.assert_close(t1, t2, atol=atol, rtol=rtol)
            except AssertionError as exc:
                print(f"Mismatch at step {step_idx} key {k}: {exc}")
                return False
    return True


def compare_state(delta_f, delta_d, atol=1e-3, rtol=1e-3):
    keys = set(delta_f.keys()) & set(delta_d.keys())
    for k in keys:
        try:
            torch.testing.assert_close(delta_f[k], delta_d[k], atol=atol, rtol=rtol)
        except AssertionError as exc:
            print(f"Param delta mismatch {k}: {exc}")
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdp", required=True)
    ap.add_argument("--ds", required=True)
    args = ap.parse_args()

    fsdp_obj = load(args.fsdp)
    ds_obj = load(args.ds)

    ok_metrics = compare_metrics(fsdp_obj.get("metrics", []), ds_obj.get("metrics", []))
    ok_state = compare_state(fsdp_obj.get("delta", {}), ds_obj.get("delta", {}))
    if ok_metrics and ok_state:
        print("All keys matched")
    else:
        raise SystemExit("Parity mismatch")


if __name__ == "__main__":
    main()
