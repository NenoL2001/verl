import argparse
import torch


TARGET_KEYS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "lm_head.weight",
    "value_head.weight",
]


def _normalize_keys(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        nk = k.replace("_fsdp_wrapped_module.", "")
        out[nk] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare state_dict slices between FSDP and DeepSpeed parity runs.")
    parser.add_argument("--fsdp_pt", required=True, help="Path to fsdp rank0.pt")
    parser.add_argument("--ds_pt", required=True, help="Path to deepspeed rank0.pt")
    args = parser.parse_args()

    fsdp_obj = torch.load(args.fsdp_pt, map_location="cpu")
    ds_obj = torch.load(args.ds_pt, map_location="cpu")
    fsdp_state = _normalize_keys(fsdp_obj.get("before_state", {}))
    ds_state = _normalize_keys(ds_obj.get("before_state", {}))

    missing = [k for k in TARGET_KEYS if k not in fsdp_state or k not in ds_state]
    if missing:
        print("Missing keys:", missing)
        raise SystemExit(1)

    all_ok = True
    for k in TARGET_KEYS:
        f = fsdp_state[k]
        d = ds_state[k]
        if f.shape != d.shape or f.dtype != d.dtype:
            print(f"{k}: shape/dtype mismatch fsdp={f.shape}/{f.dtype} ds={d.shape}/{d.dtype}")
            all_ok = False
            continue
        diff = (f.to(torch.float64) - d.to(torch.float64)).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        print(f"{k}: max_abs={max_abs} mean_abs={mean_abs}")
        if max_abs != 0:
            all_ok = False
    if not all_ok:
        raise SystemExit("State dicts differ (SYNC_INIT may not have taken effect).")
    print("All target keys match exactly.")


if __name__ == "__main__":
    main()
