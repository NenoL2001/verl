#!/usr/bin/env python3
"""
Parse DeepSpeed zero-2 benchmark logs and plot key metrics.

Usage:
  python3 examples/benchmarks/fsdp_vs_deepspeed_ppo/plot_deepspeed_metrics.py \\
    --logs outputs/logs/deepspeed/dp*_zero2_*.log \\
    --outdir outputs/plots/deepspeed
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")


def _parse_value(raw: str) -> Optional[float]:
    raw = raw.strip().rstrip(",")
    if raw.startswith("np.float"):
        raw = raw.split("(", 1)[-1].rstrip(")")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_log(path: Path) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    with path.open("r") as f:
        for line in f:
            if "step:" not in line:
                continue
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            fields = m.group(2).split(" - ")
            row: Dict[str, float] = {"step": step}
            for field in fields:
                if ":" not in field:
                    continue
                key, val = field.split(":", 1)
                parsed = _parse_value(val)
                if parsed is not None:
                    row[key] = parsed
            records.append(row)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    return df.sort_values("step").reset_index(drop=True)


def plot_metrics(run_dfs: Dict[str, pd.DataFrame], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("perf/throughput", "Throughput (tokens/s)"),
        ("perf/max_memory_reserved_gb", "Max Memory Reserved (GB)"),
        ("perf/max_memory_allocated_gb", "Max Memory Allocated (GB)"),
        ("actor/pg_loss", "Actor PG Loss"),
        ("critic/vf_loss", "Critic VF Loss"),
        ("critic/rewards/mean", "Rewards Mean"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 14), sharex=True)
    for ax, (key, title) in zip(axes, metrics):
        for name, df in run_dfs.items():
            if key not in df:
                continue
            ax.plot(df["step"], df[key], label=name)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Step")
    axes[0].legend(loc="best")
    plt.tight_layout()
    fig_path = outdir / "deepspeed_zero2_metrics.png"
    plt.savefig(fig_path, dpi=200)
    print(f"[plot] saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        nargs="+",
        default=["outputs/logs/deepspeed/dp*_zero2_*.log"],
        help="Glob patterns to log files",
    )
    parser.add_argument("--outdir", default="outputs/plots/deepspeed", help="Output directory for plots/CSVs")
    args = parser.parse_args()

    paths: List[Path] = []
    for pattern in args.logs:
        paths.extend(Path(p) for p in glob.glob(pattern))

    if not paths:
        raise SystemExit("No log files found for given patterns.")

    run_dfs: Dict[str, pd.DataFrame] = {}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for path in sorted(set(paths)):
        df = parse_log(path)
        if df.empty:
            print(f"[warn] no step metrics parsed in {path}")
            continue
        name = path.stem
        run_dfs[name] = df
        csv_path = outdir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[parse] {path} -> {csv_path} ({len(df)} rows)")

    if run_dfs:
        plot_metrics(run_dfs, outdir)
    else:
        print("[warn] nothing to plot.")


if __name__ == "__main__":
    main()
