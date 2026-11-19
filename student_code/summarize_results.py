#!/usr/bin/env python3
"""Summarize formal three-seed A or C results with sample standard deviation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

SEEDS = (42, 2026, 3407)


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["a", "c"])
    parser.add_argument("--root", required=True,
                        help="Directory containing seed42, seed2026, seed3407")
    args = parser.parse_args()
    root = Path(args.root)
    if args.kind == "a":
        values, epochs = [], []
        for seed in SEEDS:
            summary = json.loads((root / f"seed{seed}" / "summary.json").read_text())
            values.append(float(summary["best_dev_mAP"]))
            epochs.append(int(summary["best_epoch"]))
        mean, std = mean_std(values)
        print(json.dumps({"seeds": list(SEEDS), "best_epochs": epochs,
                          "mAP": values, "mean": mean, "sample_std": std}, indent=2))
        return

    rows = []
    for step in (312, 624, 936):
        values = []
        for seed in SEEDS:
            score = json.loads((root / f"seed{seed}" / f"dev_step{step}_score.json").read_text())
            values.append(float(score["metrics"]["mAP"]))
        mean, std = mean_std(values)
        rows.append({"step": step, "mAP": values, "mean": mean, "sample_std": std})
    best = max(rows, key=lambda row: row["mean"])
    print(json.dumps({"seeds": list(SEEDS), "steps": rows,
                      "selected_common_step": best["step"]}, indent=2))


if __name__ == "__main__":
    main()
