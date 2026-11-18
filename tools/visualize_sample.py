#!/usr/bin/env python3
"""Visualize one QVHighlights course annotation and its cached clip frames.

Shows the 2-second clip grid, relevant clips/windows, per-annotator saliency
scores, and (when available) the cached frames covering each ground-truth
window.

Usage:
    python tools/visualize_sample.py --split train --qid 0
    python tools/visualize_sample.py --split dev --qid 39 --output outputs/visualizations/qid39.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

CLIP_SECONDS = 2.0


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_row(data_dir: Path, split: str, qid: int) -> dict:
    name = "final_queries.jsonl" if split == "final" else f"{split}.jsonl"
    for row in read_jsonl(data_dir / name):
        if row["qid"] == qid:
            return row
    raise KeyError(f"qid {qid} not found in {data_dir / name}")


def load_cache_index(cache_dir: Path) -> dict[str, list[dict]]:
    index_path = cache_dir / "index.jsonl"
    entries: dict[str, list[dict]] = {}
    if not index_path.is_file():
        return entries
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entry = json.loads(line)
                entries.setdefault(entry["vid"], []).append(entry)
    return entries


def best_cache_entry(entries: list[dict], start: float, end: float) -> dict | None:
    overlapping = [entry for entry in entries
                   if float(entry["start"]) < end and float(entry["end"]) > start]
    if not overlapping:
        return None
    exact = [entry for entry in overlapping
             if abs(float(entry["start"]) - start) < 1e-6 and abs(float(entry["end"]) - end) < 1e-6]
    candidates = exact or overlapping
    return max(candidates, key=lambda entry: min(float(entry["end"]), end) - max(float(entry["start"]), start))


def load_frames(entry: dict, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_dir / entry["cache_file"], allow_pickle=False) as archive:
        return archive["frames"], archive["actual_times"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/course")
    parser.add_argument("--cache-dir", default="data/reference_frame_cache")
    parser.add_argument("--split", choices=["train", "dev", "final"], default="train")
    parser.add_argument("--qid", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir, cache_dir = Path(args.data_dir), Path(args.cache_dir)
    row = select_row(data_dir, args.split, args.qid)
    duration = float(row["duration"])
    n_clips = int(round(duration / CLIP_SECONDS))
    relevant_ids = [int(value) for value in row.get("relevant_clip_ids", [])]
    windows = [tuple(map(float, window)) for window in row.get("relevant_windows", [])]
    saliency = np.asarray(row.get("saliency_scores", []), dtype=float)
    cache_entries = load_cache_index(cache_dir).get(row["vid"], [])

    window_frames = []
    for start, end in windows:
        entry = best_cache_entry(cache_entries, start, end)
        if entry is None:
            window_frames.append((start, end, None, None, None))
        else:
            frames, times = load_frames(entry, cache_dir)
            window_frames.append((start, end, frames, times, entry))

    n_frame_rows = max(1, len(windows))
    figure = plt.figure(figsize=(14, 8.2 + 1.9 * (n_frame_rows - 1)))
    grid = figure.add_gridspec(2 + n_frame_rows, 1, height_ratios=[1.0, 1.15] + [1.15] * n_frame_rows, hspace=0.45)
    figure.suptitle(f"qid {row['qid']}  |  {row['vid']}  |  {duration:.0f}s", fontsize=14, fontweight="bold", y=0.985)

    # Timeline: one cell per 2-second clip; green cells are annotated relevant clips.
    timeline = figure.add_subplot(grid[0])
    timeline.set_title(f'Query: "{row["query"]}"', loc="left", fontsize=10)
    for index in range(n_clips):
        left = index * CLIP_SECONDS
        timeline.add_patch(Rectangle((left + 0.08, 0), CLIP_SECONDS - 0.16, 1,
                                     facecolor="#d9d9d9", edgecolor="white", linewidth=0.4))
    for index in relevant_ids:
        if 0 <= index < n_clips:
            timeline.add_patch(Rectangle((index * CLIP_SECONDS + 0.08, 0), CLIP_SECONDS - 0.16, 1,
                                         facecolor="#2ca02c", edgecolor="white", linewidth=0.4))
            timeline.text((index + 0.5) * CLIP_SECONDS, 1.01, str(index), rotation=90,
                          ha="center", va="bottom", fontsize=6, color="#1a6b1a")
    for start, end in windows:
        timeline.add_patch(Rectangle((start, -0.22), end - start, 1.44, facecolor="#2ca02c",
                                     alpha=0.16, edgecolor="#2ca02c", linewidth=1.5))
        timeline.text((start + end) / 2, 1.3, f"GT [{start:g}, {end:g}] s",
                      ha="center", va="bottom", fontsize=9, color="#1a6b1a")
    timeline.set_xlim(0, duration)
    timeline.set_ylim(-0.32, 1.5)
    timeline.set_yticks([])
    timeline.set_xlabel("Time (s)")
    timeline.grid(axis="x", color="#eeeeee", linewidth=0.6)

    # Saliency: one line per annotator, over the relevant clips only.
    saliency_axis = figure.add_subplot(grid[1])
    if relevant_ids and len(saliency) == len(relevant_ids) and saliency.ndim == 2:
        centers = np.asarray([(index + 0.5) * CLIP_SECONDS for index in relevant_ids])
        for annotator in range(saliency.shape[1]):
            saliency_axis.plot(centers, saliency[:, annotator], marker="o", markersize=4,
                               alpha=0.75, label=f"annotator {annotator + 1}")
        saliency_axis.plot(centers, saliency.mean(axis=1), color="black", linestyle="--",
                           linewidth=1.2, label="mean")
        saliency_axis.set_ylim(0.5, 5.5)
        saliency_axis.set_yticks([1, 2, 3, 4, 5])
        saliency_axis.set_ylabel("Saliency (1-5)")
        saliency_axis.legend(loc="upper right", ncol=4, fontsize=8)
        for start, end in windows:
            saliency_axis.axvspan(start, end, color="#2ca02c", alpha=0.08)
    else:
        saliency_axis.text(0.5, 0.5, "No saliency scores for this split/query",
                           ha="center", va="center", transform=saliency_axis.transAxes)
    saliency_axis.set_xlim(0, duration)
    saliency_axis.set_xlabel("Time (s)")
    saliency_axis.grid(axis="y", color="#eeeeee", linewidth=0.6)

    # Cached frames: one row per ground-truth window, up to four frames per row.
    for row_index, (start, end, frames, times, entry) in enumerate(window_frames):
        subgrid = grid[2 + row_index].subgridspec(1, 4, wspace=0.05)
        if frames is None:
            axis = figure.add_subplot(subgrid[0, :])
            axis.axis("off")
            axis.text(0.5, 0.5, f"No cached frames for GT [{start:g}, {end:g}] s",
                      ha="center", va="center", transform=axis.transAxes)
            continue
        for frame_index in range(min(4, len(frames))):
            axis = figure.add_subplot(subgrid[0, frame_index])
            axis.imshow(frames[frame_index])
            axis.set_title(f"{float(times[frame_index]):.2f}s", fontsize=8)
            axis.axis("off")

    figure.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.04, hspace=0.55)
    output = args.output or Path("outputs/visualizations") / f"{args.split}_qid{args.qid}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "qid": row["qid"], "vid": row["vid"], "split": args.split,
        "duration": duration, "query": row["query"],
        "relevant_clip_ids": relevant_ids,
        "relevant_windows": windows,
        "saliency_scores": saliency.tolist(),
        "cached_windows": [{"start": start, "end": end, "frames": 0 if frames is None else int(len(frames)),
                            "cache_file": None if entry is None else entry["cache_file"]}
                           for start, end, frames, times, entry in window_frames],
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
