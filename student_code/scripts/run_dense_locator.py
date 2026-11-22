"""提交两种密集定位头各三种子到 3090 的空闲 GPU 队列。"""

import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/dense_locator_20261122"
SEEDS = (42, 2026, 3407)
MATRIX = [(f"dense_{scale}", scale, 1.5e-4, 0) for scale in ("multi", "single")]


def command(task, run):
    """用固定预算从最近可用的 epoch checkpoint 继续。"""
    _, scale, lr, _, seed = task
    argv = [sys.executable, "-u", "-m", "student_code.qmt.train_dense_locator", "train",
            "--output-dir", str(run), "--scale", scale, "--seed", str(seed), "--lr", str(lr),
            "--epochs", "90", "--batch-size", "32", "--eval-every", "5",
            "--feature-root", str(ROOT / "data/features"),
            "--train-annotations", str(ROOT / "data/course/train.jsonl"),
            "--dev-annotations", str(ROOT / "data/course/dev.jsonl")]
    checkpoints = [p for p in run.parent.glob("attempt*/model_last.pt")
                   if p.parent.name.removeprefix("attempt").isdigit()]
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.parent.name.removeprefix("attempt")))
        argv.extend(("--resume", str(latest)))
    return argv


def paired_report():
    """配对汇总两组实际完成的 dev 指标；best epoch 按每个种子独立选取。"""
    results = {}
    for name, *_ in MATRIX:
        records = []
        for seed in SEEDS:
            marker = json.loads((OUT / name / f"seed{seed}" / "complete.json").read_text())
            records.append(json.loads(Path(marker["summary"]).read_text()))
        values = [row["best_dev_mAP"] for row in records]
        results[name] = {
            "epochs": [row["best_epoch"] for row in records], "mAP": values,
            "mean": statistics.mean(values), "sample_std": statistics.stdev(values),
        }
    paired = [a - b for a, b in zip(results["dense_multi"]["mAP"], results["dense_single"]["mAP"])]
    return dict(seeds=SEEDS, results=results, multi_minus_single=paired,
                paired_mean=statistics.mean(paired), paired_sample_std=statistics.stdev(paired))


def main():
    """预览矩阵或复用服务器上的空闲 GPU、互斥及磁盘保护。"""
    if sys.argv[1:] == ["--dry-run"]:
        for config in MATRIX:
            for seed in SEEDS:
                run = OUT / config[0] / f"seed{seed}" / "attempt1"
                print(json.dumps(command((*config, seed), run)))
        print("6 tasks; 90 epochs each; no training launched")
        return 0
    if sys.argv[1:] == ["--report"]:
        print(json.dumps(paired_report(), indent=2))
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parent / "local"))
    import run_foreground_queue as queue

    queue.OUT, queue.MATRIX, queue.command = OUT, MATRIX, command
    status = queue.main()
    if status == 0:
        temporary = OUT / "paired_summary.tmp"
        temporary.write_text(json.dumps(paired_report(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(OUT / "paired_summary.json")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
