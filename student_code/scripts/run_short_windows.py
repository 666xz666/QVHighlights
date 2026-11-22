"""在 3090 的空闲 GPU 队列中运行三种子短窗口实验。"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/short_windows_20261122"
BACKBONES = ROOT / "outputs/anchors_20261122/cross_tcn_anchor_multiscale"
MATRIX = [("short_window_gate", "short-windows", 3e-4, 0)]
EPOCHS = 20


def command(task, run):
    """生成与对应种子主干绑定的训练命令，并选择最近的完整状态。"""
    _, _, lr, _, seed = task
    marker = BACKBONES / f"seed{seed}" / "complete.json"
    backbone = Path(json.loads(marker.read_text())["checkpoint"])
    if not backbone.is_file():
        raise FileNotFoundError(backbone)
    cmd = [sys.executable, "-u", "-m", "student_code.qmt.train_short_windows",
           "--backbone", str(backbone), "--seed", str(seed), "--lr", str(lr),
           "--feature-root", "data/features", "--annotations", "data/course/dev.jsonl",
           "--train-annotations", "data/course/train.jsonl", "--output-dir", str(run),
           "--epochs", str(EPOCHS), "--batch-size", "32", "--eval-every", "5"]
    checkpoints = [p for p in run.parent.glob("attempt*/model_last.pt")
                   if p.parent != run and p.parent.name.removeprefix("attempt").isdigit()]
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.parent.name.removeprefix("attempt")))
        cmd.extend(("--resume", str(latest)))
    return cmd


def main():
    """复用 3090 本机的空闲 GPU、输出锁、超时和磁盘保护。"""
    if sys.argv[1:] == ["--dry-run"]:
        for task in ((*MATRIX[0], seed) for seed in (42, 2026, 3407)):
            run = OUT / task[0] / f"seed{task[4]}" / "attempt1"
            print(json.dumps(command(task, run)))
        print(f"3 tasks; {EPOCHS} epochs each; dry run")
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parent / "local"))
    import run_foreground_queue as queue

    queue.OUT, queue.MATRIX, queue.command = OUT, MATRIX, command
    return queue.main()


if __name__ == "__main__":
    raise SystemExit(main())
