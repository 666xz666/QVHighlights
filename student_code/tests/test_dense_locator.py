"""验证密集候选排序，用真实特征冒烟训练、断点恢复与无标签导出。"""

import argparse
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from scripts.prepare_course_data import read_jsonl, write_jsonl
from student_code.qmt.dense_rank import oracle_order, rank_windows, top_three


ROOT = Path(__file__).resolve().parents[2]


def check_ranking(official_ap):
    """用官方 AP 穷举验证 oracle 的排序、重复匹配和精确阈值。"""
    ranked = rank_windows([(0, 9.1), (0, 10), (0, 12), (20, 24)], [0.8, 0.9, 0.7, 0.3], 32)
    assert [window[:2] for window in ranked] == [[0.0, 10.0], [20.0, 24.0], [0.0, 12.0]]
    assert all(a[2] >= b[2] for a, b in zip(ranked, ranked[1:]))
    assert len(top_three(ranked[:1])) == 3
    rng = np.random.default_rng(3)
    def to_dict(a, b, score):
        return {"video-id": 1, "t-start": a, "t-end": b, "score": score}
    trials = [[(0, 10), (2, 10), (10, 14), (16, 22)]]
    for _ in range(16):
        start = rng.integers(0, 15, 6)
        trials.append(list(zip(start, start + rng.integers(1, 12, 6))))
    thresholds = [i / 100 for i in range(50, 100, 5)]
    truth = [(0, 10), (0, 9), (10, 14), (18, 22)]
    for spans in trials:
        windows = [[*span, 0.0] for span in spans]
        candidates = oracle_order(windows, truth)

        def score(order):
            return np.mean(official_ap([to_dict(*gt, 0) for gt in truth],
                                       [to_dict(*w) for w in order], thresholds))
        expected = max(score([[*span, 3 - i] for i, span in enumerate(order)])
                       for order in itertools.permutations(spans, 3))
        assert np.isclose(score(candidates), expected, atol=1e-12, rtol=0)
    print("Soft-NMS and 17 exact oracle comparisons passed", flush=True)


def smoke(root, device):
    """在两个真实训练查询和两个 dev 查询上验证两种尺度的完整链路。"""
    import torch

    from student_code.qmt.train_dense_locator import export, train

    torch.set_num_threads(1)
    with TemporaryDirectory(prefix="qvh_dense_smoke_") as temporary:
        tmp = Path(temporary)
        annotations = root / "data/course"
        train_rows = read_jsonl(annotations / "train.jsonl")[:2]
        dev_rows = read_jsonl(annotations / "dev.jsonl")[:2]
        write_jsonl(tmp / "train.jsonl", train_rows)
        write_jsonl(tmp / "dev.jsonl", dev_rows)
        write_jsonl(tmp / "unlabeled.jsonl", [
            {key: row[key] for key in ("qid", "vid", "query", "duration", "split")} for row in dev_rows
        ])
        for scale in ("multi", "single"):
            base = dict(epochs=2, batch_size=2, eval_every=1, workers=0, lr=1.5e-4,
                        seed=42, scale=scale, device=device, feature_root=root / "data/features",
                        train_annotations=tmp / "train.jsonl", dev_annotations=tmp / "dev.jsonl")
            reference, restored = tmp / f"{scale}_reference", tmp / f"{scale}_restored"
            train(SimpleNamespace(**base, output_dir=reference, resume=None))
            train(SimpleNamespace(**base, output_dir=restored, resume=None), stop_after=1)
            last = restored / "model_last.pt"
            first = torch.load(last, map_location="cpu", weights_only=False)
            assert first["metadata"]["epoch"] == 1
            train(SimpleNamespace(**base, output_dir=restored, resume=last))
            final = torch.load(last, map_location="cpu", weights_only=False)
            expected = torch.load(reference / "model_last.pt", map_location="cpu", weights_only=False)
            assert final["metadata"]["epoch"] == 2
            for name, tensor in final["model"].items():
                torch.testing.assert_close(tensor, expected["model"][name], atol=0, rtol=0)
            assert not torch.equal(first["model"]["head.weight"], final["model"]["head.weight"])
            assert not torch.equal(first["model"]["head.bias"], final["model"]["head.bias"])
            assert len(read_jsonl(restored / "train_log.jsonl")) == 2
            predicted = tmp / f"{scale}_unlabeled.jsonl"
            export(SimpleNamespace(checkpoint=restored / "model_best.pt",
                                   annotations=tmp / "unlabeled.jsonl", split="dev",
                                   output=predicted,
                                   **{k: base[k] for k in ("batch_size", "workers", "device", "feature_root")}))
            assert read_jsonl(predicted) == read_jsonl(restored / "dev_best_top3.jsonl")
            print(f"{scale}: full-dimension train, exact resume and unlabeled export passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    from standalone_eval.utils import compute_average_precision_detection

    check_ranking(compute_average_precision_detection)
    if not args.rank_only:
        smoke(ROOT, args.device)
