"""固定主定位器首窗，用独立边界候选的一致性重排后两窗。"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scripts.evaluate import evaluate
from scripts.prepare_course_data import read_jsonl, write_jsonl


SEEDS = (42, 2026, 3407)
MODES = {"original": None, "tail-overlap": math.inf, "tail-blend-025": 0.25,
         "tail-blend-050": 0.5}
AUXILIARY_RANK_WEIGHTS = (1.0, 0.7, 0.5)


def temporal_iou(first, second):
    """计算两个秒级闭开区间的 tIoU。

    Args:
        first (list[float]): 第一个 ``[start, end]`` 区间。
        second (list[float]): 第二个 ``[start, end]`` 区间。

    Returns:
        float: 范围为 0 到 1 的交并比。
    """
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = first[1] - first[0] + second[1] - second[0] - intersection
    return intersection / union if union > 0 else 0.0


def overlap_support(candidate, auxiliary):
    """以辅助候选的名次加权最大重叠度衡量边界一致性。

    Args:
        candidate (list[float]): 主定位器的 ``[start, end, score]`` 候选。
        auxiliary (list[list[float]]): 独立边界定位器的有序候选列表。

    Returns:
        float: 与前三个辅助候选的最大名次加权 tIoU。
    """
    return max((weight * temporal_iou(candidate, window)
                for weight, window in zip(AUXILIARY_RANK_WEIGHTS, auxiliary, strict=False)),
               default=0.0)


def rerank_tail(primary, auxiliary, blend):
    """保持首窗逐值不变，仅按边界一致性重排后两窗。

    Args:
        primary (list[list[float]]): 主定位器的三个有序候选。
        auxiliary (list[list[float]]): 独立边界定位器的有序候选。
        blend (float): 一致性相对名次的权重；正无穷表示完全按一致性排序。

    Returns:
        list[list[float]]: 首窗不变且分数严格递减的三个候选。
    """
    if len(primary) != 3:
        raise ValueError(f"Expected exactly three primary windows, got {len(primary)}")
    if not auxiliary:
        raise ValueError("Expected at least one auxiliary window")
    if not math.isfinite(primary[0][2]):
        raise ValueError("The locked first-window score must be finite")

    supports = [overlap_support(window, auxiliary) for window in primary[1:]]
    if math.isinf(blend):
        order = sorted(range(2), key=lambda index: (-supports[index], index))
    else:
        evidence_order = sorted(range(2), key=lambda index: (-supports[index], index))
        evidence_rank = {index: 1 - rank for rank, index in enumerate(evidence_order)}
        prior = (0.7, 0.5)
        order = sorted(range(2), key=lambda index: (-(prior[index] + blend * evidence_rank[index]), index))

    result = [list(primary[0]), *(list(primary[1 + index]) for index in order)]
    for rank in (1, 2):
        result[rank][2] = math.nextafter(result[rank - 1][2], -math.inf)
    return result


def indexed(rows, name):
    """按 qid 建立唯一索引，并拒绝静默覆盖重复记录。"""
    result = {row["qid"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate qid in {name}")
    return result


def evaluate_seed(seed, primary_path, auxiliary_path, labels, output):
    """生成一个种子的全部预注册重排结果并核验 R1 不变。"""
    primary = read_jsonl(primary_path)
    auxiliary = indexed(read_jsonl(auxiliary_path), str(auxiliary_path))
    label_ids = [row["qid"] for row in labels]
    if [row["qid"] for row in primary] != label_ids or set(auxiliary) != set(label_ids):
        raise ValueError(f"Seed {seed}: primary, auxiliary and labels must contain identical ordered qids")

    reports = {}
    baseline_r1 = None
    for mode, blend in MODES.items():
        predictions = []
        changed = 0
        for row in primary:
            windows = row["pred_relevant_windows"]
            ranked = windows if blend is None else rerank_tail(
                windows, auxiliary[row["qid"]]["pred_relevant_windows"], blend)
            changed += int([window[:2] for window in ranked[1:]] !=
                           [window[:2] for window in windows[1:]])
            predictions.append({**row, "pred_relevant_windows": ranked})
        report = evaluate(predictions, labels)
        current_r1 = report["detailed"]["full"]["MR-R1"]
        baseline_r1 = current_r1 if baseline_r1 is None else baseline_r1
        if current_r1 != baseline_r1:
            raise ValueError(f"Seed {seed}: {mode} changed full R1 despite locking the first window")
        reports[mode] = {"brief": report["brief"], "changed_tail_queries": changed}
        write_jsonl(output / f"seed{seed}_{mode}.jsonl", predictions)
    return reports


def aggregate(per_seed):
    """汇总三种子均值、样本标准差与相对原版的同种子差值。"""
    result = {"seeds": list(per_seed), "modes": {}}
    for mode in MODES:
        values = [per_seed[seed][mode]["brief"]["MR-full-mAP"] for seed in per_seed]
        baseline = [per_seed[seed]["original"]["brief"]["MR-full-mAP"] for seed in per_seed]
        deltas = [value - reference for value, reference in zip(values, baseline, strict=True)]
        result["modes"][mode] = {
            "seed_mAP": values,
            "mAP_mean": mean(values),
            "mAP_sample_std": stdev(values),
            "paired_deltas": deltas,
            "paired_mean": mean(deltas),
            "paired_sample_std": stdev(deltas),
            "changed_tail_queries": [per_seed[seed][mode]["changed_tail_queries"] for seed in per_seed],
            "brief": [per_seed[seed][mode]["brief"] for seed in per_seed],
        }
    return result


def main():
    """读取冻结预测，执行三种子无训练组合评估。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-template", type=str, required=True,
                        help="含 {seed} 的 AP-locked 预测路径模板")
    parser.add_argument("--auxiliary-template", type=str, required=True,
                        help="含 {seed} 的独立边界预测路径模板")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if "{seed}" not in args.primary_template or "{seed}" not in args.auxiliary_template:
        parser.error("Both prediction templates must contain {seed}")
    if tuple(args.seeds) != SEEDS:
        parser.error(f"Formal comparison requires seeds {SEEDS}")

    args.output.mkdir(parents=True, exist_ok=False)
    labels = read_jsonl(args.ground_truth)
    per_seed = {}
    for seed in args.seeds:
        per_seed[seed] = evaluate_seed(
            seed, Path(args.primary_template.format(seed=seed)),
            Path(args.auxiliary_template.format(seed=seed)), labels, args.output)
    summary = aggregate(per_seed)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
