"""密集窗口的跨层 Soft-NMS、Top-3 导出和候选覆盖诊断。"""

from itertools import permutations

import numpy as np

from scripts.evaluate import tiou
from scripts.locator_common import sanitize_window


def rank_windows(spans, scores, duration, limit=10):
    """先量化到课程网格，再跨层去重和做 Gaussian Soft-NMS。

    Args:
        spans (array, "P 2"): 有效位置预测的所有起止时间。
        scores (array, "P"): 相关性概率乘以 IoU 质量概率。
        duration (float): 对应视频的总时长。
        limit (int): 最多保留的不同候选数量。

    Returns:
        list[list]: [start, end, score] 按调整后分数降序排列。
    """
    unique = {}
    for (start, end), score in zip(spans, scores, strict=True):
        window = sanitize_window(start, end, duration)
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("Non-finite dense score")
        unique[window] = max(score, unique.get(window, 0.0))
    ranked = []
    while unique and len(ranked) < limit:
        best = max(unique, key=unique.get)
        ranked.append([*best, round(unique.pop(best), 8)])
        for window in unique:
            unique[window] *= np.exp(-(tiou(window, best) ** 2) / 0.5)
    return ranked


def top_three(ranked):
    """独立候选少于三个时，用已预测窗口补齐课程输出契约。"""
    if not ranked:
        raise ValueError("Dense locator produced no valid positions")
    return ranked[:3] + [ranked[-1]] * max(0, 3 - len(ranked))


def oracle_order(windows, truth):
    """穷举 Top-10 到 Top-3 的有序选择，并按官方独占匹配计算 AP。

    Args:
        windows (list[list]): 每个候选包含起止时间和原始分数。
        truth (list[list]): 查询的全部 GT 窗口。

    Returns:
        list[list]: AP 最优的三个窗口，分数只编码 oracle 顺序。
    """
    if len(windows) < 3:
        windows = top_three(windows)
    orders = np.array(list(permutations(range(len(windows)), 3)))
    ious = np.array([[tiou(window, gt) for gt in truth] for window in windows])
    thresholds = np.array([i / 100 for i in range(50, 100, 5)])
    gt_order = ious.argsort(axis=1)[:, ::-1]
    used = np.zeros((len(orders), len(thresholds), len(truth)), dtype=bool)
    hits = []
    for indices in orders.T:
        sorted_gt = gt_order[indices]
        values = np.take_along_axis(ious[indices], sorted_gt, axis=1)
        available = ~np.take_along_axis(used, sorted_gt[:, None, :], axis=2)
        eligible = (values[:, None, :] >= thresholds[None, :, None]) & available
        hit = eligible.any(axis=2)
        matches = sorted_gt[np.arange(len(orders))[:, None], eligible.argmax(axis=2)]
        used[np.arange(len(orders))[:, None], np.arange(len(thresholds)), matches] |= hit
        hits.append(hit)
    hits = np.stack(hits, axis=-1)
    precision = hits.cumsum(axis=-1) / np.arange(1, 4)
    envelope = np.maximum.accumulate(precision[..., ::-1], axis=-1)[..., ::-1]
    ap = (hits * envelope).sum(axis=-1).mean(axis=-1) / len(truth)
    best = orders[int(ap.argmax())]
    return [[*windows[index][:2], float(3 - rank)] for rank, index in enumerate(best)]
