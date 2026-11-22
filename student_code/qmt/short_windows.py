"""冻结定位器的短窗口分支，负责网格、区间打分及第 3 项替换。"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.evaluate import tiou
from standalone_eval.utils import compute_average_precision_detection

MAX_WIDTH = 5
CLIP_SECONDS = 2


def short_truth(row):
    """从标注行筛选官方 (0, 10] 秒桶的全部区间。"""
    return [w for w in row["relevant_windows"] if 0 < w[1] - w[0] <= 10]


def grid_features(memory, lengths):
    """池化区间与左右各两个 clip，输出窗口边界和有效性掩码。

    Args:
        memory (Float[Tensor, "B T D"]): 冻结 encoder 的视频表示。
        lengths (Long[Tensor, "B"]): 视频内完整且有特征的 clip 数。

    Returns:
        tuple: features [B W 4D+3]、windows [W 2]（秒）、valid [B W]。
    """
    # 步骤1：窗口顺序固定为宽度优先、起点升序，用于同分时稳定选择。
    starts, widths = torch.meshgrid(
        torch.arange(memory.size(1), device=memory.device),
        torch.arange(1, MAX_WIDTH + 1, device=memory.device), indexing="xy",
    )
    starts, widths = starts.flatten(), widths.flatten()
    ends = starts + widths
    valid = ends[None, :] <= lengths[:, None]
    prefix = F.pad(memory.cumsum(1), (0, 0, 1, 0))
    limit = lengths[:, None]
    starts_b = torch.minimum(starts, limit)
    ends_b = torch.minimum(ends, limit)
    left_b = torch.minimum((starts - 2).clamp_min(0), limit)
    right_b = torch.minimum(ends + 2, limit)

    def pool(a, b):
        """按前缀和取 [B W D] 均值，空区间返回零。"""
        a_sum = prefix.gather(1, a[..., None].expand(-1, -1, prefix.size(-1)))
        b_sum = prefix.gather(1, b[..., None].expand_as(a_sum))
        return (b_sum - a_sum) / (b - a).clamp_min(1)[..., None]

    # 步骤2：空上下文的标志与零向量一并传给 MLP，区分边缘和实际背景。
    inside, left, right = pool(starts_b, ends_b), pool(left_b, starts_b), pool(ends_b, right_b)
    count = (starts_b - left_b) + (right_b - ends_b)
    context = pool(left_b, right_b) * (right_b - left_b)[..., None]
    context = (context - inside * (ends_b - starts_b)[..., None]) / count.clamp_min(1)[..., None]
    context = context.masked_fill(count[..., None] == 0, 0)
    scalars = torch.stack((widths.expand_as(starts_b) / MAX_WIDTH,
                           (starts_b - left_b) / 2, (right_b - ends_b) / 2), dim=-1)
    return (torch.cat((inside, left, right, inside - context, scalars), dim=-1),
            CLIP_SECONDS * torch.stack((starts, ends), dim=-1), valid)


def make_head(hidden_dim):
    """构建区间级 MLP，输出与短 GT 最大 tIoU 对应的 logit。"""
    return nn.Sequential(nn.LayerNorm(4 * hidden_dim + 3),
                         nn.Linear(4 * hidden_dim + 3, 128), nn.GELU(), nn.Linear(128, 1))


def short_targets(windows, rows):
    """为 [W 2] 窗口构造 [B W] 最大短 GT tIoU 监督。"""
    targets = []
    for row in rows:
        truth = short_truth(row)
        if not truth:
            targets.append(windows.new_zeros(windows.size(0), dtype=torch.float32))
            continue
        gt = torch.as_tensor(truth, device=windows.device, dtype=torch.float32)
        start, end = windows[:, :1], windows[:, 1:]
        intersection = (torch.minimum(end, gt[:, 1]) - torch.maximum(start, gt[:, 0])).clamp_min(0)
        union = end - start + gt[:, 1] - gt[:, 0] - intersection
        targets.append((intersection / union).amax(1))
    return torch.stack(targets)


def score_loss(logits, targets, valid):
    """均衡重叠较高、边界偏移和无重叠窗口，保留无短 GT 查询。"""
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masks = torch.stack((valid & (targets >= .5), valid & (targets > 0) & (targets < .5),
                         valid & (targets == 0)))
    counts = masks.sum(-1)
    values = (losses * masks).sum(-1) / counts.clamp_min(1)
    return (values.sum(0) / (counts > 0).sum(0).clamp_min(1)).mean()


def rank_short(windows, scores, base):
    """稳定排序短窗口，过滤与前两项同边界的窗口。"""
    protected = {tuple(w[:2]) for w in base[:2]}
    candidates = ([float(a), float(b), float(score)] for (a, b), score in zip(windows, scores)
                  if (a, b) not in protected)
    return sorted(candidates, key=lambda w: -w[2])[:3]


def replace_third(base, candidate):
    """保留前两项与第 3 槽原分数，使官方评分器维持稳定顺序。"""
    return base[:2] + [[*candidate[:2], base[2][2]]] if candidate is not None else base


def ap_gain(base, candidate, truth):
    """使用完整 GT 与官方独占匹配计算单查询替换的 AP 差。"""
    gt = [{"video-id": 0, "t-start": a, "t-end": b} for a, b in truth]

    def ap(windows):
        """计算固定三项的平均 AP，分值范围 0 至 1。"""
        pred = [{"video-id": 0, "t-start": a, "t-end": b, "score": score}
                for a, b, score in windows]
        return compute_average_precision_detection(gt, pred).mean()

    return float(ap(replace_third(base, candidate)) - ap(base))


def gate_features(base, ranked):
    """用短窗口置信度、原排序分数和相对几何估计替换的价值。"""
    new = ranked[0]
    runner_up = ranked[1][2] if len(ranked) > 1 else new[2]
    return [new[2], new[2] - runner_up, *(w[2] for w in base),
            (new[1] - new[0]) / 10, (base[2][1] - base[2][0]) / 150,
            max(tiou(new, w) for w in base[:2]), tiou(new, base[2])]


def fit_gate(features, gains):
    """在留出的源视频上用标准化岭回归拟合带符号 AP 增益。"""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(gains, dtype=np.float64)
    if not len(y):
        raise ValueError("The gate split contains no eligible short-window proposals")
    center, scale = x.mean(0), x.std(0).clip(min=1e-6)
    z = (x - center) / scale
    penalty = .01 * len(z) * np.eye(z.shape[1])
    weights = np.linalg.solve(z.T @ z + penalty, z.T @ (y - y.mean()))
    return dict(center=center.tolist(), scale=scale.tolist(), weights=weights.tolist(),
                bias=float(y.mean()))


def predict_gain(gate, features):
    """由训练集留出子集拟合的参数给出推理时的 AP 增益估计。"""
    z = (np.asarray(features) - gate["center"]) / gate["scale"]
    return float(z @ gate["weights"] + gate["bias"])
