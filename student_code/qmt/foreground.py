"""完整训练GT上的逐clip前景与短片段空隙监督，不改变推理排序。"""
import torch
import torch.nn.functional as F


def foreground_losses(logits, valid, rows, margin=0.2):
    """计算query/类别平衡BCE与邻接短片段—空隙logistic排序损失。

    Args:
        logits (Float[Tensor, "B L"]): 独立前景头的未归一化输出。
        valid (Bool[Tensor, "B L"]): True表示有效clip，padding不监督。
        rows (list[dict]): 未截断的train标注，区间以秒计。
        margin (float): 正片段与空隙均值logit的排序间隔。

    Returns:
        tuple[Tensor, Tensor]: 标量BCE、局部排序损失；无适用项返回可导零。
    """
    dense, local = [], []
    for score, mask, row in zip(logits, valid.bool(), rows):
        # 步骤1：沿用课程2秒clip坐标，不用duration缩放或截断的匹配标签。
        centers = (torch.arange(score.numel(), device=score.device) + 0.5) * 2
        windows = sorted(row["relevant_windows"])
        components = []
        for start, end in windows:
            if components and start <= components[-1][1]:
                components[-1][1] = max(end, components[-1][1])
            else:
                components.append([start, end])
        inside = torch.zeros_like(mask)
        for start, end in components:
            inside |= (centers >= start) & (centers < end)
        positive, negative = mask & inside, mask & ~inside
        # 步骤2：类别内均值再类别间等权，避免长背景主导训练。
        classes = []
        if positive.any():
            classes.append(F.softplus(-score[positive]).mean())
        if negative.any():
            classes.append(F.softplus(score[negative]).mean())
        if classes:
            dense.append(torch.stack(classes).mean())
        gaps = []
        for left, right in zip(components, components[1:]):
            gap = mask & ~inside & (centers >= left[1]) & (centers < right[0])
            if not gap.any():
                continue
            flanks = []
            for start, end in (left, right):
                flank = mask & (centers >= start) & (centers < end)
                if end - start <= 10 and flank.any():
                    flanks.append(F.softplus(margin + score[gap].mean() - score[flank].mean()))
            if flanks:
                gaps.append(torch.stack(flanks).mean())
        # 步骤3：左右/空隙/query逐级等权，无短邻接空隙的query不参与此项。
        if gaps:
            local.append(torch.stack(gaps).mean())
    zero = logits.sum() * 0
    return (torch.stack(dense).mean() if dense else zero,
            torch.stack(local).mean() if local else zero)
