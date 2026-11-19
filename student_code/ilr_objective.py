"""Differentiable IoU-aware listwise reranking objective."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def first_answer_score(logits, labels, yes_token_id, no_token_id):
    """Return ``logit(Yes) - logit(No)`` before the first answer token.

    Args:
        logits: 语言模型输出，形状为 ``[1, L, |V|]``。
        labels: 仅答案与 EOS 非屏蔽的标签，形状为 ``[1, L]``。
        yes_token_id: ``Yes`` 的单 token 编号。
        no_token_id: ``No`` 的单 token 编号。

    Returns:
        保持梯度的标量排序分数。
    """
    supervised = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    if supervised.numel() == 0 or int(supervised[0]) == 0:
        raise ValueError("A supervised answer token must follow at least one prefix token")
    decision_logits = logits[0, supervised[0] - 1]
    return decision_logits[yes_token_id].float() - decision_logits[no_token_id].float()


def ilr_auxiliary_loss(scores, target_tiou, tau_score=1.0, tau_target=0.1,
                       tau_pair=1.0):
    """Compute two-item soft-listwise and IoU-weighted RankNet losses.

    The same function also accepts a longer list, matching fixed Top-3 inference
    semantics without requiring new training boundaries.
    """
    if scores.ndim != 1 or target_tiou.ndim != 1 or scores.shape != target_tiou.shape:
        raise ValueError("scores and target_tiou must be equal-length vectors")
    if scores.numel() < 2:
        raise ValueError("ILR requires at least two candidates")
    if min(tau_score, tau_target, tau_pair) <= 0:
        raise ValueError("ILR temperatures must be positive")
    predicted_log_probs = F.log_softmax(scores / tau_score, dim=0)
    target_probs = F.softmax(target_tiou / tau_target, dim=0)
    listwise = -(target_probs * predicted_log_probs).sum()
    pair_terms = []
    for left in range(scores.numel()):
        for right in range(left + 1, scores.numel()):
            gap = target_tiou[left] - target_tiou[right]
            if float(gap) == 0.0:
                continue
            direction = gap.sign()
            pair_terms.append(gap.abs() * F.softplus(
                -direction * (scores[left] - scores[right]) / tau_pair,
            ))
    pairwise = torch.stack(pair_terms).mean() if pair_terms else scores.sum() * 0.0
    return listwise, pairwise


def combined_ilr_loss(ce_losses, scores, target_tiou, listwise_weight=0.5,
                      pairwise_weight=0.2, tau_score=1.0,
                      tau_target=0.1, tau_pair=1.0):
    """Combine mean answer-token CE with listwise and pairwise supervision."""
    ce = torch.stack(ce_losses).mean()
    listwise, pairwise = ilr_auxiliary_loss(
        scores, target_tiou, tau_score=tau_score,
        tau_target=tau_target, tau_pair=tau_pair,
    )
    total = ce + listwise_weight * listwise + pairwise_weight * pairwise
    return total, {"ce": ce, "listwise": listwise, "pairwise": pairwise}
