"""QMT-DETR model builder implemented without modifying the course baseline."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from moment_detr.matcher import build_matcher
from moment_detr.model import MomentDETR, SetCriterion
from moment_detr.position_encoding import build_position_encoding
from moment_detr.transformer import build_transformer


class QMTDETR(MomentDETR):
    """Moment-DETR with query attention and query-weighted temporal scales."""

    def __init__(self, *args, qmt_cross_attn=False, qmt_multiscale=False,
                 qmt_num_heads=8, qmt_dilations=(1, 2, 4),
                 qmt_gate_bias=-2.0, qmt_trcc=False,
                 qmt_trcc_temperature=0.02, qmt_trcc_context_scale=2.0,
                 qmt_bridge=False, qmt_iou_rank=False,
                 qmt_iou_score_weight=0.5, qmt_feature_gate=False, qmt_tcn=False,
                 qmt_dense_foreground=False, qmt_anchor_mode='none', **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.transformer.d_model
        self.qmt_dense_foreground = qmt_dense_foreground
        self.qmt_anchor_mode = qmt_anchor_mode
        if qmt_dense_foreground:
            # 独立头不改变已有模块随机初始化；不参与候选导出。
            with torch.random.fork_rng(devices=[]):
                self.qmt_foreground_head = nn.Linear(hidden_dim, 1)
        self.qmt_cross_attn = qmt_cross_attn
        self.qmt_feature_gate = qmt_feature_gate
        self.qmt_tcn = qmt_tcn
        self.qmt_multiscale = qmt_multiscale
        self.qmt_trcc = qmt_trcc
        self.qmt_bridge = qmt_bridge
        self.qmt_iou_rank = qmt_iou_rank
        self.qmt_iou_score_weight = qmt_iou_score_weight
        self.qmt_trcc_temperature = qmt_trcc_temperature
        self.qmt_trcc_context_scale = qmt_trcc_context_scale
        if (qmt_multiscale or qmt_trcc or qmt_bridge or qmt_iou_rank) and not qmt_cross_attn:
            raise ValueError("QMT refinements require query cross-attention")
        if qmt_trcc_temperature <= 0:
            raise ValueError("TRCC temperature must be positive")
        if qmt_trcc_context_scale <= 1:
            raise ValueError("TRCC context scale must be greater than one")
        if qmt_cross_attn:
            if hidden_dim % qmt_num_heads:
                raise ValueError("QMT attention heads must divide the hidden dimension")
            self.qmt_attention = nn.MultiheadAttention(
                hidden_dim, qmt_num_heads, dropout=kwargs.get("input_dropout", 0.1),
                batch_first=True,
            )
            self.qmt_gate = nn.Linear(3 * hidden_dim, hidden_dim)
            nn.init.constant_(self.qmt_gate.bias, qmt_gate_bias)
        if qmt_multiscale:
            self.qmt_temporal_convs = nn.ModuleList([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation,
                          dilation=dilation, groups=hidden_dim)
                for dilation in qmt_dilations
            ])
            self.qmt_scale_weights = nn.Linear(hidden_dim, len(qmt_dilations))
            self.qmt_temporal_out = nn.Linear(hidden_dim, hidden_dim)
            # Identity initialization prevents a large perturbation at step zero.
            nn.init.zeros_(self.qmt_temporal_out.weight)
            nn.init.zeros_(self.qmt_temporal_out.bias)
        if qmt_trcc:
            # Zero initialization makes TRCC exactly identical to Cross at step zero.
            self.qmt_trcc_alpha = nn.Parameter(torch.zeros(()))
            self.qmt_trcc_beta = nn.Parameter(torch.zeros(()))
        if qmt_iou_rank:
            self.qmt_iou_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
            )
            # 零初始化保证首次评估时所有 query 的排序与仅 Cross 完全一致。
            nn.init.zeros_(self.qmt_iou_head[-1].weight)
            nn.init.zeros_(self.qmt_iou_head[-1].bias)

        if qmt_feature_gate:
            if kwargs["vid_dim"] != 2818 or kwargs["txt_dim"] != 512:
                raise ValueError("Feature gate requires SlowFast2304 + CLIP512 + time2 and text512")
            # 隔离初始化随机数，使配对对照的后续 dropout 随机流保持一致。
            with torch.random.fork_rng(devices=[]):
                self.qmt_feature_gate_mlp = nn.Sequential(
                    nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2),
                )
            nn.init.zeros_(self.qmt_feature_gate_mlp[-1].weight)
            nn.init.zeros_(self.qmt_feature_gate_mlp[-1].bias)

        if qmt_tcn:
            # 独立随机流不改变对照模型的参数初始化与后续 dropout。
            with torch.random.fork_rng(devices=[]):
                self.qmt_tcn_layers = nn.ModuleList([
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, dilation=1),
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
                ])
            nn.init.zeros_(self.qmt_tcn_layers[-1].weight)
            nn.init.zeros_(self.qmt_tcn_layers[-1].bias)

        if qmt_anchor_mode != 'none':
            from student_code.qmt.anchors import TemporalAnchors
            if self.num_queries != 10 or self.span_loss_type != 'l1':
                raise ValueError('Temporal anchors require 10 queries and l1 spans')
            with torch.random.fork_rng(devices=[]):
                self.qmt_anchors = TemporalAnchors(hidden_dim, qmt_anchor_mode)
            # 锚点分支必须从零修正开始；关闭时不改变既有span头。
            nn.init.zeros_(self.span_embed.layers[-1].weight)
            nn.init.zeros_(self.span_embed.layers[-1].bias)

    def _temporal_residual(self, video, valid):
        """串联非因果卷积，逐层排除 padding，残差分支感受野为7个clip。

        Args:
            video (Float[Tensor, "B V D"]): 投影后的视频特征。
            valid (Bool[Tensor, "B V"]): True代表有效clip。

        Returns:
            Float[Tensor, "B V D"]: 恒等初始化的残差特征。
        """
        # 步骤1：卷积前和每层之后清零，避免偏置经padding传播回有效位置。
        mask = valid[:, None, :].bool()
        branch = video.transpose(1, 2).masked_fill(~mask, 0)
        branch = F.relu(self.qmt_tcn_layers[0](branch)).masked_fill(~mask, 0)
        branch = self.qmt_tcn_layers[1](branch).masked_fill(~mask, 0)
        # 步骤2：保持残差输入不变；下游Transformer仍使用原视频掩码。
        return video + branch.transpose(1, 2)

    def feature_gates(self, src_txt, src_txt_mask):
        """从有效原始文本特征计算两个模态缩放门，零初始化时恒为1。

        Args:
            src_txt (Float[Tensor, "B T 512"]): 原始文本特征。
            src_txt_mask (Bool[Tensor, "B T"]): True代表有效token。

        Returns:
            Float[Tensor, "B 2"]: SlowFast和CLIP门值，范围为(0, 2)。
        """
        pooled = self._masked_text_mean(src_txt, src_txt_mask)
        return 1 + torch.tanh(self.qmt_feature_gate_mlp(pooled))

    def _gate_features(self, src_vid, gates):
        """在投影前调制两路视觉特征，保持时间通道和输入张量不变。

        Args:
            src_vid (Float[Tensor, "B V 2818"]): SF、CLIP、时间顺序的特征。
            gates (Float[Tensor, "B 2"]): 查询条件缩放门。

        Returns:
            Float[Tensor, "B V 2818"]: 保持维度的调制特征。
        """
        return torch.cat((src_vid[..., :2304] * gates[:, None, 0:1],
                          src_vid[..., 2304:2816] * gates[:, None, 1:2],
                          src_vid[..., 2816:]), dim=-1)

    @staticmethod
    def _masked_text_mean(src_txt, src_txt_mask):
        """Pool valid query tokens.

        Args:
            src_txt: 投影后的文本特征，形状为 ``[B, L_t, D]``。
            src_txt_mask: 文本有效位置掩码，形状为 ``[B, L_t]``。

        Returns:
            文本全局表示，形状为 ``[B, D]``。
        """
        weights = src_txt_mask.to(src_txt.dtype).unsqueeze(-1)
        return (src_txt * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def _refine_video(self, src_vid, src_txt, src_txt_mask, src_vid_mask):
        """Inject query context and aggregate multiple temporal scales.

        Args:
            src_vid: 投影后的视频特征，形状为 ``[B, L_v, D]``。
            src_txt: 投影后的文本特征，形状为 ``[B, L_t, D]``。
            src_txt_mask: 文本有效位置掩码，形状为 ``[B, L_t]``。
            src_vid_mask: 视频有效位置掩码，形状为 ``[B, L_v]``。

        Returns:
            查询条件化的视频特征，形状为 ``[B, L_v, D]``。
        """
        text_mean = self._masked_text_mean(src_txt, src_txt_mask)
        context, _ = self.qmt_attention(
            src_vid, src_txt, src_txt,
            key_padding_mask=~src_txt_mask.bool(), need_weights=False,
        )
        text_tokens = text_mean[:, None, :].expand_as(src_vid)
        gate = torch.sigmoid(self.qmt_gate(torch.cat([src_vid, context, text_tokens], -1)))
        refined = src_vid + gate * context
        if self.qmt_multiscale:
            conv_input = refined.transpose(1, 2)
            branches = torch.stack(
                [conv(conv_input).transpose(1, 2) for conv in self.qmt_temporal_convs], dim=2,
            )  # [B, L_v, K, D]
            weights = torch.softmax(self.qmt_scale_weights(text_mean), dim=-1)
            mixed = (branches * weights[:, None, :, None]).sum(2)
            refined = refined + self.qmt_temporal_out(mixed)
        return refined * src_vid_mask.to(refined.dtype).unsqueeze(-1)

    def _calibrate_logits(self, logits, spans, saliency_scores, video_mask):
        """Calibrate foreground confidence from span-interior and ring saliency.

        Args:
            logits: Proposal class logits, shape ``[B, Q, 2]``.
            spans: Normalized center-width proposals, shape ``[B, Q, 2]``.
            saliency_scores: Per-clip saliency logits, shape ``[B, L_v]``.
            video_mask: Valid-clip mask, shape ``[B, L_v]``.

        Returns:
            Calibrated logits with the same shape. Span coordinates are detached
            while forming soft masks, so this branch does not alter span regression.
        """
        valid = video_mask.bool()
        valid_float = valid.to(saliency_scores.dtype)
        valid_lengths = valid_float.sum(1, keepdim=True).clamp_min(1.0)
        positions = (valid_float.cumsum(1) - 0.5) / valid_lengths
        positions = positions[:, None, :]
        centers, widths = spans.detach().unbind(-1)
        half_width = widths.clamp_min(0.0) / 2
        temperature = self.qmt_trcc_temperature

        def soft_interval(left, right):
            return (
                torch.sigmoid((positions - left[..., None]) / temperature)
                * torch.sigmoid((right[..., None] - positions) / temperature)
                * valid_float[:, None, :]
            )

        inside = soft_interval(centers - half_width, centers + half_width)
        outer_half_width = half_width * self.qmt_trcc_context_scale
        outer = soft_interval(centers - outer_half_width, centers + outer_half_width)
        context = (outer - inside).clamp_min(0.0)

        scores = saliency_scores[:, None, :]
        inside_mean = (scores * inside).sum(-1) / inside.sum(-1).clamp_min(1e-6)
        context_mass = context.sum(-1)
        context_mean = (scores * context).sum(-1) / context_mass.clamp_min(1e-6)
        context_mean = torch.where(context_mass > 1e-6, context_mean, inside_mean)
        correction = (
            self.qmt_trcc_alpha * inside_mean
            + self.qmt_trcc_beta * (inside_mean - context_mean)
        )
        calibrated = logits.clone()
        calibrated[..., 0] = calibrated[..., 0] + correction
        return calibrated

    def _apply_iou_ranking(self, logits, quality_logits):
        """Use centered IoU quality to refine proposal order at evaluation.

        Args:
            logits: Proposal class logits, shape ``[B, Q, 2]``.
            quality_logits: IoU quality logits, shape ``[B, Q]``.

        Returns:
            仅调整前景通道的 logits，形状为 ``[B, Q, 2]``。中心化使该分支只改变
            query 间相对顺序，不改变单个样本的整体前景偏置。
        """
        correction = quality_logits - quality_logits.mean(dim=1, keepdim=True)
        ranked = logits.clone()
        ranked[..., 0] = ranked[..., 0] + self.qmt_iou_score_weight * correction
        return ranked

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask):
        """Run QMT-DETR with the same input/output contract as Moment-DETR."""
        gates = self.feature_gates(src_txt, src_txt_mask) if self.qmt_feature_gate else None
        if gates is not None:
            src_vid = self._gate_features(src_vid, gates)
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        if self.qmt_cross_attn:
            src_vid = self._refine_video(src_vid, src_txt, src_txt_mask, src_vid_mask)
        if self.qmt_tcn:
            src_vid = self._temporal_residual(src_vid, src_vid_mask)
        src = torch.cat([src_vid, src_txt], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()
        pos_vid = self.position_embed(src_vid, src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        if self.qmt_anchor_mode != 'none':
            from student_code.qmt.anchors import decode_anchors
            hs, memory, outputs_coord = decode_anchors(
                self, src, ~mask, torch.cat([pos_vid, pos_txt], dim=1))
        else:
            hs, memory = self.transformer(
                src, ~mask, self.query_embed.weight, torch.cat([pos_vid, pos_txt], dim=1),
            )
            outputs_coord = self.span_embed(hs)
            if self.span_loss_type == 'l1':
                outputs_coord = outputs_coord.sigmoid()
        outputs_class = self.class_embed(hs)
        txt_mem = memory[:, src_vid.shape[1]:]
        vid_mem = memory[:, :src_vid.shape[1]]
        saliency_scores = self.saliency_proj(vid_mem).squeeze(-1)
        final_logits = outputs_class[-1]
        if self.qmt_trcc:
            final_logits = self._calibrate_logits(
                final_logits, outputs_coord[-1], saliency_scores, src_vid_mask,
            )
        quality_logits = None
        if self.qmt_iou_rank:
            # 质量头只学习读取 decoder 表示，不反向改变已验证的 Cross 骨架。
            quality_logits = self.qmt_iou_head(hs[-1].detach()).squeeze(-1)
            if not self.training:
                final_logits = self._apply_iou_ranking(final_logits, quality_logits)
        out = {"pred_logits": final_logits, "pred_spans": outputs_coord[-1]}
        if gates is not None:
            out["feature_gates"] = gates
        if quality_logits is not None:
            out["qmt_iou_logits"] = quality_logits
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(proj_queries=proj_queries[-1], proj_txt_mem=proj_txt_mem,
                       proj_vid_mem=proj_vid_mem)
        out["saliency_scores"] = saliency_scores
        if self.qmt_dense_foreground:
            out["foreground_logits"] = self.qmt_foreground_head(vid_mem).squeeze(-1)
        out["video_mask"] = src_vid_mask.bool()
        if self.qmt_bridge:
            # 最终层 query 与 encoder clip 位于同一隐藏空间；归一化后只监督方向相似度。
            out["bridge_queries"] = F.normalize(hs[-1], p=2, dim=-1)
            out["bridge_vid_mem"] = F.normalize(vid_mem, p=2, dim=-1)
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": logits, "pred_spans": spans}
                for logits, spans in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
            if self.contrastive_align_loss:
                for index, projected in enumerate(proj_queries[:-1]):
                    out["aux_outputs"][index].update(
                        proj_queries=projected, proj_txt_mem=proj_txt_mem,
                    )
        return out


class QMTSetCriterion(SetCriterion):
    """Baseline criterion plus optional top-layer saliency calibration losses."""

    def __init__(self, *args, qmt_hard_negative=False,
                 qmt_hard_negative_margin=0.2,
                 qmt_negative_query=False,
                 qmt_negative_query_margin=0.2,
                 qmt_bridge=False, qmt_bridge_temperature=0.07,
                 qmt_iou_rank=False, qmt_iou_rank_margin=0.1,
                 qmt_iou_pair_coef=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.qmt_hard_negative = qmt_hard_negative
        self.qmt_hard_negative_margin = qmt_hard_negative_margin
        self.qmt_negative_query = qmt_negative_query
        self.qmt_negative_query_margin = qmt_negative_query_margin
        self.qmt_bridge = qmt_bridge
        self.qmt_bridge_temperature = qmt_bridge_temperature
        self.qmt_iou_rank = qmt_iou_rank
        self.qmt_iou_rank_margin = qmt_iou_rank_margin
        self.qmt_iou_pair_coef = qmt_iou_pair_coef
        if qmt_bridge_temperature <= 0:
            raise ValueError("Bridge temperature must be positive")
        if qmt_iou_rank_margin < 0:
            raise ValueError("IoU ranking margin must be non-negative")
        if qmt_iou_pair_coef < 0:
            raise ValueError("IoU pair coefficient must be non-negative")

    def loss_qmt_hard_negative(self, outputs, targets):
        """Contrast GT-interior saliency with the strongest exterior clip."""
        scores, video_mask = outputs["saliency_scores"], outputs["video_mask"]
        losses = []
        for batch_index, span_target in enumerate(targets["span_labels"]):
            valid_length = int(video_mask[batch_index].sum())
            if valid_length < 2:
                continue
            centers = (torch.arange(valid_length, device=scores.device,
                                    dtype=scores.dtype) + 0.5) / valid_length
            spans = span_target["spans"].to(scores.dtype)
            starts, ends = spans[:, 0] - spans[:, 1] / 2, spans[:, 0] + spans[:, 1] / 2
            inside = ((centers[:, None] >= starts) & (centers[:, None] <= ends)).any(1)
            if not inside.any() or inside.all():
                continue
            valid_scores = scores[batch_index, :valid_length]
            losses.append(F.relu(
                self.qmt_hard_negative_margin - valid_scores[inside].mean()
                + valid_scores[~inside].max()
            ))
        return torch.stack(losses).mean() if losses else scores.sum() * 0.0

    def loss_qmt_negative_query(self, outputs, negative_outputs, targets):
        """Separate matched salient clips from every valid mismatched-query clip.

        Args:
            outputs: 正确视频—查询配对的模型输出。
            negative_outputs: 视频不变、文本在 batch 维循环移位后的模型输出。
            targets: 基线 ``prepare_batch_inputs`` 生成的训练目标。

        Returns:
            标量 margin loss。batch size 小于 2 时由训练循环跳过该项。
        """
        positive_scores = outputs["saliency_scores"]
        negative_scores = negative_outputs["saliency_scores"]
        video_mask = outputs["video_mask"].bool()
        if "saliency_pos_labels" not in targets:
            return positive_scores.sum() * 0.0
        positive_indices = targets["saliency_pos_labels"].long()
        batch_indices = torch.arange(
            positive_scores.shape[0], device=positive_scores.device,
        )[:, None]
        positive_anchor = positive_scores[batch_indices, positive_indices].mean(1)
        strongest_mismatch = negative_scores.masked_fill(
            ~video_mask, torch.finfo(negative_scores.dtype).min,
        ).max(1).values
        valid_rows = video_mask.any(1)
        if not valid_rows.any():
            return positive_scores.sum() * 0.0
        return F.relu(
            self.qmt_negative_query_margin
            - positive_anchor[valid_rows]
            + strongest_mismatch[valid_rows]
        ).mean()

    def loss_qmt_bridge(self, outputs, targets, indices):
        """Align each matched decoder query with clips inside its GT span.

        Args:
            outputs: 最终层输出，含归一化 query、video memory 与有效 clip 掩码。
            targets: 基线 prepare_batch_inputs 生成的训练目标。
            indices: Hungarian matcher 返回的逐样本 (query_idx, target_idx)。

        Returns:
            区间内、外二项等权的标量 logistic loss。padding clip 不参与监督。
        """
        queries = outputs["bridge_queries"]
        frames = outputs["bridge_vid_mem"]
        video_mask = outputs["video_mask"].bool()
        pair_losses = []
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            valid_length = int(video_mask[batch_index].sum())
            if valid_length == 0:
                continue
            positions = (
                torch.arange(valid_length, device=queries.device, dtype=queries.dtype) + 0.5
            ) / valid_length
            target_spans = targets["span_labels"][batch_index]["spans"][target_indices]
            target_spans = target_spans.to(device=queries.device, dtype=queries.dtype)
            for query_index, target_span in zip(query_indices, target_spans):
                center, width = target_span.unbind(0)
                inside = (positions >= center - width / 2) & (positions <= center + width / 2)
                # 极窄标注仍对齐到最近 clip，避免该正匹配没有局部监督。
                if not inside.any():
                    inside[(positions - center).abs().argmin()] = True
                logits = torch.matmul(
                    frames[batch_index, :valid_length], queries[batch_index, query_index],
                ) / self.qmt_bridge_temperature
                inside_loss = F.softplus(-logits[inside]).mean()
                outside = ~inside
                outside_loss = (
                    F.softplus(logits[outside]).mean() if outside.any()
                    else logits.sum() * 0.0
                )
                pair_losses.append(inside_loss + outside_loss)
        return torch.stack(pair_losses).mean() if pair_losses else queries.sum() * 0.0

    def loss_qmt_iou_rank(self, outputs, targets):
        """Regress proposal IoU and rank queries by detached boundary quality.

        Args:
            outputs: 最终层输出，含 ``[B, Q]`` 质量 logits 与 ``[B, Q, 2]`` span。
            targets: 基线 ``prepare_batch_inputs`` 生成的训练目标。

        Returns:
            soft-IoU BCE 与有序 query 对 RankNet 损失之和。IoU target 由停止梯度的
            预测边界计算，因此该项只训练质量头，不反向改变 Cross 定位骨架。
        """
        quality_logits = outputs["qmt_iou_logits"]
        predicted_spans = outputs["pred_spans"].detach()
        target_ious = []
        for batch_index, span_target in enumerate(targets["span_labels"]):
            predicted = predicted_spans[batch_index]
            ground_truth = span_target["spans"].to(
                device=predicted.device, dtype=predicted.dtype,
            )
            predicted_start = predicted[:, 0] - predicted[:, 1] / 2
            predicted_end = predicted[:, 0] + predicted[:, 1] / 2
            target_start = ground_truth[:, 0] - ground_truth[:, 1] / 2
            target_end = ground_truth[:, 0] + ground_truth[:, 1] / 2
            intersection = (
                torch.minimum(predicted_end[:, None], target_end[None, :])
                - torch.maximum(predicted_start[:, None], target_start[None, :])
            ).clamp_min(0.0)
            union = (
                (predicted_end - predicted_start).clamp_min(0.0)[:, None]
                + (target_end - target_start).clamp_min(0.0)[None, :]
                - intersection
            )
            target_ious.append((intersection / union.clamp_min(1e-6)).max(dim=1).values)
        target_ious = torch.stack(target_ious)
        regression = F.binary_cross_entropy_with_logits(quality_logits, target_ious)

        differences = target_ious[:, :, None] - target_ious[:, None, :]
        ordered = differences > self.qmt_iou_rank_margin
        if ordered.any():
            logit_differences = quality_logits[:, :, None] - quality_logits[:, None, :]
            pairwise = (
                F.softplus(-logit_differences[ordered]) * differences[ordered]
            ).sum() / differences[ordered].sum().clamp_min(1e-6)
        else:
            pairwise = quality_logits.sum() * 0.0
        return regression + self.qmt_iou_pair_coef * pairwise

    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        if self.qmt_hard_negative:
            losses["loss_qmt_hard_negative"] = self.loss_qmt_hard_negative(outputs, targets)
        if self.qmt_bridge:
            # 基线 criterion 不暴露匹配索引；仅为最终层重复一次轻量 Hungarian 匹配。
            outputs_without_aux = {
                key: value for key, value in outputs.items() if key != "aux_outputs"
            }
            indices = self.matcher(outputs_without_aux, targets)
            losses["loss_qmt_bridge"] = self.loss_qmt_bridge(outputs, targets, indices)
        if self.qmt_iou_rank:
            losses["loss_qmt_iou_rank"] = self.loss_qmt_iou_rank(outputs, targets)
        return losses


def build_qmt_model(args):
    """Build the student model and criterion from a checkpoint-safe config."""
    device = torch.device(args.device)
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)
    model = QMTDETR(
        transformer, position_embedding, txt_position_embedding,
        txt_dim=args.t_feat_dim, vid_dim=args.v_feat_dim, num_queries=args.num_queries,
        input_dropout=args.input_dropout, aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim, max_v_l=args.max_v_l,
        span_loss_type=args.span_loss_type, use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        qmt_cross_attn=getattr(args, "qmt_cross_attn", False),
        qmt_feature_gate=getattr(args, "qmt_feature_gate", False),
        qmt_tcn=getattr(args, "qmt_tcn", False),
        qmt_dense_foreground=getattr(args, "qmt_dense_foreground", False),
        qmt_anchor_mode=getattr(args, "qmt_anchor_mode", 'none'),
        qmt_multiscale=getattr(args, "qmt_multiscale", False),
        qmt_num_heads=getattr(args, "qmt_num_heads", 8),
        qmt_dilations=getattr(args, "qmt_dilations", (1, 2, 4)),
        qmt_gate_bias=getattr(args, "qmt_gate_bias", -2.0),
        qmt_trcc=getattr(args, "qmt_trcc", False),
        qmt_trcc_temperature=getattr(args, "qmt_trcc_temperature", 0.02),
        qmt_trcc_context_scale=getattr(args, "qmt_trcc_context_scale", 2.0),
        qmt_bridge=getattr(args, "qmt_bridge", False),
        qmt_iou_rank=getattr(args, "qmt_iou_rank", False),
        qmt_iou_score_weight=getattr(args, "qmt_iou_score_weight", 0.5),
    )
    weight_dict = {
        "loss_span": args.span_loss_coef, "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef, "loss_saliency": args.lw_saliency,
    }
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if getattr(args, "qmt_hard_negative", False):
        weight_dict["loss_qmt_hard_negative"] = getattr(args, "qmt_hard_negative_coef", 0.1)
    if getattr(args, "qmt_negative_query", False):
        weight_dict["loss_qmt_negative_query"] = getattr(args, "qmt_negative_query_coef", 0.1)
    if getattr(args, "qmt_bridge", False):
        weight_dict["loss_qmt_bridge"] = getattr(args, "qmt_bridge_coef", 0.1)
    if getattr(args, "qmt_iou_rank", False):
        weight_dict["loss_qmt_iou_rank"] = getattr(args, "qmt_iou_rank_coef", 0.2)
    if args.aux_loss:
        for index in range(args.dec_layers - 1):
            weight_dict.update({
                key + f"_{index}": value for key, value in list(weight_dict.items())
                if key not in {"loss_saliency", "loss_qmt_hard_negative",
                               "loss_qmt_negative_query", "loss_qmt_bridge",
                               "loss_qmt_iou_rank"}
                and not key.rsplit("_", 1)[-1].isdigit()
            })
    losses = ["spans", "labels", "saliency"]
    if args.contrastive_align_loss:
        losses.append("contrastive_align")
    criterion = QMTSetCriterion(
        matcher=build_matcher(args), weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,
        qmt_hard_negative=getattr(args, "qmt_hard_negative", False),
        qmt_hard_negative_margin=getattr(args, "qmt_hard_negative_margin", 0.2),
        qmt_negative_query=getattr(args, "qmt_negative_query", False),
        qmt_negative_query_margin=getattr(args, "qmt_negative_query_margin", 0.2),
        qmt_bridge=getattr(args, "qmt_bridge", False),
        qmt_bridge_temperature=getattr(args, "qmt_bridge_temperature", 0.07),
        qmt_iou_rank=getattr(args, "qmt_iou_rank", False),
        qmt_iou_rank_margin=getattr(args, "qmt_iou_rank_margin", 0.1),
        qmt_iou_pair_coef=getattr(args, "qmt_iou_pair_coef", 0.5),
    )
    criterion.to(device)
    return model, criterion
