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
                 **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.transformer.d_model
        self.qmt_cross_attn = qmt_cross_attn
        self.qmt_multiscale = qmt_multiscale
        self.qmt_trcc = qmt_trcc
        self.qmt_trcc_temperature = qmt_trcc_temperature
        self.qmt_trcc_context_scale = qmt_trcc_context_scale
        if (qmt_multiscale or qmt_trcc) and not qmt_cross_attn:
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

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask):
        """Run QMT-DETR with the same input/output contract as Moment-DETR."""
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        if self.qmt_cross_attn:
            src_vid = self._refine_video(src_vid, src_txt, src_txt_mask, src_vid_mask)
        src = torch.cat([src_vid, src_txt], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()
        pos_vid = self.position_embed(src_vid, src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        hs, memory = self.transformer(
            src, ~mask, self.query_embed.weight, torch.cat([pos_vid, pos_txt], dim=1),
        )
        outputs_class = self.class_embed(hs)
        outputs_coord = self.span_embed(hs)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        txt_mem = memory[:, src_vid.shape[1]:]
        vid_mem = memory[:, :src_vid.shape[1]]
        saliency_scores = self.saliency_proj(vid_mem).squeeze(-1)
        final_logits = outputs_class[-1]
        if self.qmt_trcc:
            final_logits = self._calibrate_logits(
                final_logits, outputs_coord[-1], saliency_scores, src_vid_mask,
            )
        out = {"pred_logits": final_logits, "pred_spans": outputs_coord[-1]}
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(proj_queries=proj_queries[-1], proj_txt_mem=proj_txt_mem,
                       proj_vid_mem=proj_vid_mem)
        out["saliency_scores"] = saliency_scores
        out["video_mask"] = src_vid_mask.bool()
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
                 qmt_negative_query_margin=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.qmt_hard_negative = qmt_hard_negative
        self.qmt_hard_negative_margin = qmt_hard_negative_margin
        self.qmt_negative_query = qmt_negative_query
        self.qmt_negative_query_margin = qmt_negative_query_margin

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

    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        if self.qmt_hard_negative:
            losses["loss_qmt_hard_negative"] = self.loss_qmt_hard_negative(outputs, targets)
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
        qmt_multiscale=getattr(args, "qmt_multiscale", False),
        qmt_num_heads=getattr(args, "qmt_num_heads", 8),
        qmt_dilations=getattr(args, "qmt_dilations", (1, 2, 4)),
        qmt_gate_bias=getattr(args, "qmt_gate_bias", -2.0),
        qmt_trcc=getattr(args, "qmt_trcc", False),
        qmt_trcc_temperature=getattr(args, "qmt_trcc_temperature", 0.02),
        qmt_trcc_context_scale=getattr(args, "qmt_trcc_context_scale", 2.0),
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
    if args.aux_loss:
        for index in range(args.dec_layers - 1):
            weight_dict.update({
                key + f"_{index}": value for key, value in list(weight_dict.items())
                if key not in {"loss_saliency", "loss_qmt_hard_negative",
                               "loss_qmt_negative_query"}
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
    )
    criterion.to(device)
    return model, criterion
