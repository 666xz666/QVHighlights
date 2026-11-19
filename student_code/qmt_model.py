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
                 qmt_gate_bias=-2.0, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.transformer.d_model
        self.qmt_cross_attn = qmt_cross_attn
        self.qmt_multiscale = qmt_multiscale
        if qmt_multiscale and not qmt_cross_attn:
            raise ValueError("QMT multi-scale refinement requires query cross-attention")
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
        out = {"pred_logits": outputs_class[-1], "pred_spans": outputs_coord[-1]}
        txt_mem = memory[:, src_vid.shape[1]:]
        vid_mem = memory[:, :src_vid.shape[1]]
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(proj_queries=proj_queries[-1], proj_txt_mem=proj_txt_mem,
                       proj_vid_mem=proj_vid_mem)
        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)
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
    """Baseline criterion plus an optional top-layer hard-negative margin."""

    def __init__(self, *args, qmt_hard_negative=False,
                 qmt_hard_negative_margin=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.qmt_hard_negative = qmt_hard_negative
        self.qmt_hard_negative_margin = qmt_hard_negative_margin

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
    )
    weight_dict = {
        "loss_span": args.span_loss_coef, "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef, "loss_saliency": args.lw_saliency,
    }
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if getattr(args, "qmt_hard_negative", False):
        weight_dict["loss_qmt_hard_negative"] = getattr(args, "qmt_hard_negative_coef", 0.1)
    if args.aux_loss:
        for index in range(args.dec_layers - 1):
            weight_dict.update({
                key + f"_{index}": value for key, value in list(weight_dict.items())
                if key not in {"loss_saliency", "loss_qmt_hard_negative"}
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
    )
    criterion.to(device)
    return model, criterion
