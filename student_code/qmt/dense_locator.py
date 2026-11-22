"""用 CrossTCN 编码器和共享密集头直接预测查询相关的时间窗口。"""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from student_code.qmt.model import QMTDETR, build_qmt_model


LEVEL_STRIDES = {"single": (2,), "multi": (2, 4, 8, 16)}
LENGTH_RANGES = ((0, 8), (4, 16), (8, 32), (16, float("inf")))


class DenseLocator(nn.Module):
    """以秒为单位预测每个有效位置的中心偏移、宽度、相关性与 IoU。"""

    _masked_text_mean = staticmethod(QMTDETR._masked_text_mean)
    _refine_video = QMTDETR._refine_video
    _temporal_residual = QMTDETR._temporal_residual

    def __init__(self, config, scale):
        """保留 CrossTCN 的编码部分，两种尺度配置使用相同的共享头。

        Args:
            config (dict): 课程模型维度和视频/文本特征配置。
            scale (str): single 仅使用 2 秒层，multi 使用全部四层。
        """
        super().__init__()
        self.strides = LEVEL_STRIDES[scale]
        self.use_txt_pos = config["use_txt_pos"]
        self.qmt_multiscale = False
        base, _ = build_qmt_model(SimpleNamespace(
            **{**config, "qmt_cross_attn": True, "qmt_tcn": True,
               "contrastive_align_loss": False, "aux_loss": False},
        ))
        for name in ("input_vid_proj", "input_txt_proj", "position_embed", "txt_position_embed",
                     "qmt_attention", "qmt_gate", "qmt_tcn_layers"):
            setattr(self, name, getattr(base, name))
        self.encoder = base.transformer.encoder
        dim = base.transformer.d_model
        self.context = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 4)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1] = 0.5  # 初始宽度约为两个本层步长。
            self.head.bias[2] = -2.0  # 密集位置中背景占比高。

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, duration):
        """沿有效视频轴构建非重叠池化金字塔并解码到秒。

        Args:
            src_txt (Float[Tensor, "B T D_t"]): 文本 token 特征。
            src_txt_mask (Bool[Tensor, "B T"]): True 为有效文本。
            src_vid (Float[Tensor, "B V D_v"]): 视频 clip 特征。
            src_vid_mask (Bool[Tensor, "B V"]): True 为有效 clip，右侧 padding。
            duration (Float[Tensor, "B"]): 视频实际时长，限制输出边界。

        Returns:
            dict: spans 和 raw_spans [B P 2] 分别为裁剪后与裁剪前的端点；
            valid、points、relevance、quality 为 [B P]；
            strides、levels 为 [P]。P 拼接各层位置。
        """
        # 步骤 1：编码器与 CrossTCN 相同，粗层仅从有效视频 token 聚合。
        valid = src_vid_mask.bool()
        text = self.input_txt_proj(src_txt)
        video = self.input_vid_proj(src_vid)
        video = self._refine_video(video, text, src_txt_mask, valid)
        video = self._temporal_residual(video, valid)
        src = torch.cat((video, text), dim=1)
        padding = ~torch.cat((valid, src_txt_mask.bool()), dim=1)
        text_pos = self.txt_position_embed(text) if self.use_txt_pos else torch.zeros_like(text)
        pos = torch.cat((self.position_embed(video, src_vid_mask), text_pos), dim=1)
        memory = self.encoder(src.transpose(0, 1), src_key_padding_mask=padding,
                              pos=pos.transpose(0, 1)).transpose(0, 1)[:, :video.shape[1]]
        # 步骤 2：尾部不满一组时按有效数量求均值，时间中心也使用实际覆盖范围。
        valid = valid & (2 * torch.arange(video.shape[1], device=video.device) < duration[:, None])
        memory = memory.masked_fill(~valid[..., None], 0)
        lengths = valid.sum(dim=1)
        level_outputs = []
        for level, stride in enumerate(self.strides):
            ratio = stride // 2
            pad = -memory.shape[1] % ratio
            groups = F.pad(memory, (0, 0, 0, pad)).unflatten(1, (-1, ratio))
            counts = F.pad(valid, (0, pad)).unflatten(1, (-1, ratio)).sum(dim=-1)
            pooled = groups.sum(dim=2) / counts.clamp_min(1)[..., None]
            local = self.context(pooled.transpose(1, 2)).transpose(1, 2)
            offset, width, relevance, quality = self.head(self.norm(pooled + F.gelu(local))).unbind(-1)
            left = stride * torch.arange(pooled.shape[1], device=pooled.device, dtype=pooled.dtype)
            right = torch.minimum(left + stride, torch.minimum(2 * lengths, duration)[:, None])
            points = (left + right) / 2
            center = points + offset * stride
            half_width = F.softplus(width) * stride
            raw_spans = torch.stack((center - half_width, center + half_width), dim=-1)
            spans = raw_spans.clamp_min(0).minimum(duration[:, None, None])
            level_outputs.append(dict(
                spans=spans, raw_spans=raw_spans, valid=counts > 0, points=points,
                relevance=relevance, quality=quality,
                strides=left.new_full(left.shape, stride),
                levels=torch.full_like(left, level, dtype=torch.long),
            ))
        return {key: torch.cat([out[key] for out in level_outputs],
                               dim=0 if key in ("strides", "levels") else 1)
                for key in level_outputs[0]}


def assign_targets(output, truth, valid_truth, scale):
    """中心采样与尺度区间决定一对多监督，重叠冲突优先较短真值。

    Args:
        output (dict): DenseLocator 输出，时间单位为秒。
        truth (Float[Tensor, "B G 2"]): 所有 GT 起止时间，padding 至 G。
        valid_truth (Bool[Tensor, "B G"]): True 为有效 GT。
        scale (str): 单尺度时取消时长分流，使 2 秒层可负责任意长度。

    Returns:
        tuple: 正位置掩码 [B P]，每个位置选择的真值 [B P 2]。
    """
    width = truth[..., 1] - truth[..., 0]
    distance = (output["points"][..., None] - truth.mean(dim=-1)[:, None]).abs()
    step = output["strides"][None, :, None]
    eligible = output["valid"][..., None] & valid_truth[:, None, :]
    if scale == "multi":
        ranges = truth.new_tensor(LENGTH_RANGES)[output["levels"]]
        in_range = (width[:, None, :] >= ranges[None, :, 0, None]) & (
            width[:, None, :] <= ranges[None, :, 1, None])
        eligible = eligible & in_range
    radius = torch.minimum(width[:, None, :] / 2, 1.5 * step)
    matches = eligible & (distance <= radius)
    # 极窄、截尾或对齐到网格外的 GT 仍从其尺度中选择最近的有效点。
    nearest = distance.masked_fill(~eligible, float("inf")).argmin(dim=1, keepdim=True)
    fallback = torch.zeros_like(matches).scatter_(1, nearest, True)
    matches = matches | (eligible & fallback & ~matches.any(dim=1, keepdim=True))
    selected = width[:, None, :].expand_as(distance).masked_fill(~matches, float("inf")).argmin(dim=-1)
    target = truth.gather(1, selected[..., None].expand(-1, -1, 2))
    return matches.any(dim=-1), target


def dense_losses(output, truth, valid_truth, scale):
    """联合优化 focal 相关性、尺度归一 L1/GIoU 和正位置 IoU 质量。

    Args:
        output (dict): 密集预测与有效位置掩码。
        truth (Float[Tensor, "B G 2"]): 以秒存储的真值区间。
        valid_truth (Bool[Tensor, "B G"]): True 为有效 GT。
        scale (str): single 或 multi。

    Returns:
        dict: 四项按查询平均的损失。
    """
    positive, target = assign_targets(output, truth, valid_truth, scale)
    def overlap(spans):
        intersection = (spans[..., 1].minimum(target[..., 1]) -
                        spans[..., 0].maximum(target[..., 0])).clamp_min(0)
        union = (spans.diff().squeeze(-1) + target.diff().squeeze(-1) - intersection).clamp_min(1e-6)
        return intersection / union, union

    # 原始端点保留越界预测的回归梯度；置信度刻画实际裁剪后的窗口。
    spans = output["raw_spans"]
    iou, union = overlap(spans)
    enclosure = (spans[..., 1].maximum(target[..., 1]) -
                 spans[..., 0].minimum(target[..., 0])).clamp_min(1e-6)
    giou = iou - (enclosure - union) / enclosure
    logits = output["relevance"]
    probability_error = torch.where(positive, (-logits).sigmoid(), logits.sigmoid())
    relevance = F.binary_cross_entropy_with_logits(logits, positive.to(logits.dtype), reduction="none")
    relevance = relevance * probability_error.square() * torch.where(positive, 0.25, 0.75)
    quality = F.binary_cross_entropy_with_logits(
        output["quality"], overlap(output["spans"].detach())[0], reduction="none",
    )
    l1 = (spans - target).abs().sum(dim=-1) / output["strides"]
    # 每个查询独立归一，避免长区间或粗细层分配数改变查询权重。
    normalizer = positive.sum(dim=1).clamp_min(1)
    return {
        "relevance": ((relevance * output["valid"]).sum(dim=1) / normalizer).mean(),
        "l1": ((l1 * positive).sum(dim=1) / normalizer).mean(),
        "giou": (((1 - giou) * positive).sum(dim=1) / normalizer).mean(),
        "quality": ((quality * positive).sum(dim=1) / normalizer).mean(),
    }
