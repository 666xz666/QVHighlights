"""时间锚点query与逐层残差解码；复用上游层，不修改基线Transformer。"""
import math

import torch
from torch import nn


class TemporalAnchors(nn.Module):
    """可学习中心/宽度及几何位置编码；两组实验仅宽度初值不同。"""

    def __init__(self, hidden_dim, mode):
        """Args:
            hidden_dim (int): decoder隐藏维度，须为4的倍数。
            mode (str): uniform或multiscale；固定10个query。
        """
        super().__init__()
        if mode not in ('uniform', 'multiscale') or hidden_dim % 4:
            raise ValueError('Expected uniform/multiscale and hidden_dim divisible by 4')
        # 步骤1：共享中心初始化，保证消融只改变尺度；各尺度覆盖整段视频。
        centers = torch.cat([(torch.arange(n) + .5) / n for n in (5, 3, 2)])
        widths = torch.tensor([.04] * 5 + [.12] * 3 + [.32] * 2)
        if mode == 'uniform':
            widths.fill_(.12)
        self.logits = nn.Parameter(torch.logit(torch.stack((centers, widths), -1)))
        self.hidden_dim = hidden_dim
        self.position = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim))

    def forward(self, anchors):
        """将当前几何坐标转换为decoder query位置向量。

        Args:
            anchors (Float[Tensor, "Q B 2"]): 归一化中心和宽度。

        Returns:
            Float[Tensor, "Q B D"]: 正余弦编码经MLP映射的位置向量。
        """
        count = self.hidden_dim // 2
        index = torch.arange(count, device=anchors.device, dtype=anchors.dtype)
        scale = 10000 ** (2 * torch.div(index, 2, rounding_mode='floor') / count)
        phases = anchors[..., None] * (2 * math.pi) / scale
        encoded = torch.stack((phases[..., 0::2].sin(), phases[..., 1::2].cos()), -1)
        return self.position(encoded.flatten(-3))


def decode_anchors(model, src, padding_mask, pos):
    """使用当前锚点驱动每层query，并输出各层修正后的区间。

    Args:
        model (QMTDETR): 含既有encoder/decoder、共享span头与时间锚点。
        src (Float[Tensor, "B L D"]): 视频与文本拼接特征。
        padding_mask (Bool[Tensor, "B L"]): True代表padding，与上游契约一致。
        pos (Float[Tensor, "B L D"]): 既有位置编码。

    Returns:
        tuple: hs [layers,B,Q,D]、memory [B,L,D]、spans [layers,B,Q,2]。
    """
    # 步骤1：保持encoder路径及padding语义不变。
    transformer = model.transformer
    pos = pos.transpose(0, 1)
    memory = transformer.encoder(src.transpose(0, 1),
                                 src_key_padding_mask=padding_mask, pos=pos)
    content = model.query_embed.weight[:, None].expand(-1, src.shape[0], -1)
    output = torch.zeros_like(content)
    anchors = model.qmt_anchors.logits.sigmoid()[:, None].expand(-1, src.shape[0], -1)
    hidden, spans = [], []
    # 步骤2：保留原query内容向量，新增当前几何编码；共享回归头学习相对修正。
    for layer in transformer.decoder.layers:
        output = layer(output, memory, memory_key_padding_mask=padding_mask,
                       pos=pos, query_pos=content + model.qmt_anchors(anchors))
        normalized = transformer.decoder.norm(output)
        predicted = (torch.logit(anchors.clamp(1e-5, 1 - 1e-5))
                     + model.span_embed(normalized)).sigmoid()
        hidden.append(normalized.transpose(0, 1))
        spans.append(predicted.transpose(0, 1))
        # 各层都有原aux监督；阻断跨层坐标梯度，保留decoder内容梯度。
        anchors = predicted.detach()
    return torch.stack(hidden), memory.transpose(0, 1), torch.stack(spans)
