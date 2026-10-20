import torch.nn as nn
import torch
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


class OptimizedAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction_ratio

        # 特征投影层
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_kv = nn.Conv2d(channels, 2 * channels, 1)

        # 低秩压缩
        self.compress = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.LayerNorm([channels // reduction_ratio, 1, 1]),
            nn.GELU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1)
        )

    def forward(self, feat_current, feat_other):
        B, C, H, W = feat_current.shape
        N = H * W

        # 投影特征
        q = self.to_q(feat_current).view(B, C, N)  # [B, C, N]
        k, v = self.to_kv(feat_other).chunk(2, dim=1)  # [B, C, N] each

        # 线性注意力核函数（近似点积）
        k = F.softmax(k, dim=-1)  # 归一化键向量
        context = torch.einsum('bcn,bcv->bnv', k, v)  # [B, N, C]

        # 聚合上下文
        attn_out = torch.einsum('bcn,bnv->bcv', q, context)  # [B, C, N]
        attn_out = attn_out.view(B, C, H, W)

        # 低秩增强
        attn_out = attn_out + self.compress(attn_out)
        return attn_out + feat_current  # 残差连接


class WindowCrossAttention(nn.Module):
    """窗口级别的跨特征图注意力"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat_1, feat_2):
        """
        feat_1: 第一张特征图 [B, C, M, M]
        feat_2: 第二张特征图 [B, C, M, M]
        """
        B, C, M, _ = feat_1.shape
        Q = self.query(feat_1).view(B, C, -1).permute(0, 2, 1)  # [B, M*M, C]
        K = self.key(feat_2).view(B, C, -1)                     # [B, C, M*M]
        V = self.value(feat_2).view(B, C, -1).permute(0, 2, 1)  # [B, M*M, C]

        attn = torch.bmm(Q, K) / (C ** 0.5)  # [B, M*M, M*M]
        attn = self.softmax(attn)
        out = torch.bmm(attn, V)  # [B, M*M, C]
        out = out.permute(0, 2, 1).view(B, C, M, M)  # [B, C, M, M]

        return out + feat_1  # 残差连接

class DynamicWindowCrossAttention(nn.Module):
    """适应任意尺寸的窗口跨特征图注意力"""
    def __init__(self, channels, base_size=128):
        super().__init__()
        self.base_size = base_size
        self.cross_attn = WindowCrossAttention(channels)

    def get_dynamic_window_size(self, H, W):
        """ 计算动态窗口大小 M，使其尽可能整除 H, W """
        return min(self.base_size, H, W)

    def forward(self, feat_1, feat_2):
        """
        feat_1: 第一张特征图 [B, C, H, W]
        feat_2: 第二张特征图 [B, C, H, W]
        """
        B, C, H, W = feat_1.shape
        M = self.get_dynamic_window_size(H, W)

        # 计算 padding
        pad_h = (M - H % M) % M
        pad_w = (M - W % M) % M

        # 填充到 M 的整数倍
        feat_1 = F.pad(feat_1, (0, pad_w, 0, pad_h), mode="reflect")
        feat_2 = F.pad(feat_2, (0, pad_w, 0, pad_h), mode="reflect")

        Hp, Wp = feat_1.shape[2], feat_1.shape[3]

        # 窗口划分
        feat_1_windows = feat_1.unfold(2, M, M).unfold(3, M, M)  # [B, C, num_h, num_w, M, M]
        feat_2_windows = feat_2.unfold(2, M, M).unfold(3, M, M)

        num_h, num_w = feat_1_windows.shape[2], feat_1_windows.shape[3]
        feat_1_windows = feat_1_windows.contiguous().view(B, C, num_h * num_w, M, M)  # [B, C, num_windows, M, M]
        feat_2_windows = feat_2_windows.contiguous().view(B, C, num_h * num_w, M, M)

        # 逐个窗口计算交叉注意力
        out_windows = []
        for i in range(num_h * num_w):
            out_windows.append(self.cross_attn(feat_1_windows[:, :, i], feat_2_windows[:, :, i]))

        out_windows = torch.stack(out_windows, dim=2)  # [B, C, num_windows, M, M]

        # 还原窗口
        out_windows = out_windows.view(B, C, num_h, num_w, M, M)
        out = out_windows.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hp, Wp)  # [B, C, Hp, Wp]

        # 去除 padding，恢复原始尺寸
        out = out[:, :, :H, :W]
        return out