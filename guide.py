import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAwareEnhancer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 通道注意力聚合（将C通道相似性转换为空间注意力）
        self.channel_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Mish()
        )
        # 可学习增强系数
        # 可学习增强系数，每个通道独立
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)

    def forward(self, low_feat, dir_sim_feat):
        """
        输入:
            low_feat: 低频特征 [B, C, H_low, W_low]
            dir_sim_feat: 单方向跨模态相似性特征 [B, C, H_high, W_high]
        输出:
            增强后的低频特征 [B, C, H_low, W_low]
        """
        # 生成通道聚合的注意力图
        attn = self.channel_attn(dir_sim_feat)  # [B, 1, H_high, W_high]
        return low_feat + self.alpha * attn


class MultiDirectionGuidanceModule(nn.Module):
    def __init__(self, in_channels, directions=3, shared_enhancer=False):
        super().__init__()
        self.directions = directions
        self.shared_enhancer = shared_enhancer

        if shared_enhancer:
            self.enhancer = ChannelAwareEnhancer(in_channels)
        else:
            self.enhancers = nn.ModuleList([
                ChannelAwareEnhancer(in_channels) for _ in range(directions)
            ])

        self.dir_weights = nn.Parameter(torch.ones(directions))
        self.fuse_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, low_feat, high_sim_attns):
        B, C, D, H, W = high_sim_attns.shape
        assert D == self.directions, f"方向数{D}与预设{self.directions}不符"

        enhanced_list = []
        for d in range(D):
            dir_feat = high_sim_attns[:, :, d, :, :]
            if self.shared_enhancer:
                enhanced = self.enhancer(low_feat, dir_feat)
            else:
                enhanced = self.enhancers[d](low_feat, dir_feat)
            enhanced_list.append(enhanced)

        weights = torch.softmax(self.dir_weights, dim=0)
        # # 打印当前每个方向的权重
        # if self.training:  # 只在训练时打印，避免验证时干扰
        #     print(f"[Guidance Weights] {weights.detach().cpu().numpy()}")
        fused = sum(w * feat for w, feat in zip(weights, enhanced_list))
        return self.fuse_conv(fused) + low_feat




if __name__ == "__main__":
    # 使用示例
    in_channels = 64  # 输入的通道数
    low_feat = torch.randn(1, in_channels, 32, 32)  # 低频特征示例
    high_feat = torch.randn(1, 3, 3, 32, 32)  # 五维高频特征示例 (B, 3, 3, H, W)

    guidance_module = MultiDirectionGuidanceModule(in_channels)
    enhanced_low_feat = guidance_module(low_feat, high_feat)

    print(enhanced_low_feat.shape)  # 输出增强后的低频特征
