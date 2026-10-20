import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18_NoDownsample(nn.Module):
    def __init__(self, in_channels=1,out_channels=3):
        super().__init__()
        resnet = models.resnet18(pretrained=False)  # 加载 ResNet18 结构

        # 修改 conv1：stride=1，防止尺寸缩小
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu

        # 移除 maxpool，防止尺寸缩小
        # self.maxpool = resnet.maxpool  # 删除

        # 让所有层的 stride=1，防止尺寸变化
        self.layer1 = resnet.layer1  # 这一层本来就是 stride=1，无需修改
        self.layer2 = self._make_layer_no_downsample(resnet.layer2)  # 确保 stride=1
        self.layer3 = self._make_layer_no_downsample(resnet.layer3)  # 确保 stride=1
        self.layer4 = self._make_layer_no_downsample(resnet.layer4)  # 确保 stride=1
        # **新增 1x1 卷积，把通道数变成 3**
        self.final_conv = nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0)

    def _make_layer_no_downsample(self, layer):
        """修改 ResNet Block，去掉 stride=2，防止下采样"""
        for block in layer:
            if hasattr(block, 'conv1'):
                block.conv1.stride = (1, 1)  # 让 stride=1，防止特征图尺寸变化
            if block.downsample is not None:
                # **去掉 downsample 中的 stride**
                block.downsample[0] = nn.Conv2d(
                    block.downsample[0].in_channels,
                    block.downsample[0].out_channels,
                    kernel_size=1,
                    stride=1,  # 设为 1，防止尺寸变化
                    bias=False
                )
        return layer

    def forward(self, x):
        x = self.conv1(x)  # 7x7 卷积（已修正 stride=1）
        x = self.bn1(x)
        x = self.relu(x)
        # self.maxpool(x)  # 这里不再做最大池化

        x = self.layer1(x)  # 经过 ResNet 各个 Block
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_conv(x)

        return x  # 输出特征图，尺寸不变

# # 测试
# model = ResNet18_NoDownsample(in_channels=6)
# inp = torch.randn(1, 6, 277, 277)  # 输入任意尺寸
# out = model(inp)
# print(out.shape)  # 预期输出: (1, 512, 277, 277)，不改变尺寸





class SimpleResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet9_NoDownsample(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        """
        一个极简版的 ResNet-风格网络，不进行下采样，保持输入尺寸不变，
        并使用更少的残差块和较小的通道数以降低内存占用。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # 仅使用两个残差块
        self.resblock1 = SimpleResidualBlock(base_channels)
        self.resblock2 = SimpleResidualBlock(base_channels)

        # 最后 1x1 卷积调整通道数
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        return x





