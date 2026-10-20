import torch
import torch.nn as nn
import torch.nn.functional as F
from wtconv.util import wavelet
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    import copy  # 导入copy模块

    def forward(self, x1, x2):
        x1_ll_in_levels = []  # 存储x1的低频信息
        x2_ll_in_levels = []  # 存储x2的低频信息
        x_h_in_levels_1 = []  # 存储x1的高频信息
        x_h_in_levels_2 = []  # 存储x2的高频信息
        shapes_in_levels = []  # 用于存储图像尺寸信息

        curr_x_ll_1 = x1
        curr_x_ll_2 = x2

        # 对两个模态进行小波变换
        for i in range(self.wt_levels):  # 开始进行小波变换
            curr_shape_1 = curr_x_ll_1.shape
            curr_shape_2 = curr_x_ll_2.shape
            shapes_in_levels.append(curr_shape_1)

            # 填充，确保卷积正常进行
            if (curr_shape_1[2] % 2 > 0) or (curr_shape_1[3] % 2 > 0):
                curr_pads = (0, curr_shape_1[3] % 2, 0, curr_shape_1[2] % 2)
                curr_x_ll_1 = F.pad(curr_x_ll_1, curr_pads)
                curr_x_ll_2 = F.pad(curr_x_ll_2, curr_pads)

            # 小波变换
            curr_x_1 = wavelet.wavelet_transform(curr_x_ll_1, self.wt_filter)
            curr_x_2 = wavelet.wavelet_transform(curr_x_ll_2, self.wt_filter)

            # 提取低频部分（保留第一个通道作为低频信息）
            curr_x_ll_1 = curr_x_1[:, :, 0, :, :]  # 只保留低频分量
            curr_x_ll_2 = curr_x_2[:, :, 0, :, :]  # 只保留低频分量

            shape_x_1 = curr_x_1.shape
            shape_x_2 = curr_x_2.shape

            curr_x_tag_1 = curr_x_1.reshape(shape_x_1[0], shape_x_1[1] * 4, shape_x_1[3], shape_x_1[4])
            curr_x_tag_2 = curr_x_2.reshape(shape_x_2[0], shape_x_2[1] * 4, shape_x_1[3], shape_x_2[4])

            # 通过卷积处理低频部分
            curr_x_tag_1 = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag_1))
            curr_x_tag_2 = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag_2))

            curr_x_tag_1 = curr_x_tag_1.reshape(shape_x_1)
            curr_x_tag_2 = curr_x_tag_2.reshape(shape_x_2)

            # 存储x1和x2的低频部分
            x1_ll_in_levels.append(curr_x_tag_1[:, :, 0, :, :])  # 存储x1的低频部分
            x2_ll_in_levels.append(curr_x_tag_2[:, :, 0, :, :])  # 存储x2的低频部分
            # 存储x1和x2的高频部分
            x_h_in_levels_1.append(curr_x_tag_1[:, :, 1:4, :, :])  # 存储x1的高频部分
            x_h_in_levels_2.append(curr_x_tag_2[:, :, 1:4, :, :])  # 存储x2的高频部分

        # 用clone()替代deepcopy来深拷贝每个张量
        saved_x1_ll_in_levels = [x.clone() for x in x1_ll_in_levels]
        saved_x2_ll_in_levels = [x.clone() for x in x2_ll_in_levels]
        saved_x_h_in_levels_1 = [x.clone() for x in x_h_in_levels_1]
        saved_x_h_in_levels_2 = [x.clone() for x in x_h_in_levels_2]

        next_x_ll_1 = 0
        next_x_ll_2 = 0

        # 进行逆小波变换，逐层重建图像
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll_1 = x1_ll_in_levels.pop()
            curr_x_ll_2 = x2_ll_in_levels.pop()
            curr_x_h_1 = x_h_in_levels_1.pop()
            curr_x_h_2 = x_h_in_levels_2.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll_1 = curr_x_ll_1 + next_x_ll_1
            curr_x_ll_2 = curr_x_ll_2 + next_x_ll_2

            curr_x_1 = torch.cat([curr_x_ll_1.unsqueeze(2), curr_x_h_1], dim=2)
            curr_x_2 = torch.cat([curr_x_ll_2.unsqueeze(2), curr_x_h_2], dim=2)

            next_x_ll_1 = wavelet.inverse_wavelet_transform(curr_x_1, self.iwt_filter)
            next_x_ll_2 = wavelet.inverse_wavelet_transform(curr_x_2, self.iwt_filter)

            next_x_ll_1 = next_x_ll_1[:, :, :curr_shape[2], :curr_shape[3]]
            next_x_ll_2 = next_x_ll_2[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag_1 = next_x_ll_1
        x_tag_2 = next_x_ll_2

        # 融合低频和卷积特征
        x1 = self.base_scale(self.base_conv(x1)) + x_tag_1
        x2 = self.base_scale(self.base_conv(x2)) + x_tag_2

        if self.do_stride is not None:
            x1 = self.do_stride(x1)
            x2 = self.do_stride(x2)

        # 返回小波变换后的低频信息和高频信息，以及重建后的低频信息
        return saved_x1_ll_in_levels, saved_x2_ll_in_levels, saved_x_h_in_levels_1, saved_x_h_in_levels_2, x1,  x2


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)





def save_wavelet_components(output_dict, original_image1, original_image2, output_dir):
    """保存所有层级的小波变换结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存原始图像
    save_tensor_as_image(original_image1.squeeze(0), os.path.join(output_dir, 'original_x1.png'))
    save_tensor_as_image(original_image2.squeeze(0), os.path.join(output_dir, 'original_x2.png'))

    # 遍历每个层级
    for level in range(len(output_dict['saved_x1_ll_in_levels'])):
        level_dir = os.path.join(output_dir, f'level_{level + 1}')
        os.makedirs(level_dir, exist_ok=True)

        # 保存模态1的低频和高频分量
        save_single_modality(
            output_dict['saved_x1_ll_in_levels'][level],
            output_dict['saved_x_h_in_levels_1'][level],
            level_dir, modality='x1'
        )

        # 保存模态2的低频和高频分量
        save_single_modality(
            output_dict['saved_x2_ll_in_levels'][level],
            output_dict['saved_x_h_in_levels_2'][level],
            level_dir, modality='x2'
        )

    # 保存重建结果
    save_tensor_as_image(output_dict['x1'].squeeze(0), os.path.join(output_dir, 'reconstructed_x1.png'))
    save_tensor_as_image(output_dict['x2'].squeeze(0), os.path.join(output_dir, 'reconstructed_x2.png'))


def save_single_modality(ll_tensor, hf_tensor, save_dir, modality='x1'):
    """保存单个模态的分量"""
    # 保存低频分量
    save_tensor_as_image(ll_tensor.squeeze(0), os.path.join(save_dir, f'{modality}_LL.png'))

    # 保存高频分量
    for i, name in enumerate(['LH', 'HL', 'HH']):
        hf = hf_tensor[:, :, i].squeeze(0)  # 提取单个高频分量 [C, H, W]
        save_tensor_as_image(hf, os.path.join(save_dir, f'{modality}_{name}.png'), is_high_freq=True)


def save_tensor_as_image(tensor, save_path, is_high_freq=False):
    """将张量保存为图像文件"""
    # 处理批次维度
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # [1, C, H, W] → [C, H, W]

    # 转换为 numpy 数组
    if tensor.dim() == 3:
        img = tensor.permute(1, 2, 0).cpu().numpy()  # CHW → HWC
    else:
        img = tensor.cpu().numpy()

    # 处理高频分量
    if is_high_freq:
        img = np.abs(img)
        img = (img / img.max()) * 255 if img.max() > 0 else img  # 避免除以零

    # 转换为 PIL 图像并保存
    if img.shape[-1] == 1:  # 单通道
        img = img.squeeze(-1).astype(np.uint8)
    else:  # 多通道
        img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

    Image.fromarray(img).save(save_path)


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 配置参数
    image_path1 = "/opt/data/private/Datasets/MCD/MCD_Scene/MTWHU/train/rgb/1001.tif"  # 模态1路径
    image_path2 = "/opt/data/private/Datasets/MCD/MCD_Scene/MTWHU/train/sar/1001.png"  # 模态2路径
    output_dir = "/opt/data/private/xj/my_experience/sence_MCD/wavelet_results"
    wt_levels = 3  # 分解层级数
    wt_type = 'db3'  # 小波类型

    # 加载图像
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 假设输入范围为[-1,1]
    ])

    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')
    image_tensor1 = transform(image1).unsqueeze(0)  # [1, C, H, W]
    image_tensor2 = transform(image2).unsqueeze(0)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WTConv2d(
        in_channels=3,  # RGB图像通道数
        out_channels=3,
        wt_levels=wt_levels,
        wt_type=wt_type
    ).to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model(image_tensor1.to(device), image_tensor2.to(device))
        output_dict = {
            'saved_x1_ll_in_levels': outputs[0],
            'saved_x2_ll_in_levels': outputs[1],
            'saved_x_h_in_levels_1': outputs[2],
            'saved_x_h_in_levels_2': outputs[3],
            'x1': outputs[4],
            'x2': outputs[5]
        }

    # 保存结果
    save_wavelet_components(
        output_dict,
        image_tensor1,  # 原始模态1
        image_tensor2,  # 原始模态2
        output_dir
    )
    print(f"结果已保存至: {os.path.abspath(output_dir)}")