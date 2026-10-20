import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import pywt
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import cv2

def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    """创建小波分解和重建滤波器"""
    w = pywt.Wavelet(wave)

    # 分解滤波器
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 重建滤波器
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    """执行小波变换"""
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_wavelet_transform(x, filters):
    """执行逆小波变换"""
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)


def wavelet_transform_init(filters):
    """小波变换自动求导函数"""

    class WaveletTransform(Function):
        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = inverse_wavelet_transform(grad_output, filters)
            return grad, None

    return WaveletTransform().apply


def inverse_wavelet_transform_init(filters):
    """逆小波变换自动求导函数"""

    class InverseWaveletTransform(Function):
        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = inverse_wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = wavelet_transform(grad_output, filters)
            return grad, None

    return InverseWaveletTransform().apply


class _ScaleModule(nn.Module):
    """可学习的缩放模块"""

    def __init__(self, shape, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full(shape, init_scale))

    def forward(self, x):
        return x * self.scale


class WTConv2d(nn.Module):
    """小波变换卷积模块"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 bias=True,
                 wt_levels=3,
                 wt_type='db1',
                 use_separate_convs=False):
        super().__init__()

        # 网络参数
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.use_separate_convs = use_separate_convs

        # 初始化小波滤波器
        self.wt_filter, self.iwt_filter = create_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 初始化变换函数
        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        # # 基础卷积层
        # self.base_conv = nn.Conv2d(
        #     in_channels, in_channels, kernel_size,
        #     padding='same', stride=1, dilation=1,
        #     groups=in_channels, bias=bias)
        # self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # # 小波域卷积层
        # self.wavelet_convs = nn.ModuleList([
        #     nn.Conv2d(
        #         in_channels * 4, in_channels * 4, kernel_size,
        #         padding='same', stride=1, dilation=1,
        #         groups=in_channels * 4, bias=False)
        #     for _ in range(self.wt_levels)
        # ])
        # self.wavelet_scale = nn.ModuleList([
        #     _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
        #     for _ in range(self.wt_levels)
        # ])

        # # 下采样处理
        # if self.stride > 1:
        #     self.stride_filter = nn.Parameter(
        #         torch.ones(in_channels, 1, 1, 1),
        #         requires_grad=False)
        #     self.do_stride = lambda x: F.conv2d(
        #         x, self.stride_filter, bias=None,
        #         stride=self.stride, groups=in_channels)
        # else:
        #     self.do_stride = None

    def forward(self, x):
        # 初始化各层级存储
        x_ll_in_levels = []
        x_all_12 = []
        x_h_in_levels = []
        reconstructed_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        # 前向小波变换
        for i in range(self.wt_levels):
            # 记录当前形状
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # 处理奇数尺寸
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 执行小波变换
            curr_x = self.wt_function(curr_x_ll)
            """经过这句代码之后反回的是小波变换的结果，返回的也是四个分量上的结果，
            其中第一个分量上是低频信息
            """
            # 可视化当前层的小波变换结果
            save_wavelet_transform(curr_x, i)
            curr_x_ll = curr_x[:, :, 0, :, :]

            # 处理频带数据
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            """这句代码将所有的通道的频域分量都放在了一起
            """
            if not self.use_separate_convs:
                curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
                x_all_12.append(curr_x_tag)

            # 保存中间结果
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        # 保存克隆结果
        saved_x1_ll_in_levels = [x.clone() for x in x_ll_in_levels]
        saved_x_h_in_levels_1 = [x.clone() for x in x_h_in_levels]

        # 反向重建过程
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            # 执行逆变换
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
            reconstructed_levels.append(next_x_ll)



        return {
            'saved_x1_ll_in_levels': saved_x1_ll_in_levels,
            'saved_x_h_in_levels_1': saved_x_h_in_levels_1,
            'reconstructed_levels': reconstructed_levels,
            'x_all_12': x_all_12
        }


def save_wavelet_components(output_dict, original_image, output_dir):
    """保存所有层级的小波变换结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存原始图像（去除批次维度）
    save_tensor_as_image(original_image.squeeze(0), os.path.join(output_dir, 'original.png'))

    # 遍历每个层级
    for level in range(len(output_dict['saved_x1_ll_in_levels'])):
        level_dir = os.path.join(output_dir, f'level_{level + 1}')
        os.makedirs(level_dir, exist_ok=True)

        # 保存低频分量 (LL)，去除批次维度
        ll = output_dict['saved_x1_ll_in_levels'][level].squeeze(0)  # [1, C, H, W] → [C, H, W]
        save_tensor_as_image(ll, os.path.join(level_dir, 'LL.png'))

        # 保存高频分量 (LH, HL, HH)，去除批次维度
        hf_components = output_dict['saved_x_h_in_levels_1'][level].squeeze(0)  # [1, C, 3, H, W] → [C, 3, H, W]
        for i, name in enumerate(['LH', 'HL', 'HH']):
            hf = hf_components[:, i, :, :]  # [C, H, W]
            save_tensor_as_image(hf, os.path.join(level_dir, f'{name}.png'), is_high_freq=True)

    # 保存重建结果，去除批次维度
    for i, recon in enumerate(output_dict['reconstructed_levels']):
        save_tensor_as_image(recon.squeeze(0), os.path.join(output_dir, f'reconstructed_level_{i + 1}.png'))
def save_tensor_as_image(tensor, save_path, is_high_freq=False):
    """
    将张量保存为图像文件
    tensor: [C, H, W] 或 [H, W]
    """
    # 转换为 numpy 数组
    if tensor.dim() == 3:
        img = tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    else:
        img = tensor.cpu().numpy()

    # 处理高频分量
    if is_high_freq:
        img = np.abs(img)  # 取绝对值
        img = (img / img.max()) * 255  # 归一化到 0-255

    # 转换为 PIL 图像并保存
    if img.shape[-1] == 1:  # 单通道
        img = img.squeeze(-1).astype(np.uint8)
    else:  # 多通道 (RGB)
        img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

    Image.fromarray(img).save(save_path)


def save_wavelet_transform(curr_x, level, save_dir="/opt/data/private/xj/my_experience/sence_MCD/wavele"):
    """保存小波变换的四个子带为图片"""

    os.makedirs(save_dir, exist_ok=True)


    b, c, _, h, w = curr_x.shape  # 获取形状

    # 提取四个分量
    LL = curr_x[:, :, 0, :, :].squeeze().detach().cpu().numpy()
    LH = curr_x[:, :, 1, :, :].squeeze().detach().cpu().numpy()
    HL = curr_x[:, :, 2, :, :].squeeze().detach().cpu().numpy()
    HH = curr_x[:, :, 3, :, :].squeeze().detach().cpu().numpy()

    # 归一化到 0-255，避免负值
    def normalize(img):
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return img.astype(np.uint8)

    LL = normalize(LL).transpose(1, 2, 0)
    LH = normalize(LH).transpose(1, 2, 0)
    HL = normalize(HL).transpose(1, 2, 0)
    HH = normalize(HH).transpose(1, 2, 0)

    # 保存图像，并检查是否成功
    for name, img in zip(["LL", "LH", "HL", "HH"], [LL, LH, HL, HH]):
        path = f"{save_dir}/level_{level}_{name}.png"
        success = cv2.imwrite(path, img)
        if not success:
            print(f"Error: Failed to save {name} image at {path}")
    print(f"Saved wavelet level {level} images to {save_dir}")
# ================== 使用示例 ==================
if __name__ == "__main__":
    # 配置参数
    image_path = "/opt/data/private/xj/most_new_experience/data/dataset/T1.png"  # 替换为实际路径1

    output_dir = '/opt/data/private/xj/my_experience/sence_MCD/dwt_results'
    os.makedirs(output_dir, exist_ok=True)
    wt_levels = 3  # 分解层级数
    wt_type = 'db1'  # 小波类型

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]

    # 初始化小波变换模块
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wt_conv = WTConv2d(
        in_channels=3,  # 输入通道数（RGB为3，灰度图为1）
        out_channels=3,
        wt_levels=wt_levels,
        wt_type=wt_type
    ).to(device)

    # 执行变换
    with torch.no_grad():
        output_dict = wt_conv(image_tensor.to(device))

    # 保存结果
    save_wavelet_components(
        output_dict,
        image_tensor.squeeze(0),  # 原始图像 [C, H, W]
        output_dir
    )
    print(f"结果已保存至: {os.path.abspath(output_dir)}")



