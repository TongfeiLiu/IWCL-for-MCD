import torch
import numpy as np
from PIL import Image
from pytorch_wavelets import DWTForward
import torch.nn.functional as F

def save_tensor_as_image(tensor, filename, mode='high'):
    """增强版保存函数：支持高频（带符号）和低频（无符号）"""
    tensor = tensor.squeeze(0).cpu().numpy()  # 去除 batch 维度
    tensor = np.transpose(tensor, (1, 2, 0))  # (H, W, C)

    if mode == 'high':
        # 高频：将 [-k, k] 缩放到 [0, 255]，保留符号可视化
        k = np.abs(tensor).max()  # 动态范围
        tensor = (tensor + k) / (2 * k) * 255  # [-k, k] → [0, 255]
    else:
        # 低频：直接归一化到 [0, 255]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255

    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    img = Image.fromarray(tensor.squeeze())  # 去除单通道维度（若适用）
    img.save(filename)
    print(f"✅ 已保存: {filename}")


def guided_filter_torch(I, p, r, eps):
    """引导滤波函数（保持不变）"""
    kernel_size = 2 * r + 1
    mean_I = F.avg_pool2d(I, kernel_size, stride=1, padding=r, count_include_pad=False)
    mean_p = F.avg_pool2d(p, kernel_size, stride=1, padding=r, count_include_pad=False)
    mean_Ip = F.avg_pool2d(I * p, kernel_size, stride=1, padding=r, count_include_pad=False)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = F.avg_pool2d(I * I, kernel_size, stride=1, padding=r, count_include_pad=False)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = F.avg_pool2d(a, kernel_size, stride=1, padding=r, count_include_pad=False)
    mean_b = F.avg_pool2d(b, kernel_size, stride=1, padding=r, count_include_pad=False)
    return mean_a * I + mean_b


# 主处理流程
def process_image(image_path):
    # 读取图像并转换为张量
    image = Image.open(image_path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).cuda()  # (1, 3, H, W)

    # 逐级小波分解（保存每一级的低频）
    yl_list = [image]  # yl_list[0] = 原图
    yh_list = []  # 保存每一级的高频

    for level in range(3):
        dwt = DWTForward(J=1, wave='haar', mode='symmetric').cuda()  # 每次分解一级
        yl, yh = dwt(yl_list[-1])  # 分解当前低频
        yl_list.append(yl)
        yh_list.append(yh)
        save_tensor_as_image(yl, f"low_freq_level{level + 1}.png", mode='low')

    # 对每一级高频进行引导滤波（使用该级的低频作为引导图）
    for level in range(3):
        current_yl = yl_list[level + 1]  # 当前级的低频（yl_list[1]对应第1级）
        current_yh = yh_list[level]  # 当前级的高频

        # 提取三个方向的分量（cH, cV, cD）
        cH = current_yh[0][:, :, 0, :, :]  # (1, C, H, W)
        cV = current_yh[0][:, :, 1, :, :]
        cD = current_yh[0][:, :, 2, :, :]

        # 使用当前级的低频作为引导图去噪高频
        cH_filtered = guided_filter_torch(current_yl, cH, r=4, eps=0.01)
        cV_filtered = guided_filter_torch(current_yl, cV, r=4, eps=0.01)
        cD_filtered = guided_filter_torch(current_yl, cD, r=4, eps=0.01)

        # 保存去噪后的高频
        save_tensor_as_image(cH, f"filtered_cH_level{level + 1}.png", mode='high')
        save_tensor_as_image(cV, f"filtered_cV_level{level + 1}.png", mode='high')
        save_tensor_as_image(cD, f"filtered_cD_level{level + 1}.png", mode='high')


# 执行示例
process_image("/opt/data/private/xj/my_experience/sence_MCD/data/dataset/T1.png")