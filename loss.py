from pytorch_ssim import ssim
import torch
import torch.nn.functional as F
import torch.nn as nn
def hybrid_loss(output, target, alpha=0.8, beta=0.2):
    mse = F.mse_loss(output, target)
    ssim_loss = 1 - ssim(output, target)
    return  alpha*mse + beta * ssim_loss

# 对比损失（InfoNCE）
# 对比损失（InfoNCE）
def contrastive_loss(ll1, ll2, temperature=0.1, batch_size=512):
    C, H, W = ll1.shape
    ll1 = F.normalize(ll1.view(C, -1), dim=0)  # [C, HW]
    ll2 = F.normalize(ll2.view(C, -1), dim=0)  # [C, HW]
    HW = H * W
    labels = torch.arange(HW, device=ll1.device)

    total_loss = 0
    for i in range(0, HW, batch_size):
        end_i = i + batch_size
        ll1_block = ll1.t()[i:end_i]  # [batch_size, C]
        logits = torch.mm(ll1_block, ll2) / temperature  # [batch_size, HW]
        loss = F.cross_entropy(logits, labels[i:end_i])
        total_loss += loss * (end_i - i)

    return total_loss / HW


def mmd_loss(z1, z2, kernel_mul=5.0, kernel_num=3, fix_sigma=None):
    """
    计算多核最大均值差异（MK-MMD）损失

    参数:
      z1, z2: 张量，形状为 (B, C, H, W)
      kernel_mul: 多核宽度调节因子
      kernel_num: 使用的高斯核数量
      fix_sigma: 固定的 sigma 值；若为 None，则采用中位数启发式计算

    返回:
      mmd: 标量，两个分布之间的 MMD 损失
    """
    batch_size = z1.size(0)
    z1 = z1.reshape(batch_size, -1)  # (B, C*H*W)
    z2 = z2.reshape(batch_size, -1)

    # 合并两个分布的样本 -> (2B, C*H*W)
    z = torch.cat([z1, z2], dim=0)

    # 预计算所有样本之间的成对欧氏距离的平方，避免循环中重复计算
    pairwise_dists = torch.cdist(z, z, p=2) ** 2  # (2B, 2B)

    # 如果未提供固定 sigma，则使用非零距离的中位数作为基础 sigma
    if fix_sigma is not None:
        base_sigma = fix_sigma
    else:
        nonzero_dists = pairwise_dists[pairwise_dists > 0]
        base_sigma = torch.median(nonzero_dists) if nonzero_dists.numel() > 0 else 1.0

    # 构造多个 sigma 值，以 base_sigma 为中心扩展
    sigmas = [base_sigma * (kernel_mul ** (i - kernel_num // 2)) for i in range(kernel_num)]

    # 利用预先计算的 pairwise_dists，为每个 sigma 计算高斯核，然后累加
    kernel_val = 0
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma ** 2)
        kernel_val += torch.exp(-gamma * pairwise_dists)

    # 多核 MMD：将多个核矩阵累加（也可以考虑加权平均）
    kernel_matrix = kernel_val

    # 将核矩阵划分为对应子矩阵
    K_XX = kernel_matrix[:batch_size, :batch_size]  # z1 与 z1 之间
    K_YY = kernel_matrix[batch_size:, batch_size:]  # z2 与 z2 之间
    K_XY = kernel_matrix[:batch_size, batch_size:]  # z1 与 z2 之间

    # 根据公式计算 MMD 损失
    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd


def low_freq_alignment(coeffs_x, coeffs_y, eps=1e-8, scale_weights=None, loss_type='mse'):
    """
    带能量归一化的低频对齐损失
    coeffs_x: 模态X的各尺度小波系数列表，每个元素形状为 (B, C, H, W)
    coeffs_y: 模态Y的对应系数列表
    eps: 防止除0的小常数
    scale_weights: 列表，对每个尺度加权，如果为 None，则默认各尺度权重相同
    loss_type: 'mse' 或 'l1'
    """
    total_loss = 0.0
    num_levels = len(coeffs_x)
    # 如果没有指定权重，默认各尺度权重为1
    if scale_weights is None:
        scale_weights = [1.0] * num_levels

    for (a_x, a_y), weight in zip(zip(coeffs_x, coeffs_y), scale_weights):
        # 能量归一化：对整个张量归一化
        energy_x = torch.norm(a_x, p=2) + eps
        energy_y = torch.norm(a_y, p=2) + eps
        a_x_norm = a_x / energy_x
        a_y_norm = a_y / energy_y

        # 计算损失
        if loss_type == 'mse':
            level_loss = F.mse_loss(a_x_norm, a_y_norm)
        elif loss_type == 'l1':
            level_loss = F.l1_loss(a_x_norm, a_y_norm)
        else:
            raise ValueError("Unsupported loss_type: choose 'mse' or 'l1'")

        total_loss += weight * level_loss

    return total_loss