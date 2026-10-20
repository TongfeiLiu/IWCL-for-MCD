import os
import numpy as np
from PIL import Image
import os
import scipy.io as sio
from ARCov import ARConv
import torch.nn as nn
from evaluation import Evaluation
from skimage.filters.thresholding import threshold_otsu, threshold_local
from skimage import io
from pytorch_wavelets import DWTForward, DWTInverse
from preprocess import process_images
import torch
import torch.nn.functional as F
from attention import OptimizedAttention, DynamicWindowCrossAttention
from wtconv.guide import MultiDirectionGuidanceModule
import torchvision.transforms as transforms
import cv2
import torchvision.utils as vutils


class WaveletDecomposer(nn.Module):
    def __init__(self, J=2, wt_type='db1', mode='symmetric', enhance_factor=1.5, save_dir="./wavelet_results"):
        super(WaveletDecomposer, self).__init__()
        self.J = J
        self.wt_type = wt_type
        self.mode = mode
        self.enhance_factor = enhance_factor
        self.dwt = DWTForward(J=1, wave=wt_type, mode=mode)
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def forward(self, x,save=False, prefix="sample"):
        LL_list, HF_list = [], []
        current_LL = x
        pad_params = []
        for level in range(self.J):
            h, w = current_LL.shape[-2:]
            pad_h = (2 - h % 2) % 2
            pad_w = (2 - w % 2) % 2
            if pad_h > 0 or pad_w > 0:
                current_LL = F.pad(current_LL, (0, pad_w, 0, pad_h), mode='reflect')
                pad_params.append((pad_h, pad_w))
            else:
                pad_params.append((0, 0))

            yl, yh = self.dwt(current_LL)
            LL_list.append(yl)
            HF_list.append(yh[0])
            current_LL = yl
            # 保存结果
            if save and self.save_dir is not None:
                # 保存低频 LL
                vutils.save_image(
                    yl, os.path.join(self.save_dir, f"{prefix}_LL_level{level + 1}.png"), normalize=True
                )
                # 高频有三部分 (LH, HL, HH)
                for i in range(3):
                    vutils.save_image(
                        yh[0][:, i, :, :],
                        os.path.join(self.save_dir, f"{prefix}_HF{i + 1}_level{level + 1}.png"),
                        normalize=True
                    )
        return {
            'LL': LL_list,
            'HF': HF_list,
            'pad_params': pad_params
        }


class WaveletDecoder(nn.Module):
    def __init__(self, in_channels, wt_levels, wt_type='db1', mode='symmetric', fuse_levels=2):
        super(WaveletDecoder, self).__init__()
        self.in_channels = in_channels
        self.levels = wt_levels
        self.wt_type = wt_type
        self.mode = mode
        self.fuse_levels = fuse_levels
        self.idwt = DWTInverse(wave=self.wt_type, mode=self.mode)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, 3, kernel_size=3, padding=1),
                nn.Mish(),
                nn.Conv2d(3, 3, kernel_size=3, padding=1),
            )
            for _ in range(self.levels)
        ])

    def forward(self, decomposed_features):
        low_freq_features = decomposed_features['ll']
        high_freq_features = decomposed_features['hh']
        pad_params = decomposed_features['shapes']
        current_LL = low_freq_features[-1]
        multi_scale_recons = []
        for idx, level in enumerate(reversed(range(self.levels))):
            high_freq = high_freq_features[level]
            pad_h, pad_w = pad_params[level]  # 假设 pad_params 是按层级存储的
            current_LL = self.idwt((current_LL, [high_freq]))
            # 去除当前层级的 padding（如果存在）
            if pad_h > 0 or pad_w > 0:
                current_LL = current_LL[..., :current_LL.size(-2) - pad_h, :current_LL.size(-1) - pad_w]
            # 残差连接（通道数已匹配）
            residual = self.conv_blocks[level](current_LL)
            current_LL = current_LL + residual
            # 保存重建结果用于监督
            multi_scale_recons.append(current_LL)
        return {
            'final': current_LL,
            'multi_scales': list(reversed(multi_scale_recons))  # 从浅到深排列
        }


def save_tensor_as_image(tensor, folder, epoch, image_type="T1", prefix="reconstructed"):
    # 确保文件夹存在
    os.makedirs(folder, exist_ok=True)

    # 去除 batch 维度并归一化到 [0, 1]
    tensor = tensor.detach().cpu().squeeze(0)  # 去掉 batch 维度
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # 归一化到 [0, 1]

    # 生成文件名，使用 epoch 和 image_type（T1 或 T2）作为文件名的一部分
    filename = os.path.join(folder, f"{prefix}_epoch_{epoch}_{image_type}.png")

    # 根据通道数保存图像
    if tensor.shape[0] == 3:
        # RGB 图像
        img = Image.fromarray((tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    else:
        # 单通道（灰度）图像
        img = Image.fromarray((tensor.numpy()[0] * 255).astype(np.uint8), mode='L')

    # 保存图像
    img.save(filename)
    print(f"图像已保存至: {filename}")


class SharedConv(nn.Module):
    def __init__(self, level, in_channels=3, growth_rate=3, base_channels=3, max_fuse_levels=1):
        super(SharedConv, self).__init__()
        self.level = level
        self.max_fuse_levels = max_fuse_levels
        self.base_channels = base_channels
        self.in_channels = in_channels

        # 历史特征处理策略
        self.num_fuse = min(level, max_fuse_levels)

        # 核心修改：根据通道数决定是否压缩
        if base_channels > 16 and max_fuse_levels > 0:
            # 需要压缩的情况
            self.compress = True
            self.historical_channels_per_layer = base_channels // 2
            self.prev_conv = nn.Sequential(
                nn.Conv2d(base_channels, self.historical_channels_per_layer, 1),
                nn.Mish(inplace=True)
            )
        else:
            # 不压缩的情况
            self.compress = False
            self.historical_channels_per_layer = base_channels
            self.prev_conv = None  # 无压缩操作

        # 计算总历史通道数
        self.historical_channels = self.historical_channels_per_layer * self.num_fuse
        self.input_channels = in_channels + self.historical_channels

        # 特征提取层
        self.arconv = ARConv(self.input_channels, growth_rate)
        self.combined_channels = self.input_channels + growth_rate

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(self.combined_channels, growth_rate // 1, 1),
            nn.Conv2d(growth_rate // 1, 1, 1),
            nn.Mish()
        )

        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.combined_channels, base_channels, 3, padding=1),
            # nn.Mish()

        )

    def forward(self, current_ll, prev_ll_list, epoch=500, hw_range=[1, 18]):
        aligned_features = []
        for prev_ll in prev_ll_list[-self.max_fuse_levels:]:
            if len(aligned_features) >= self.num_fuse:
                break

            # 尺寸对齐
            if prev_ll.shape[-2:] != current_ll.shape[-2:]:
                prev_ll = F.interpolate(prev_ll, size=current_ll.shape[-2:],
                                        mode='bilinear', align_corners=False)

            # 通道压缩（仅在需要时）
            if self.prev_conv is not None:
                prev_ll = self.prev_conv(prev_ll)

            aligned_features.append(prev_ll)

        # 特征拼接
        fused = torch.cat([current_ll] + aligned_features, dim=1)

        # 特征提取与融合
        feat = self.arconv(fused, epoch, hw_range)
        feat_combined = torch.cat([fused, feat], dim=1)

        gate = self.fusion_gate(feat_combined)
        final_feature = gate * self.output_conv(feat_combined) + (1 - gate) * current_ll

        return final_feature


class HFConv(nn.Module):
    def __init__(self, in_channels, shared_channels=3):
        super().__init__()
        self.directions = ['LH', 'HL', 'HH']  # 高频方向
        self.shared_channels = shared_channels

        # 编码器
        self.shared_encoder = nn.ModuleDict({
            d: nn.Sequential(
                nn.Conv2d(in_channels, shared_channels, kernel_size=3, padding=1),
                nn.Mish(),
                nn.Conv2d(shared_channels, shared_channels, kernel_size=3, padding=1),
                nn.Mish()

            )
            for d in self.directions
        })
        # 为每个方向添加注意力模块
        # self.global_attn = nn.ModuleDict({
        #     d: OptimizedAttention(shared_channels) for d in self.directions
        # })
        # self.global_attn = OptimizedAttention(channels)
        # self.local_attn = DynamicWindowCrossAttention(channels)
        self.local_attn = nn.ModuleDict({
            d: DynamicWindowCrossAttention(shared_channels, base_size=64) for d in self.directions
        })

    def forward(self, high_t1, high_t2):
        B, C, D, H, W = high_t1.shape
        assert D == len(self.directions), f"方向数{D}与预设{len(self.directions)}不符"

        # 提取当前方向的高频特征
        fused_feats_t1 = {}
        fused_feats_t2 = {}
        rest_t1 = {}
        rest_t2 = {}

        for idx, d in enumerate(self.directions):
            # 提取当前方向子带
            subband_t1 = high_t1[:, :, idx, :, :].contiguous()
            subband_t2 = high_t2[:, :, idx, :, :].contiguous()

            # 共享编码
            encoded_t1 = self.shared_encoder[d](subband_t1)
            encoded_t2 = self.shared_encoder[d](subband_t2)
            local_feat = self.local_attn[d](encoded_t2, encoded_t1)
            fused_feats_t1[d] = local_feat
            fused_feats_t2[d] = local_feat
            rest_t1[d] = encoded_t1
            rest_t2[d] = encoded_t2
        # 重组输出：将每个方向的特征进行拼接或堆叠
        res_1 = torch.stack([rest_t1[d] for d in self.directions], dim=2)
        res_2 = torch.stack([rest_t2[d] for d in self.directions], dim=2)

        five_d1 = torch.stack([fused_feats_t1[d] for d in self.directions], dim=2)
        five_d2 = torch.stack([fused_feats_t2[d] for d in self.directions], dim=2)

        return res_1, res_2, five_d1, five_d2


class Wave_Con_Net(nn.Module):
    def __init__(self, in_channels=3, wt_levels=3, sar_wt_type='haar', fuse_levels=2):
        super().__init__()
        self.levels = wt_levels
        self.fuse_levels = fuse_levels
        self.wavelet = WaveletDecomposer(J=wt_levels, wt_type=sar_wt_type, mode='symmetric')
        self.shared_conv = nn.ModuleList([
            SharedConv(level=i, max_fuse_levels=fuse_levels)
            for i in range(self.levels)
        ])
        self.shared_convH = nn.ModuleList([
            HFConv(in_channels, shared_channels=3)
            for _ in range(self.levels)
        ])
        self.guidance_module = MultiDirectionGuidanceModule(in_channels)

    def forward(self, x1, x2, epoch):
        wave_output1 = self.wavelet(x1)
        wave_output2 = self.wavelet(x2)
        LL1, LL2 = [], []
        cnn_HH1, cnn_HH2 = [], []
        ori_ll1, ori_ll2 = [], []
        prev_ll_list1, prev_ll_list2 = [], []
        hf_restruct_h1, hf_restruct_h2 = [], []

        for level in range(self.levels):
            # 低频处理
            ll1 = wave_output1['LL'][level]
            ll2 = wave_output2['LL'][level]

            cnn_LL1 = self.shared_conv[level](ll1, prev_ll_list1, epoch=300, hw_range=[1, 35])
            cnn_LL2 = self.shared_conv[level](ll2, prev_ll_list2, epoch=300, hw_range=[1, 35])
            ori_ll1.append(cnn_LL1)
            ori_ll2.append(cnn_LL2)
            # 高频处理
            highs1 = wave_output1['HF'][level]
            highs2 = wave_output2['HF'][level]
            rest_1, rest_2, five_d1, five_d2 = self.shared_convH[level](highs1, highs2)
            # 特征增强
            L_enhanced1 = self.guidance_module(cnn_LL1, five_d1)
            L_enhanced2 = self.guidance_module(cnn_LL2, five_d1)
            prev_ll_list1.append(L_enhanced1)
            prev_ll_list2.append(L_enhanced2)
            if len(prev_ll_list1) > self.fuse_levels:
                prev_ll_list1 = prev_ll_list1[-self.fuse_levels:]
                prev_ll_list2 = prev_ll_list2[-self.fuse_levels:]

            LL1.append(L_enhanced1)
            LL2.append(L_enhanced2)
            hf_restruct_h1.append(five_d1)
            hf_restruct_h2.append(five_d1)
            cnn_HH1.append(rest_1)
            cnn_HH2.append(rest_2)

        return {
            'cnn_LL1': LL1,
            'cnn_LL2': LL2,
            'ori_ll1': ori_ll1,
            'ori_ll2': ori_ll2,
            'hf_recon_h1': hf_restruct_h1,
            'hf_recon_h2': hf_restruct_h2,
            'cnn_H1': cnn_HH1,
            'cnn_H2': cnn_HH2,
            'wave_output1': wave_output1,
            'wave_output2': wave_output2,
        }


if __name__ == "__main__":

    root_dir = '/opt/data/private/xj/my_experience/sence_MCD'
    load_data_dir = '/data/dataset'

    result_folder = '/opt/data/private/xj/my_experience/sence_MCD'
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(0, 1):

        def load_images(data_path):
            mat = sio.loadmat('/opt/data/private/xj/my_experience/sence_MCD/data/dataset/California.mat')
            data_variable1 = mat['image_t1']
            data_variable2 = mat['image_t2']
            ref = mat['gt'] * 255
            a = data_variable1[:, :, 1]
            b = data_variable2[:, :, 8]

            # 如果 a 和 b 是单通道，则扩展为 3 通道
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)
            if b.ndim == 2:
                b = np.stack([b] * 3, axis=-1)

                # 加载 GT 参考图像，并转换为 NumPy 数组
            ref = np.array(ref)  # (H, W) 或 (H, W, C)
            # 如果 ref 是三通道 (H, W, 3)，去掉通道但不合并
            if ref.ndim == 3:
                ref = ref[..., 0]  # 仅保留第一个通道

            # 将 ndarray 转换为 PILImage
            if a.shape[1] >= 1500:
                pil_a = Image.fromarray(a)
                pil_b = Image.fromarray(b)

                # # 创建一个 transform 对象，用于缩小图像
                transform = transforms.Compose([
                    transforms.Resize((1000, 1000)),  # 将图像缩小到指定的尺寸
                ])
                #
                #  应用 transform
                a = transform(pil_a)
                b = transform(pil_b)
                a = np.array(a)
                b = np.array(b)

            # 二值化处理
            unique_values = np.unique(ref)
            if not np.array_equal(unique_values, [0, 255]):
                ref[ref < 200] = 0
                ref[ref >= 200] = 255

            return torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), ref


        load_data = os.path.join(root_dir + load_data_dir)
        img_t1, img_t2, ref = load_images(load_data)


        def normalize_img_per_channel(image):
            # image: (B, C, H, W)
            min_val = image.amin(dim=(-2, -1), keepdim=True)
            max_val = image.amax(dim=(-2, -1), keepdim=True)
            eps = 1e-6
            return (image - min_val) / (max_val - min_val + eps)


        def normalize_single_img(img):
            # img: (C, H, W)
            min_val = img.amin(dim=(1, 2), keepdim=True)
            max_val = img.amax(dim=(1, 2), keepdim=True)
            eps = 1e-6
            return (img - min_val) / (max_val - min_val + eps)


        height, width, channel, = img_t1.shape
        # img_t1 = img_t1.float()
        # img_t2 = img_t2.float()
        # img_t1 = (img_t1.permute(2, 0, 1))
        # img_t2 = (img_t2.permute(2, 0, 1))
        # img_t1 = normalize_single_img(img_t1)
        # img_t2 = normalize_single_img(img_t2)

        # img_t1 = img_t1.unsqueeze(0)
        # img_t2 = img_t2.unsqueeze(0)
        img_t1, img_t2 = process_images(img_t1, img_t2, type_a='sar', type_b='opt')
        # t1 = (t1.permute(2, 0, 1)).unsqueeze(0)
        # t2 = (t2.permute(2, 0, 1)).unsqueeze(0)

        img_t1 = (img_t1.permute(2, 0, 1)).unsqueeze(0)
        img_t2 = (img_t2.permute(2, 0, 1)).unsqueeze(0)
        # img_t1 = normalize_img_per_channel(img_t1)
        # img_t2 = normalize_img_per_channel(img_t2)
        file_path = os.path.join(result_folder, f'California_{i}')
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # 初始化 WaveletBranch
        WC_e = Wave_Con_Net(in_channels=3, wt_levels=3, sar_wt_type='db1').to(device)
        # 定义逆小波变换卷积层
        inverse_wt1 = WaveletDecoder(in_channels=3, wt_levels=3, wt_type='db1').to(device)
        inverse_wt2 = WaveletDecoder(in_channels=3, wt_levels=3, wt_type='db1').to(device)
        optimizer_e = torch.optim.Adam(WC_e.parameters(), lr=0.01, weight_decay=0.001)
        optimizer_d_a = torch.optim.Adam(inverse_wt1.parameters(), lr=0.01, weight_decay=0.0001)
        optimizer_d_b = torch.optim.Adam(inverse_wt2.parameters(), lr=0.01, weight_decay=0.001)

        # 2. 加载保存的权重字典
        model_dict = torch.load('weight_epoch_93.pth', weights_only=True)


        # 3. 分别载入到各个模型中
        WC_e.load_state_dict(model_dict['encoder'])
        inverse_wt1.load_state_dict(model_dict['decoder_a'])
        inverse_wt2.load_state_dict(model_dict['decoder_b'])




        def gradcam_from_encode_output(model, encode_output, input_img, target_key='cnn_LL1', level=-2):
            """
            Grad-CAM for model outputs that are feature map lists (encode_output).

            Args:
                model: the model (used for hooks if needed)
                encode_output: dict, output from your forward()
                input_img: input image tensor, shape [B, C, H, W]
                target_key: key in encode_output to select feature maps (e.g., 'cnn_LL1')
                level: which level to use (-1 for last)

            Returns:
                heatmap: [B, H, W] normalized heatmap
            """
            # 取指定层的特征图
            target = encode_output[target_key][level]  # [B, C, H, W]
            target = target.requires_grad_(True)  # 确保可以计算梯度

            # hook 保存梯度
            gradients = []

            def save_grad(grad):
                gradients.append(grad)

            target.register_hook(save_grad)

            # backward 生成梯度
            model.zero_grad()
            # 对每个通道求和或简单使用 ones_like
            target.backward(torch.ones_like(target), retain_graph=True)

            grad = gradients[0]  # [B, C, H, W]

            # 权重：对 H,W 求均值
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

            # Grad-CAM: 权重 * 特征图
            cam = torch.sum(weights * target, dim=1)  # [B, H, W]

            # ReLU
            # cam = torch.(cam)


            # 归一化到 0~1
            cam_min, cam_max = cam.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0], \
            cam.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            # 转为 numpy
            cam_np = cam.detach().cpu().numpy()  # [B, H, W]

            # 可选：resize 到原始输入大小
            B, _, H, W = input_img.shape
            heatmaps = []
            for i in range(B):
                hmap = cv2.resize(cam_np[i], (W, H))
                heatmaps.append(hmap)
            heatmaps = np.stack(heatmaps, axis=0)  # [B, H, W]

            return heatmaps


        def test(epoch,  save_path='gradcam_results'):
            WC_e.eval()
            inverse_wt1.eval()
            inverse_wt2.eval()
            data_path='/opt/data/private/xj/my_experience/sence_MCD/data/dataset'

            a = io.imread(os.path.join(data_path, 'C_t2.png'))
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)

            data_variable1 = (torch.from_numpy(a).permute(2, 0, 1)).unsqueeze(0)




            # 前向
            encode_output = WC_e(img_t1, img_t2, epoch)

            # 获取 Grad-CAM 热力图
            heatmaps = gradcam_from_encode_output(WC_e, encode_output, data_variable1, target_key='cnn_LL2', level=2)
            # heatmaps: list of H x W, 已归一化到 [0,1]

            # 处理原图
            data_variable1 = (data_variable1.squeeze(0)).permute(1,2, 0)

            orig = data_variable1.cpu().detach().numpy()  # HWC
            # orig = (orig * 255).astype(np.uint8)
            if orig.shape[2] == 1:
                orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)


            # 处理热力图
            heatmap = heatmaps[0]  # H x W
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # 叠加
            overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)



            # 保存
            save_file = os.path.join(save_path, f'gradcam_epoch{2}.png')
            cv2.imwrite(save_file, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f'Grad-CAM 图像已保存到 {save_file}')



        for epoch in range(1):

            test(epoch)