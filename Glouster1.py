import os
import numpy as np
from PIL import Image
from skimage.transform import resize
from ARCov import ARConv
import torch.nn as nn
from evaluation import Evaluation
from skimage.filters.thresholding import threshold_otsu,threshold_local
from skimage import io
from pytorch_wavelets import DWTForward, DWTInverse
from preprocess import process_images
from preprocess_1 import process_images_1
from wtconv.loss import ssim,hybrid_loss,low_freq_alignment,mmd_loss,contrastive_loss
import torch
import torch.nn.functional as F
from attention import OptimizedAttention,DynamicWindowCrossAttention
from wtconv.guide import MultiDirectionGuidanceModule
import torchvision.transforms as transforms



class WaveletDecomposer(nn.Module):
    def __init__(self, J=2, wt_type='db1', mode='symmetric', enhance_factor=1.5):
        super(WaveletDecomposer, self).__init__()
        self.J = J
        self.wt_type = wt_type
        self.mode = mode
        self.enhance_factor = enhance_factor
        self.dwt = DWTForward(J=1, wave=wt_type, mode=mode)
    def forward(self, x):
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
            'final':current_LL ,
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

    def forward(self, current_ll, prev_ll_list, epoch=500,hw_range=[1, 18]):
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
        feat = self.arconv(fused, epoch,hw_range)
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
                nn.Conv2d(shared_channels, shared_channels, kernel_size=3 ,padding=1),
                nn.Mish()

            )
                for d in self.directions
        })

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

    def forward(self, x1, x2,epoch):
        wave_output1 = self.wavelet(x1)
        wave_output2 = self.wavelet(x2)
        LL1, LL2 = [], []
        cnn_HH1, cnn_HH2 = [], []
        ori_ll1, ori_ll2 = [], []
        prev_ll_list1, prev_ll_list2 = [], []
        hf_restruct_h1,hf_restruct_h2=[], []

        for level in range(self.levels):
            # 低频处理
            ll1 = wave_output1['LL'][level]
            ll2 = wave_output2['LL'][level]

            cnn_LL1 = self.shared_conv[level](ll1, prev_ll_list1,epoch=300,hw_range=[1,35])
            cnn_LL2 = self.shared_conv[level](ll2, prev_ll_list2,epoch=300,hw_range=[1,35])
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for l in [1,2, 4,5]:
        for i in range(0, 5):

            def load_images(data_path):
                a = io.imread(os.path.join(data_path, 'T1-Img17-Bc.png'))

                b = io.imread(os.path.join(data_path, 'T2-Img17-A.png'))


                # 如果 a 和 b 是单通道，则扩展为 3 通道
                if a.ndim == 2:
                    a = np.stack([a] * 3, axis=-1)
                if b.ndim == 2:
                    b = np.stack([b] * 3, axis=-1)

                    # 加载 GT 参考图像，并转换为 NumPy 数组
                ref = Image.open(os.path.join(data_path, 'Img17-C.png'))
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

                unique_values = np.unique(ref)
                if not np.array_equal(unique_values, [0, 255]):
                    ref[ref < 200] = 0
                    ref[ref >= 200] = 255

                # 确保 ref 是 uint8 类型
                # ref = ref.astype(np.uint8)*255

                return torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), ref
            load_data = os.path.join(root_dir + load_data_dir)
            img_t1, img_t2, ref= load_images(load_data)
            # img_t1 = img_t1.permute(2, 0, 1)
            # img_t2 = img_t2.permute(2, 0, 1)


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

            height, width,channel, = img_t1.shape
            # img_t1 = img_t1.float()
            # img_t2 = img_t2.float()
            # img_t1 = (img_t1.permute(2, 0, 1))
            # img_t2 = (img_t2.permute(2, 0, 1))
            # img_t1 = normalize_single_img(img_t1)
            # img_t2 = normalize_single_img(img_t2)

            # img_t1 = img_t1.unsqueeze(0)
            # img_t2 = img_t2.unsqueeze(0)
            img_t1, img_t2 = process_images(img_t1, img_t2, type_a='opt', type_b='opt')
            # t1 = (t1.permute(2, 0, 1)).unsqueeze(0)
            # t2 = (t2.permute(2, 0, 1)).unsqueeze(0)

            img_t1 = (img_t1.permute(2, 0, 1)).unsqueeze(0)
            img_t2 = (img_t2.permute(2, 0, 1)).unsqueeze(0)
            # img_t1 = normalize_img_per_channel(img_t1)
            # img_t2 = normalize_img_per_channel(img_t2)
            file_path = os.path.join(result_folder, f'G1_{i}',f'scale_{l}')
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            # 初始化 WaveletBranch
            WC_e = Wave_Con_Net(in_channels=3, wt_levels=l,sar_wt_type='db1').to(device)
            # 定义逆小波变换卷积层
            inverse_wt1 = WaveletDecoder(in_channels=3, wt_levels=l, wt_type='db1').to(device)
            inverse_wt2 = WaveletDecoder(in_channels=3, wt_levels=l, wt_type='db1').to(device)
            optimizer_e = torch.optim.Adam(WC_e.parameters(), lr=0.01,weight_decay=0.001)
            optimizer_d_a = torch.optim.Adam(inverse_wt1.parameters(), lr=0.01,weight_decay=0.001 )
            optimizer_d_b = torch.optim.Adam(inverse_wt2.parameters(), lr=0.01,weight_decay=0.001 )


            weight_path = os.path.join(file_path, 'init_weight.pth')
            model_dict = {
                    'encoder': WC_e.state_dict(),
                'decoder_a': inverse_wt1.state_dict(),
                'decoder_b': inverse_wt2.state_dict()
            }
            torch.save(model_dict, weight_path)


            def train(epoch):
                WC_e.train()
                inverse_wt1.train()
                inverse_wt2.train()

                optimizer_e.zero_grad()
                optimizer_d_a.zero_grad()
                optimizer_d_b.zero_grad()

                encode_output = WC_e(img_t1, img_t2,epoch)

                # 对T1进行逆小波变换
                wt_output_dict1 = {
                    'll': encode_output['cnn_LL1'],
                    'hh': encode_output['hf_recon_h1'],
                    'shapes': encode_output['wave_output1']['pad_params']
                }
                output_x1 = inverse_wt1(wt_output_dict1)

                # 对T2进行逆小波变换
                wt_output_dict2 = {
                    'll': encode_output['cnn_LL2'],
                    'hh': encode_output['hf_recon_h2'],
                    'shapes': encode_output['wave_output1']['pad_params']
                }
                output_x2 = inverse_wt2(wt_output_dict2)

                # 保存重建结果
                save_tensor_as_image(output_x1['final'], os.path.join(file_path,'train_reconstructed'),
                                     epoch, image_type="T1", prefix="train_restruct")
                save_tensor_as_image(output_x2['final'], os.path.join(file_path,'train_reconstructed'),
                                     epoch, image_type="T2", prefix="train_restruct")

                # 重建损失
                recon_loss1 = F.mse_loss(output_x1['final'], img_t1)
                recon_loss2 = F.mse_loss(output_x2['final'], img_t2)
                # 低频共性损失
                def align_loss(ll1_list, ll2_list):
                    loss = 0
                    for ll1, ll2 in zip(ll1_list, ll2_list):
                        loss += F.mse_loss(ll1, ll2)
                    return loss
                alig_loss_dell = align_loss(output_x1['multi_scales'], output_x2['multi_scales'])
                # 总损失
                total_loss =recon_loss1+recon_loss2+alig_loss_dell
                total_loss.backward()
                optimizer_e.step()
                optimizer_d_a.step()
                optimizer_d_b.step()
                # 保存模型
                weight_path = os.path.join(file_path, 'weight_epoch_%d.pth' % (epoch))
                model_dict = {
                    'encoder': WC_e.state_dict(),
                    'decoder_a': inverse_wt1.state_dict(),
                    'decoder_b': inverse_wt2.state_dict(),
                }
                torch.save(model_dict, weight_path)

                print(f'Epoch [{epoch}], Loss: {total_loss}')


            def test(epoch):
                WC_e.eval()
                inverse_wt1.eval()
                inverse_wt2.eval()
                with torch.no_grad():
                    encode_output = WC_e(img_t1, img_t2,epoch)
                    # 创建逆小波变换输出字典
                    wt_output_dict1 = {'ll': encode_output['cnn_LL1'],
                                       'hh': encode_output['hf_recon_h1'],
                                       'shapes': encode_output['wave_output1']['pad_params']}
                    output_x1 = inverse_wt1(wt_output_dict1)
                    # 创建T2逆小波变换输出字典
                    wt_output_dict2 = {'ll': encode_output['cnn_LL2'],
                                       'hh': encode_output['hf_recon_h2'],
                                       'shapes': encode_output['wave_output1']['pad_params']}

                    output_x2 = inverse_wt2(wt_output_dict2)
                    #保存重建图像
                    # 假设测试过程中生成的 T1 和 T2 重建图像分别为 reconstructed_x2_T1 和 reconstructed_x2_T2
                    save_tensor_as_image(output_x1['final'],
                                         os.path.join(file_path,'test_reconstructed'), epoch,
                                         image_type="T1", prefix="test_restruct")
                    save_tensor_as_image(output_x2['final'],
                                         os.path.join(file_path,'test_reconstructed'), epoch,
                                         image_type="T2", prefix="test_restruct")

                    # ll1 = encode_output['cnn_LL2'][2]  # 假设形状为 [N, C, H1, W1]
                    # ll2 = encode_output['cnn_LL1'][2]  # 假设形状为 [N, C, H2, W2]
                    #
                    # # 调整尺寸使两者一致（若需要）
                    # if ll1.shape[-2:] != ll2.shape[-2:]:
                    #     ll2 = F.interpolate(ll2, size=ll1.shape[-2:], mode='bilinear', align_corners=False)
                    #
                    # # 计算逐元素平方差，并在通道维度求和得到单通道差异图
                    # diff_squared = (ll1 - ll2) ** 2
                    # diff_sum = torch.sum(diff_squared, dim=1, keepdim=True)  # 形状 [N, 1, H, W]
                    # diff_map = torch.sqrt(diff_sum + 1e-6)  # 添加epsilon在sqrt内部
                    #
                    # # 上采样到原始图像尺寸
                    # diff_map = F.interpolate(
                    #     diff_map,
                    #     size=img_t1.shape[-2:],
                    #     mode='bilinear',
                    #     align_corners=False
                    # )
                    class MultiLevelDiff(nn.Module):
                        def __init__(self, levels):
                            super().__init__()
                            self.weights = nn.Parameter(torch.ones(levels))  # 可学习权重

                            # 低通滤波卷积核（均值滤波器）
                            self.smooth_filter = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to(device)
                            self.smooth_filter.weight.data.fill_(1.0 / 9)  # 初始化为均值滤波
                            # self.smooth_filter.requires_grad_(True)  # 让它固定，不参与训练

                        def forward(self, encode_output, img_shape):
                            diff_maps = []
                            for level in range(len(self.weights)):
                                # diff = 1 - F.cosine_similarity(encode_output['cnn_LL1'][level], encode_output['cnn_LL2'][level], dim=1)

                                diff_squared = (encode_output['cnn_LL1'][level] - encode_output['cnn_LL2'][level]) ** 2
                                diff_sum = torch.sum(diff_squared, dim=1, keepdim=True)
                                diff_map = torch.sqrt(diff_sum + 1e-6)

                                # # 低通滤波
                                diff = self.smooth_filter(diff_map)

                                # 统一尺寸
                                diff = F.interpolate(diff, size=img_shape, mode='bilinear', align_corners=False)
                                diff_maps.append(diff)

                            # softmax 归一化权重
                            weights = torch.softmax(self.weights, dim=0)

                            # 加权融合
                            return sum(w * d for w, d in zip(weights, diff_maps))

                    Mdiff = MultiLevelDiff(levels=l)
                    diff_map = Mdiff(encode_output, img_t1.shape[-2:])
                    diff_map = diff_map.squeeze(0)



                    diff_map_min = diff_map.min()

                    diff_map_max = diff_map.max()
                    if diff_map_max - diff_map_min != 0:
                        diff_map_nor = (diff_map - diff_map_min) / (diff_map_max - diff_map_min)
                    else:
                        small_constant = 1e-10
                        diff_map_nor = (diff_map - diff_map_min) / (diff_map_max - diff_map_min + small_constant)

                    filename1 = f"DI_iter_{epoch}.png"
                    filename2 = f"CM_iter_{epoch}.png"
                    diff_map_save = diff_map_nor * 255
                    diff_map_save = diff_map_save.cpu().numpy()
                    if ref.shape[1] >= 1000:
                        diff_map_save = diff_map_save.squeeze(0)
                        diff_map_save_pil = Image.fromarray(diff_map_save.astype(np.uint8))
                            # 如果是三维 RGB 图像


                        # 使用 transform
                        transform = transforms.Compose([
                            transforms.Resize((2000, 2000)),  # 将图像缩小到指定的尺寸
                        ])

                        # 应用 transform
                        diff_map_save_size_pil = transform(diff_map_save_pil)

                        # 转换回 NumPy 数组
                        diff_map_save = np.array(diff_map_save_size_pil)
                        diff_map_save = np.expand_dims(diff_map_save, axis=0)




                    io.imsave(os.path.join(file_path, filename1), diff_map_save.astype(np.uint8))
                    thre1 = threshold_otsu(diff_map_save)
                    CM1 = (diff_map_save >= thre1) * 255
                    CM1 = CM1[0]
                    io.imsave(os.path.join(file_path, filename2), (CM1).astype(np.uint8))
                    Indicators1 = Evaluation(ref, CM1)
                    OA1, kappa1, AA1 = Indicators1.Classification_indicators()
                    P1, R1, F11 = Indicators1.ObjectExtract_indicators()
                    TP1, TN1, FP1, FN1 = Indicators1.matrix()

                    file_name = "eval.txt"
                    # # 将内容写入文件
                    file_path_txt = os.path.join(file_path, file_name)
                    val_acc = open(file_path_txt, 'a')
                    val_acc.write(
                        '===============================Parameters settings==============================\n')
                    val_acc.write('=== epoch={} || \n'.format(epoch))
                    val_acc.write('Domain t1:\n')
                    val_acc.write('TP={} || TN={} || FP={} || FN={}\n'.format(TP1, TN1, FP1, FN1))
                    val_acc.write("\"OA\":\"" + "{}\"\n".format(OA1))
                    val_acc.write("\"Kappa\":\"" + "{}\"\n".format(kappa1))
                    val_acc.write("\"AA\":\"" + "{}\"\n".format(AA1))
                    val_acc.write("\"Precision\":\"" + "{}\"\n".format(P1))
                    val_acc.write("\"Recall\":\"" + "{}\"\n".format(R1))
                    val_acc.write("\"F1\":\"" + "{}\"\n".format(F11))
                    val_acc.close()

            for epoch in range(100):
                train_loss = train(epoch)
                test(epoch)