import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from pytorch_wavelets import DWTForward
import numpy as np

class WaveletDecomposer:
    def __init__(self, J=2, wt_type='db1', mode='symmetric', device='cpu'):
        self.J = J
        self.wt_type = wt_type
        self.mode = mode
        self.device = device
        self.dwt = DWTForward(J=1, wave=wt_type, mode=mode)

    def pad_to_divisible(self, x, div=2):
        h, w = x.shape[-2:]
        pad_h = (div - h % div) % div
        pad_w = (div - w % div) % div
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x

    def decompose(self, x):
        LL_list, HF_list = [], []
        current_LL = x
        for level in range(self.J):
            current_LL = self.pad_to_divisible(current_LL, div=2)
            yl, yh = self.dwt(current_LL)
            LL_list.append(yl)
            HF_list.append(yh[0])
            current_LL = yl
        return LL_list, HF_list

def save_results(LL_list, HF_list, save_dir, prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)
    for level, (LL, HF) in enumerate(zip(LL_list, HF_list)):
        vutils.save_image(
            LL, os.path.join(save_dir, f"{prefix}_LL_level{level+1}.png"), normalize=True
        )
        for i, name in enumerate(["LH", "HL", "HH"]):
            vutils.save_image(
                HF[:, :, i, :, :],
                os.path.join(save_dir, f"{prefix}_HF_{name}_level{level+1}.png"),
                normalize=True
            )

def save_HF_pseudo_color(save_dir, prefix="sample", J=3, gamma=0.5, out_dir="./pseudo_HF"):
    directions = ["LH", "HL", "HH"]
    os.makedirs(out_dir, exist_ok=True)
    pseudo_paths = []

    for level in range(1, J+1):
        imgs = []
        for name in directions:
            file_path = os.path.join(save_dir, f"{prefix}_HF_{name}_level{level}.png")
            if not os.path.exists(file_path):
                imgs.append(np.zeros((256,256), dtype=np.uint8))
                continue
            img = Image.open(file_path).convert("L")
            img_np = np.array(img, dtype=np.float32)
            img_np = np.abs(img_np)
            img_np = img_np ** gamma
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_np = (img_np * 255).astype(np.uint8)
            imgs.append(img_np)

        rgb = np.stack(imgs, axis=-1)
        out_path = os.path.join(out_dir, f"{prefix}_HF_pseudo_level{level}.png")
        Image.fromarray(rgb).save(out_path)
        pseudo_paths.append(out_path)
        print(f"保存伪彩色高频图: {out_path}")
    return pseudo_paths

def merge_LL_HF(LL_list, pseudo_paths, merge_dir, prefix="sample"):
    """将每层 LL 和伪彩色 HF 拼接到同一张图保存"""
    os.makedirs(merge_dir, exist_ok=True)
    for level, (LL, hf_path) in enumerate(zip(LL_list, pseudo_paths)):
        LL_img = LL[0].permute(1,2,0).cpu().numpy()  # [H,W,C]
        LL_img = (LL_img - LL_img.min()) / (LL_img.max() - LL_img.min() + 1e-8)
        LL_img = (LL_img * 255).astype(np.uint8)
        LL_img = Image.fromarray(LL_img)

        HF_img = Image.open(hf_path)
        # 调整 HF 尺寸与 LL 一致
        HF_img = HF_img.resize(LL_img.size, Image.BICUBIC)

        # 左右拼接
        merged = Image.new("RGB", (LL_img.width + HF_img.width, LL_img.height))
        merged.paste(LL_img.convert("RGB"), (0,0))
        merged.paste(HF_img, (LL_img.width,0))

        save_path = os.path.join(merge_dir, f"{prefix}_merged_level{level+1}.png")
        merged.save(save_path)
        print(f"保存合并图像: {save_path}")

if __name__ == "__main__":
    img_path = "./sence_MCD/data/dataset/T2.png"
    save_dir = "./wavelet_outT2"
    pseudo_dir = "./pseudo_HF_T2"
    merge_dir = "./merged_T2"
    J = 3

    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    x = transform(img).unsqueeze(0)

    decomposer = WaveletDecomposer(J=J, wt_type='db1', device='cpu')
    LL_list, HF_list = decomposer.decompose(x)

    prefix_name = os.path.splitext(os.path.basename(img_path))[0]
    save_results(LL_list, HF_list, save_dir, prefix=prefix_name)
    pseudo_paths = save_HF_pseudo_color(save_dir, prefix=prefix_name, J=J, gamma=0.5, out_dir=pseudo_dir)
    merge_LL_HF(LL_list, pseudo_paths, merge_dir, prefix=prefix_name)

    print(f"分解结果已保存到 {save_dir}")
    print(f"伪彩色高频已保存到 {pseudo_dir}")
    print(f"合并图像已保存到 {merge_dir}")


